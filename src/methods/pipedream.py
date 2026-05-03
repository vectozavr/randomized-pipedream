from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.methods.base import Method
from src.objectives.base import Objective
from src.state.microbatch import MicrobatchRuntime
from src.state.timeline import Timeline, num_microbatches_from_timeline
from src.state.trace import SimulationTrace
from src.state.versions import VersionTracker
from src.utils.partitioning import clone_stage_weights, combine_stage_weights


@dataclass
class PipeDreamMethod(Method):
    timeline: Timeline
    learning_rate: float
    training_batch_indices: list[int]
    init_stage_weights: list[np.ndarray] | None = None
    log_full_objective: bool = True
    log_forward_loss: bool = False
    name: str = "PipeDream"

    def run(self, objective: Objective) -> SimulationTrace:
        num_stages = objective.num_stages
        num_microbatches = num_microbatches_from_timeline(self.timeline)
        batches = objective.get_batches()

        if len(self.training_batch_indices) < num_microbatches:
            raise ValueError("training_batch_indices must cover all microbatches in the timeline")

        # Initialize stage weights. The stage_weights is a list of parameters for every stage.
        if self.init_stage_weights is None:
            stage_weights = objective.initial_stage_weights(mode="zeros")
        else:
            stage_weights = clone_stage_weights(self.init_stage_weights)

        def compute_full_grad_norm_sq(current_stage_weights: list[np.ndarray]) -> float:
            full_grad = objective.full_gradient(current_stage_weights)
            return float(sum(np.sum(g ** 2) for g in full_grad))

        forward_history_indices = -np.ones((num_microbatches, num_stages), dtype=int)
        history = [clone_stage_weights(stage_weights)]

        # Track versions of weights for staleness calculation and debugging.
        # In a real implementation, we would only need to track the version of the weights at the time
        # of the forward pass for each microbatch, but here we track all versions for easier debugging and analysis.
        versions = VersionTracker(num_stages=num_stages)
        micro: dict[int, MicrobatchRuntime] = {}

        # Track the version of weights used for forward and backward passes for each microbatch and stage,
        # to verify correct staleness and weight stashing behavior.
        forward_versions = -np.ones((num_microbatches, num_stages), dtype=int)
        backward_versions = -np.ones((num_microbatches, num_stages), dtype=int)
        backward_staleness = -np.ones((num_microbatches, num_stages), dtype=int)

        latest_forward_loss = np.nan
        forward_loss_trace: list[float] = []
        forward_loss_time_trace: list[int] = []

        initial_obj = objective.full_objective(stage_weights) if self.log_full_objective else latest_forward_loss
        objective_trace = [initial_obj]  # Track objective/loss at the end of each time step.
        block_update_objective: list[float] = []  # Track objective after each block update (backward pass on stage 0).
        stage_version_history = [versions.snapshot()]  # Track version history for debugging and analysis.
        time_completed = [0]
        completion_objective: list[float] = []


        initial_full_grad_norm_sq = compute_full_grad_norm_sq(stage_weights)
        full_grad_norm_sq_trace = [initial_full_grad_norm_sq]
        avg_full_grad_norm_sq_trace = [initial_full_grad_norm_sq]
        cumulative_full_grad_norm_sq = initial_full_grad_norm_sq
        grad_norm_trace: list[float] = []

        for t, ops in enumerate(self.timeline):  # Iterate over time steps and operations in the timeline.
            for stage, op in enumerate(ops):  # Iterate over stages & their corresponding operations at this time step.
                if op is None:
                    continue

                kind, mb = op

                if kind == "F":  # Forward pass on stage `stage` for microbatch `mb`.
                    if stage == 0:
                        # For the first stage, we need to initialize the microbatch runtime state
                        # and load the input batch.
                        batch_id = self.training_batch_indices[mb]
                        batch = batches[batch_id]
                        micro[mb] = MicrobatchRuntime(
                            batch_id=batch_id,
                            activations=[None] * (num_stages + 1),
                            grad_to_left=None,
                            stashed_weights=[None] * num_stages,
                            stashed_versions=[None] * num_stages,
                            loss_on_forward=None,
                        )
                        micro[mb].activations[0] = objective.initial_activation(batch)

                    # For subsequent stages, we assume the microbatch runtime state has been
                    # initialized during the forward pass of stage 0.
                    state = micro[mb]
                    batch = batches[state.batch_id]

                    current_history_index = len(history) - 1
                    forward_history_indices[mb, stage] = current_history_index

                    # We take the currently available weights (last version) and perform weights stashing.
                    w_used = stage_weights[stage].copy()
                    state.stashed_weights[stage] = w_used
                    state.stashed_versions[stage] = int(versions.versions[stage])
                    forward_versions[mb, stage] = int(versions.versions[stage])

                    # We take the output from the previous stage as input activation.
                    # For stage 0, this will be the initialized zeros.
                    activation_in = state.activations[stage]
                    if activation_in is None:
                        raise RuntimeError(f"Missing input activation for stage {stage}, microbatch {mb}")

                    # Run the forward pass for this stage and store the output activation for the next stage.
                    activation_out, _ = objective.forward_stage(
                        batch=batch,
                        stage=stage,
                        w_stage=w_used,
                        activation_in=activation_in,
                    )
                    state.activations[stage + 1] = activation_out

                    if stage == num_stages - 1:
                        # If this is the last stage, we also compute the loss & gradient and
                        # store it for later analysis.
                        loss, grad_out = objective.loss_and_output_grad(batch, activation_out)
                        state.loss_on_forward = loss
                        state.grad_to_left = grad_out
                        if self.log_forward_loss:
                            latest_forward_loss = float(loss)
                            forward_loss_trace.append(latest_forward_loss)
                            forward_loss_time_trace.append(t)

                elif kind == "B":  # Backward pass on stage `stage` for microbatch `mb`.

                    # For the backward pass, we assume the microbatch runtime state has been initialized during the
                    # forward pass of stage 0, and the forward pass for this stage has also been executed
                    # (so that weights are stashed and available).
                    state = micro[mb]
                    batch = batches[state.batch_id]

                    grad_out = state.grad_to_left
                    if grad_out is None:
                        raise RuntimeError(f"Backward on stage {stage}, microbatch {mb} has no incoming gradient.")

                    # We retrieve the stashed weights from the forward pass of this stage.
                    stashed = state.stashed_weights[stage]
                    if stashed is None:
                        raise RuntimeError(f"Missing stashed weights for stage {stage}, microbatch {mb}")

                    # having stashed weights, the batch, and incoming gradient, we can compute the backward pass
                    # for this stage to get the weight gradient and the gradient to send to the previous stage.
                    grad_w, grad_in = objective.backward_stage(
                        batch=batch,
                        stage=stage,
                        w_stage=stashed,
                        cache={"activation_in": state.activations[stage]},
                        grad_out=grad_out,
                    )

                    grad_norm_trace.append(float(np.linalg.norm(grad_w)))

                    stale_version = state.stashed_versions[stage]
                    if stale_version is None:
                        raise RuntimeError(f"Missing stashed version for stage {stage}, microbatch {mb}")

                    # Record the version of weights used for this backward pass,
                    # and the staleness compared to the latest version.
                    backward_versions[mb, stage] = int(stale_version)
                    backward_staleness[mb, stage] = int(versions.versions[stage] - stale_version)

                    # Update the weights for this stage using the computed gradient.
                    # In a real implementation, this would be done asynchronously and might involve
                    # locking or atomic updates, but here we do it synchronously for simplicity.
                    stage_weights[stage] = stage_weights[stage] - self.learning_rate * grad_w
                    versions.increment(stage)
                    history.append(clone_stage_weights(stage_weights))

                    # After the backward pass, the gradient to send to the previous stage becomes available.
                    if stage > 0:
                        state.grad_to_left = grad_in
                    else:
                        state.grad_to_left = None

                    # We measure the new value for the objective after each block update (backward pass on any stage),
                    # to track the block-update curve.
                    current_obj = (
                        objective.full_objective(stage_weights)
                        if self.log_full_objective
                        else latest_forward_loss
                    )
                    block_update_objective.append(current_obj)  # update after each backward update

                    # We also track the objective after the backward pass on stage 0 for each microbatch
                    if stage == 0:
                        completion_objective.append(current_obj)  # update after backward pass on stage 0.

                    full_grad_norm_sq_now = compute_full_grad_norm_sq(stage_weights)
                    full_grad_norm_sq_trace.append(full_grad_norm_sq_now)
                    cumulative_full_grad_norm_sq += full_grad_norm_sq_now
                    avg_full_grad_norm_sq_trace.append(cumulative_full_grad_norm_sq / len(full_grad_norm_sq_trace))

                else:
                    raise ValueError(f"Unknown op kind: {kind}")

            # This is the loss history after each time step, which includes multiple forward and backward passes.
            # We expect to see a decrease in the objective over time,
            # but it may be noisy due to the asynchronous updates and staleness.
            if self.log_full_objective:
                objective_trace.append(objective.full_objective(stage_weights))
            else:
                objective_trace.append(latest_forward_loss)

            # We also track how many microbatches have completed (i.e. finished their backward pass on stage 0)
            # at the end of each time step,
            time_completed.append(sum(1 for mb in range(num_microbatches) if backward_versions[mb, 0] >= 0))

            # and the history of stage weight versions for debugging and analysis.
            stage_version_history.append(versions.snapshot())

        if np.any(forward_versions != backward_versions):
            # In a correct implementation of PipeDream, the version of weights used for the forward pass
            # of each stage and microbatch should match the version used for the backward pass.
            bad = np.argwhere(forward_versions != backward_versions)
            raise RuntimeError(f"Weight stashing failed for entries: {bad[:10]}")

        metadata = {
            "time_completed": np.array(time_completed),
            "completion_objective": np.array(completion_objective),
            "training_batch_indices": np.array(self.training_batch_indices[:num_microbatches]),
            "final_weight": combine_stage_weights(stage_weights),
            "forward_history_indices": forward_history_indices,
        }

        return SimulationTrace(
            method_name=self.name,
            objective_trace=np.array(objective_trace),
            block_update_objective=np.array(block_update_objective),
            stage_version_history=np.array(stage_version_history),
            forward_versions=forward_versions,
            backward_versions=backward_versions,
            backward_staleness=backward_staleness,

            grad_norm_trace=np.array(grad_norm_trace),
            forward_loss_trace=np.array(forward_loss_trace),
            forward_loss_time_trace=np.array(forward_loss_time_trace),
            full_grad_norm_sq_trace=np.array(full_grad_norm_sq_trace),
            avg_full_grad_norm_sq_trace=np.array(avg_full_grad_norm_sq_trace),

            metadata=metadata,
        )
