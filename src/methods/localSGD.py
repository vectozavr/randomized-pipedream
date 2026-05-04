from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.methods.base import Method
from src.objectives.base import Objective
from src.state.microbatch import MicrobatchRuntime
from src.state.timeline import Timeline, num_microbatches_from_timeline
from src.state.trace import SimulationTrace
from src.state.versions import VersionTracker
from src.utils.partitioning import (
    clone_stage_weights,
    clone_weight,
    combine_stage_weights,
    is_torch_tensor,
    stage_weight_norm,
    sum_squared_stage_weights,
)


@dataclass
class LocalMinibatchSGD1F1BMethod(Method):
    timeline: Timeline
    learning_rate: float
    training_batch_indices: list[int]
    num_runs: int = 1  # M
    local_steps: int = 1  # K
    init_stage_weights: list[np.ndarray] | None = None
    log_full_objective: bool = True
    log_forward_loss: bool = False
    log_grad_norms: bool = True
    store_final_weight: bool = True
    name: str = "LocalSGD-1F1B"

    def run(self, objective: Objective) -> SimulationTrace:
        num_stages = objective.num_stages
        num_microbatches = num_microbatches_from_timeline(self.timeline)
        batches = objective.get_batches()

        M = self.num_runs
        K = self.local_steps
        round_size = M * K

        if len(self.training_batch_indices) < num_microbatches:
            raise ValueError("training_batch_indices must cover all microbatches in the timeline")

        if self.init_stage_weights is None:
            base_weights = objective.initial_stage_weights(mode="zeros")
        else:
            base_weights = clone_stage_weights(self.init_stage_weights)

        # Initialize M independent runs
        models = [clone_stage_weights(base_weights) for _ in range(M)]
        versions = [VersionTracker(num_stages=num_stages) for _ in range(M)]

        def get_averaged_weights() -> list[np.ndarray]:
            averaged_stage_weights = []
            for stage in range(num_stages):
                if is_torch_tensor(models[0][stage]):
                    import torch

                    stacked = torch.stack([models[m][stage] for m in range(M)])
                    averaged_stage_weights.append(torch.mean(stacked, dim=0))
                else:
                    stacked = np.stack([models[m][stage] for m in range(M)])
                    averaged_stage_weights.append(np.mean(stacked, axis=0))
            return averaged_stage_weights

        def compute_full_grad_norm_sq(current_stage_weights: list[np.ndarray]) -> float:
            full_grad = objective.full_gradient(current_stage_weights)
            return sum_squared_stage_weights(full_grad)

        forward_history_indices = -np.ones((num_microbatches, num_stages), dtype=int)
        history_len = 1
        micro: dict[int, MicrobatchRuntime] = {}

        forward_versions = -np.ones((num_microbatches, num_stages), dtype=int)
        backward_versions = -np.ones((num_microbatches, num_stages), dtype=int)
        backward_staleness = -np.ones((num_microbatches, num_stages), dtype=int)

        latest_forward_loss = np.nan
        forward_loss_trace: list[float] = []
        forward_loss_time_trace: list[int] = []

        initial_obj = objective.full_objective(base_weights) if self.log_full_objective else latest_forward_loss
        objective_trace = [initial_obj]
        block_update_objective: list[float] = []
        time_completed = [0]
        completion_objective: list[float] = []

        initial_full_grad_norm_sq = compute_full_grad_norm_sq(base_weights) if self.log_grad_norms else np.nan
        full_grad_norm_sq_trace = [initial_full_grad_norm_sq] if self.log_grad_norms else []
        avg_full_grad_norm_sq_trace = [initial_full_grad_norm_sq] if self.log_grad_norms else []
        cumulative_full_grad_norm_sq = initial_full_grad_norm_sq
        grad_norm_trace: list[float] = []

        synced_rounds = set()  # Track which rounds have been averaged

        for t, ops in enumerate(self.timeline):
            for stage, op in enumerate(ops):
                if op is None:
                    continue

                kind, mb = op
                m = mb % M  # Which run does this microbatch belong to?

                if kind == "F":
                    if stage == 0:
                        batch_id = self.training_batch_indices[mb]
                        batch = batches[batch_id]
                        micro[mb] = MicrobatchRuntime(
                            batch_id=batch_id, activations=[None] * (num_stages + 1),
                            grad_to_left=None, stashed_weights=[None] * num_stages,
                            stashed_versions=[None] * num_stages, loss_on_forward=None,
                        )
                        micro[mb].activations[0] = objective.initial_activation(batch)

                    state = micro[mb]
                    batch = batches[state.batch_id]

                    forward_history_indices[mb, stage] = history_len - 1

                    # Fetch the current local model's weight
                    w_used = clone_weight(models[m][stage])

                    state.stashed_weights[stage] = w_used
                    state.stashed_versions[stage] = int(versions[m].versions[stage])
                    forward_versions[mb, stage] = int(versions[m].versions[stage])

                    activation_in = state.activations[stage]
                    activation_out, _ = objective.forward_stage(batch=batch, stage=stage, w_stage=w_used,
                                                                activation_in=activation_in)
                    state.activations[stage + 1] = activation_out

                    if stage == num_stages - 1:
                        loss, grad_out = objective.loss_and_output_grad(batch, activation_out)
                        state.loss_on_forward = loss
                        state.grad_to_left = grad_out
                        state.activations[num_stages] = None
                        if self.log_forward_loss:
                            latest_forward_loss = float(loss)
                            forward_loss_trace.append(latest_forward_loss)
                            forward_loss_time_trace.append(t)

                elif kind == "B":
                    state = micro[mb]
                    batch = batches[state.batch_id]
                    grad_out = state.grad_to_left

                    stashed = state.stashed_weights[stage]
                    grad_w, grad_in = objective.backward_stage(
                        batch=batch,
                        stage=stage,
                        w_stage=stashed,
                        cache={"activation_in": state.activations[stage]},
                        grad_out=grad_out
                    )

                    if self.log_grad_norms:
                        grad_norm_trace.append(stage_weight_norm(grad_w))
                    backward_versions[mb, stage] = int(state.stashed_versions[stage])

                    # Staleness strictly 0 because of scheduler barriers
                    backward_staleness[mb, stage] = int(versions[m].versions[stage] - state.stashed_versions[stage])

                    # Local Update Step
                    if is_torch_tensor(models[m][stage]):
                        models[m][stage].add_(grad_w, alpha=-self.learning_rate)
                    else:
                        models[m][stage] = models[m][stage] - self.learning_rate * grad_w
                    versions[m].increment(stage)

                    state.stashed_weights[stage] = None
                    state.stashed_versions[stage] = None
                    state.activations[stage] = None
                    if stage + 1 <= num_stages:
                        state.activations[stage + 1] = None

                    if stage > 0:
                        state.grad_to_left = grad_in
                    else:
                        state.grad_to_left = None
                        del micro[mb]

                    current_obj = (
                        objective.full_objective(get_averaged_weights())
                        if self.log_full_objective
                        else latest_forward_loss
                    )
                    block_update_objective.append(current_obj)
                    if stage == 0: completion_objective.append(current_obj)

                    if self.log_grad_norms:
                        current_avg_weights = get_averaged_weights()
                        full_grad_norm_sq_now = compute_full_grad_norm_sq(current_avg_weights)
                        full_grad_norm_sq_trace.append(full_grad_norm_sq_now)
                        cumulative_full_grad_norm_sq += full_grad_norm_sq_now
                        avg_full_grad_norm_sq_trace.append(cumulative_full_grad_norm_sq / len(full_grad_norm_sq_trace))

            # ==========================================================
            # CHECK FOR ROUND SYNCHRONIZATION BARRIER
            # ==========================================================
            # A round is completed when all its microbatches finish Backward on stage 0.
            for r in range(num_microbatches // round_size):
                if r in synced_rounds:
                    continue

                round_start = r * round_size
                round_end = round_start + round_size

                # Check if all backwards in this round have completed on stage 0
                round_finished = True
                for prev_mb in range(round_start, round_end):
                    if backward_versions[prev_mb, 0] < 0:
                        round_finished = False
                        break

                if round_finished:
                    # HARD SYNC: Average weights and broadcast to all M models!
                    avg_weights = get_averaged_weights()
                    for m_idx in range(M):
                        models[m_idx] = clone_stage_weights(avg_weights)
                        # We increment the version tracker just so debugging knows an update happened
                        for s in range(num_stages):
                            versions[m_idx].increment(s)

                    synced_rounds.add(r)
                    del avg_weights
            # ==========================================================

            if self.log_full_objective:
                current_avg_weights = get_averaged_weights()
                objective_trace.append(objective.full_objective(current_avg_weights))
            else:
                objective_trace.append(latest_forward_loss)
            time_completed.append(sum(1 for mb_idx in range(num_microbatches) if backward_versions[mb_idx, 0] >= 0))
            history_len += 1

        final_averaged_weights = get_averaged_weights() if self.store_final_weight else None

        metadata = {
            "time_completed": np.array(time_completed),
            "completion_objective": np.array(completion_objective),
            "training_batch_indices": np.array(self.training_batch_indices[:num_microbatches]),
            "forward_history_indices": forward_history_indices,
            "num_runs_M": M,
            "local_steps_K": K,
        }
        if self.store_final_weight:
            metadata["final_weight"] = combine_stage_weights(final_averaged_weights)

        return SimulationTrace(
            method_name=self.name,
            objective_trace=np.array(objective_trace),
            block_update_objective=np.array(block_update_objective),
            stage_version_history=np.empty(0),
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
