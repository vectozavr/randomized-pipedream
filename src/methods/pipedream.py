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
    selected_batch_indices: list[int]
    init_stage_weights: list[np.ndarray] | None = None
    name: str = "PipeDream"

    def run(self, objective: Objective) -> SimulationTrace:
        num_stages = objective.num_stages
        num_microbatches = num_microbatches_from_timeline(self.timeline)
        batches = objective.get_batches()

        if len(self.selected_batch_indices) < num_microbatches:
            raise ValueError("selected_batch_indices must cover all microbatches in the timeline")

        if self.init_stage_weights is None:
            stage_weights = objective.initial_stage_weights(mode="zeros")
        else:
            stage_weights = clone_stage_weights(self.init_stage_weights)

        versions = VersionTracker(num_stages=num_stages)
        micro: dict[int, MicrobatchRuntime] = {}

        forward_versions = -np.ones((num_microbatches, num_stages), dtype=int)
        backward_versions = -np.ones((num_microbatches, num_stages), dtype=int)
        backward_staleness = -np.ones((num_microbatches, num_stages), dtype=int)

        objective_trace = [objective.full_objective(stage_weights)]
        block_update_objective: list[float] = []
        stage_version_history = [versions.snapshot()]
        time_completed = [0]
        completion_objective: list[float] = []

        for t, ops in enumerate(self.timeline):
            for stage, op in enumerate(ops):
                if op is None:
                    continue

                kind, mb = op

                if kind == "F":
                    if stage == 0:
                        batch_id = self.selected_batch_indices[mb]
                        Xb, yb = batches[batch_id]
                        micro[mb] = MicrobatchRuntime(
                            batch_id=batch_id,
                            activations=[None] * (num_stages + 1),
                            grad_to_left=None,
                            stashed_weights=[None] * num_stages,
                            stashed_versions=[None] * num_stages,
                            loss_on_forward=None,
                        )
                        micro[mb].activations[0] = np.zeros(len(yb))

                    state = micro[mb]
                    batch = batches[state.batch_id]

                    w_used = stage_weights[stage].copy()
                    state.stashed_weights[stage] = w_used
                    state.stashed_versions[stage] = int(versions.versions[stage])
                    forward_versions[mb, stage] = int(versions.versions[stage])

                    activation_in = state.activations[stage]
                    if activation_in is None:
                        raise RuntimeError(f"Missing input activation for stage {stage}, microbatch {mb}")

                    activation_out, _ = objective.forward_stage(
                        batch=batch,
                        stage=stage,
                        w_stage=w_used,
                        activation_in=activation_in,
                    )
                    state.activations[stage + 1] = activation_out

                    if stage == num_stages - 1:
                        loss, grad_out = objective.loss_and_output_grad(batch, activation_out)
                        state.loss_on_forward = loss
                        state.grad_to_left = grad_out

                elif kind == "B":
                    state = micro[mb]
                    batch = batches[state.batch_id]

                    grad_out = state.grad_to_left
                    if grad_out is None:
                        raise RuntimeError(f"Backward on stage {stage}, microbatch {mb} has no incoming gradient.")

                    stashed = state.stashed_weights[stage]
                    if stashed is None:
                        raise RuntimeError(f"Missing stashed weights for stage {stage}, microbatch {mb}")

                    grad_w, grad_in = objective.backward_stage(
                        batch=batch,
                        stage=stage,
                        w_stage=stashed,
                        cache={},
                        grad_out=grad_out,
                    )

                    stale_version = state.stashed_versions[stage]
                    if stale_version is None:
                        raise RuntimeError(f"Missing stashed version for stage {stage}, microbatch {mb}")

                    backward_versions[mb, stage] = int(stale_version)
                    backward_staleness[mb, stage] = int(versions.versions[stage] - stale_version)

                    stage_weights[stage] = stage_weights[stage] - self.learning_rate * grad_w
                    versions.increment(stage)

                    if stage > 0:
                        state.grad_to_left = grad_in
                    else:
                        state.grad_to_left = None

                    current_obj = objective.full_objective(stage_weights)
                    block_update_objective.append(current_obj)
                    if stage == 0:
                        completion_objective.append(current_obj)
                else:
                    raise ValueError(f"Unknown op kind: {kind}")

            objective_trace.append(objective.full_objective(stage_weights))
            time_completed.append(sum(1 for mb in range(num_microbatches) if backward_versions[mb, 0] >= 0))
            stage_version_history.append(versions.snapshot())

        if np.any(forward_versions != backward_versions):
            bad = np.argwhere(forward_versions != backward_versions)
            raise RuntimeError(f"Weight stashing failed for entries: {bad[:10]}")

        metadata = {
            "time_completed": np.array(time_completed),
            "completion_objective": np.array(completion_objective),
            "selected_batch_indices": np.array(self.selected_batch_indices[:num_microbatches]),
            "final_weight": combine_stage_weights(stage_weights),
        }

        return SimulationTrace(
            method_name=self.name,
            objective_trace=np.array(objective_trace),
            block_update_objective=np.array(block_update_objective),
            stage_version_history=np.array(stage_version_history),
            forward_versions=forward_versions,
            backward_versions=backward_versions,
            backward_staleness=backward_staleness,
            metadata=metadata,
        )
