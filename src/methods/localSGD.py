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
class LocalMinibatchSGD1F1BMethod(Method):
    timeline: Timeline
    learning_rate: float
    training_batch_indices: list[int]
    num_runs: int = 1  # M
    local_steps: int = 1  # K
    init_stage_weights: list[np.ndarray] | None = None
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
                stacked = np.stack([models[m][stage] for m in range(M)])
                averaged_stage_weights.append(np.mean(stacked, axis=0))
            return averaged_stage_weights

        def compute_full_grad_norm_sq(current_stage_weights: list[np.ndarray]) -> float:
            full_grad = objective.full_gradient(current_stage_weights)
            return float(sum(np.sum(g ** 2) for g in full_grad))

        forward_history_indices = -np.ones((num_microbatches, num_stages), dtype=int)
        history = [clone_stage_weights(base_weights)]
        micro: dict[int, MicrobatchRuntime] = {}

        forward_versions = -np.ones((num_microbatches, num_stages), dtype=int)
        backward_versions = -np.ones((num_microbatches, num_stages), dtype=int)
        backward_staleness = -np.ones((num_microbatches, num_stages), dtype=int)

        objective_trace = [objective.full_objective(base_weights)]
        block_update_objective: list[float] = []
        time_completed = [0]
        completion_objective: list[float] = []

        initial_full_grad_norm_sq = compute_full_grad_norm_sq(base_weights)
        full_grad_norm_sq_trace = [initial_full_grad_norm_sq]
        avg_full_grad_norm_sq_trace = [initial_full_grad_norm_sq]
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

                    forward_history_indices[mb, stage] = len(history) - 1

                    # Fetch the current local model's weight
                    w_used = models[m][stage].copy()

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

                elif kind == "B":
                    state = micro[mb]
                    batch = batches[state.batch_id]
                    grad_out = state.grad_to_left

                    stashed = state.stashed_weights[stage]
                    grad_w, grad_in = objective.backward_stage(batch=batch, stage=stage, w_stage=stashed, cache={},
                                                               grad_out=grad_out)

                    grad_norm_trace.append(float(np.linalg.norm(grad_w)))
                    backward_versions[mb, stage] = int(state.stashed_versions[stage])

                    # Staleness strictly 0 because of scheduler barriers
                    backward_staleness[mb, stage] = int(versions[m].versions[stage] - state.stashed_versions[stage])

                    # Local Update Step
                    models[m][stage] = models[m][stage] - self.learning_rate * grad_w
                    versions[m].increment(stage)

                    if stage > 0:
                        state.grad_to_left = grad_in
                    else:
                        state.grad_to_left = None

                    current_avg_weights = get_averaged_weights()
                    current_obj = objective.full_objective(current_avg_weights)
                    block_update_objective.append(current_obj)
                    if stage == 0: completion_objective.append(current_obj)

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
            # ==========================================================

            current_avg_weights = get_averaged_weights()
            objective_trace.append(objective.full_objective(current_avg_weights))
            time_completed.append(sum(1 for mb_idx in range(num_microbatches) if backward_versions[mb_idx, 0] >= 0))
            history.append(clone_stage_weights(current_avg_weights))

        final_averaged_weights = get_averaged_weights()

        metadata = {
            "time_completed": np.array(time_completed),
            "completion_objective": np.array(completion_objective),
            "training_batch_indices": np.array(self.training_batch_indices[:num_microbatches]),
            "final_weight": combine_stage_weights(final_averaged_weights),
            "forward_history_indices": forward_history_indices,
            "num_runs_M": M,
            "local_steps_K": K,
        }

        return SimulationTrace(
            method_name=self.name,
            objective_trace=np.array(objective_trace),
            block_update_objective=np.array(block_update_objective),
            stage_version_history=np.empty(0),
            forward_versions=forward_versions,
            backward_versions=backward_versions,
            backward_staleness=backward_staleness,
            grad_norm_trace=np.array(grad_norm_trace),
            full_grad_norm_sq_trace=np.array(full_grad_norm_sq_trace),
            avg_full_grad_norm_sq_trace=np.array(avg_full_grad_norm_sq_trace),
            metadata=metadata,
        )
