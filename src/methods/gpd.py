from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.methods.base import Method
from src.methods.utils import build_prediction_from_stage_weights, sample_stale_read_indices
from src.objectives.base import Objective
from src.state.trace import SimulationTrace
from src.utils.partitioning import clone_stage_weights, combine_stage_weights


@dataclass
class GPDMethod(Method):
    num_iterations: int
    learning_rate: float
    delta: int
    seed: int = 0
    stage_sampling: str = "uniform"
    batch_sampling: str = "uniform"
    stale_sampling: str = "uniform"
    init_stage_weights: list[np.ndarray] | None = None
    name: str = "GPD"

    def run(self, objective: Objective) -> SimulationTrace:
        rng = np.random.default_rng(self.seed)
        batches = objective.get_batches()
        num_stages = objective.num_stages
        num_batches = len(batches)

        if self.init_stage_weights is None:
            stage_weights = objective.initial_stage_weights(mode="zeros")
        else:
            stage_weights = clone_stage_weights(self.init_stage_weights)

        history = [clone_stage_weights(stage_weights)]

        objective_trace = [objective.full_objective(stage_weights)]
        block_update_objective: list[float] = []
        sampled_stages = []
        sampled_batches = []
        sampled_delays = []
        stale_distance_trace = []
        grad_norm_trace = []
        stage_update_counts = np.zeros(num_stages, dtype=int)

        for k in range(self.num_iterations):
            if self.stage_sampling == "uniform":
                s = int(rng.integers(0, num_stages))
            else:
                raise ValueError(f"Unknown stage sampling: {self.stage_sampling}")

            if self.batch_sampling == "uniform":
                m = int(rng.integers(0, num_batches))
            elif self.batch_sampling == "cyclic":
                m = k % num_batches
            else:
                raise ValueError(f"Unknown batch sampling: {self.batch_sampling}")

            read_indices, delays = sample_stale_read_indices(
                k=k,
                num_stages=num_stages,
                delta=self.delta,
                rng=rng,
                mode=self.stale_sampling,
            )
            z_stage_weights = [history[read_indices[ss]][ss].copy() for ss in range(num_stages)]

            batch = batches[m]
            Xb, yb = batch
            pred_stale = build_prediction_from_stage_weights(Xb, objective.stage_slices, z_stage_weights)
            residual_stale = pred_stale - yb

            sl = objective.stage_slices[s]
            grad_s = Xb[:, sl].T @ (residual_stale / len(yb))

            stage_weights[s] = stage_weights[s] - self.learning_rate * grad_s
            stage_update_counts[s] += 1
            history.append(clone_stage_weights(stage_weights))

            obj_now = objective.full_objective(stage_weights)
            objective_trace.append(obj_now)
            block_update_objective.append(obj_now)

            sampled_stages.append(s)
            sampled_batches.append(m)
            sampled_delays.append(delays.copy())

            current_full = combine_stage_weights(stage_weights)
            stale_full = combine_stage_weights(z_stage_weights)
            stale_distance_trace.append(np.linalg.norm(current_full - stale_full))
            grad_norm_trace.append(np.linalg.norm(grad_s))

        return SimulationTrace(
            method_name=self.name,
            objective_trace=np.array(objective_trace),
            block_update_objective=np.array(block_update_objective),
            sampled_stages=np.array(sampled_stages),
            sampled_batches=np.array(sampled_batches),
            sampled_delays=np.array(sampled_delays),
            stale_distance_trace=np.array(stale_distance_trace),
            grad_norm_trace=np.array(grad_norm_trace),
            metadata={
                "delta": self.delta,
                "stage_update_counts": stage_update_counts,
                "final_weight": combine_stage_weights(stage_weights),
            },
        )
