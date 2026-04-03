from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.methods.base import Method
from src.objectives.base import Objective
from src.state.trace import SimulationTrace
from src.utils.partitioning import clone_stage_weights, combine_stage_weights


@dataclass
class SGDMethod(Method):
    num_iterations: int
    learning_rate: float
    seed: int = 0
    batch_sampling: str = "uniform"
    init_stage_weights: list[np.ndarray] | None = None
    name: str = "SGD"

    def run(self, objective: Objective) -> SimulationTrace:
        rng = np.random.default_rng(self.seed)
        batches = objective.get_batches()
        num_batches = len(batches)
        num_stages = objective.num_stages

        if self.init_stage_weights is None:
            stage_weights = objective.initial_stage_weights(mode="zeros")
        else:
            stage_weights = clone_stage_weights(self.init_stage_weights)

        objective_trace = [objective.full_objective(stage_weights)]
        block_update_objective: list[float] = []
        sampled_batches = []
        grad_norm_trace = []

        for k in range(self.num_iterations):
            if self.batch_sampling == "uniform":
                m = int(rng.integers(0, num_batches))
            elif self.batch_sampling == "cyclic":
                m = k % num_batches
            else:
                raise ValueError(f"Unknown batch_sampling: {self.batch_sampling}")

            batch = batches[m]
            Xb, yb = batch

            pred = np.zeros(Xb.shape[0])
            for s, sl in enumerate(objective.stage_slices):
                pred += Xb[:, sl] @ stage_weights[s]

            residual = pred - yb
            total_grad_norm_sq = 0.0

            for s, sl in enumerate(objective.stage_slices):
                grad_s = Xb[:, sl].T @ (residual / len(yb))
                stage_weights[s] = stage_weights[s] - self.learning_rate * grad_s
                total_grad_norm_sq += float(np.sum(grad_s ** 2))

            obj_now = objective.full_objective(stage_weights)
            objective_trace.append(obj_now)
            sampled_batches.append(m)
            grad_norm_trace.append(np.sqrt(total_grad_norm_sq))
            block_update_objective.extend([obj_now] * num_stages)

        return SimulationTrace(
            method_name=self.name,
            objective_trace=np.array(objective_trace),
            block_update_objective=np.array(block_update_objective),
            sampled_batches=np.array(sampled_batches),
            grad_norm_trace=np.array(grad_norm_trace),
            metadata={"final_weight": combine_stage_weights(stage_weights)},
        )
