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

        # Initialize stage weights. The stage_weights is a list of parameters for every stage.
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

            activations: list[np.ndarray | None] = [None] * (num_stages + 1)
            caches: list[dict] = [{} for _ in range(num_stages)]
            activations[0] = objective.initial_activation(batch)

            # Full forward pass through all stages using the current model
            for stage in range(num_stages):
                activation_in = activations[stage]
                if activation_in is None:
                    raise RuntimeError(f"Missing activation for stage {stage} in SGD forward pass")

                activation_out, cache = objective.forward_stage(
                    batch=batch,
                    stage=stage,
                    w_stage=stage_weights[stage],
                    activation_in=activation_in,
                )
                activations[stage + 1] = activation_out
                caches[stage] = cache

            final_activation = activations[num_stages]
            if final_activation is None:
                raise RuntimeError("Missing final activation in SGD")

            _, grad_out = objective.loss_and_output_grad(batch, final_activation)

            # Full backward pass: compute all block gradients first
            grads: list[np.ndarray | None] = [None] * num_stages
            current_grad = grad_out
            total_grad_norm_sq = 0.0

            for stage in range(num_stages - 1, -1, -1):
                grad_w, grad_in = objective.backward_stage(
                    batch=batch,
                    stage=stage,
                    w_stage=stage_weights[stage],
                    cache=caches[stage],
                    grad_out=current_grad,
                )
                grads[stage] = grad_w
                total_grad_norm_sq += float(np.sum(grad_w ** 2))
                current_grad = grad_in

            # Apply updates simultaneously after all gradients are computed
            for stage in range(num_stages):
                grad_w = grads[stage]
                if grad_w is None:
                    raise RuntimeError(f"Missing gradient for stage {stage} in SGD")
                stage_weights[stage] = stage_weights[stage] - self.learning_rate * grad_w

            obj_now = objective.full_objective(stage_weights)
            objective_trace.append(obj_now)
            sampled_batches.append(m)
            grad_norm_trace.append(np.sqrt(total_grad_norm_sq))

            # For fair comparison with block-update methods, repeat once per stage
            block_update_objective.extend([obj_now] * num_stages)

        return SimulationTrace(
            method_name=self.name,
            objective_trace=np.array(objective_trace),
            block_update_objective=np.array(block_update_objective),
            sampled_batches=np.array(sampled_batches),
            grad_norm_trace=np.array(grad_norm_trace),
            metadata={"final_weight": combine_stage_weights(stage_weights)},
        )
