from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.methods.base import Method
from src.methods.utils import sample_stale_read_indices
from src.objectives.base import Objective
from src.state import Timeline
from src.state.trace import SimulationTrace
from src.utils.partitioning import clone_stage_weights, combine_stage_weights


def extract_pipedream_backward_ops(timeline):
    backward_ops = []
    for ops_this_step in timeline:
        for stage, op in enumerate(ops_this_step):
            if op is None:
                continue
            kind, mb = op
            if kind == "B":
                backward_ops.append((stage, mb))
    return backward_ops

@dataclass
class GPDMethod(Method):
    num_iterations: int
    learning_rate: float
    delta: int
    seed: int = 0

    stage_sampling: str = "uniform"
    batch_sampling: str = "uniform"
    stale_sampling: str = "uniform"

    training_batch_indices: list[int] | None = None
    timeline: Timeline = None

    init_stage_weights: list[np.ndarray] | None = None
    name: str = "GPD"

    def run(self, objective: Objective) -> SimulationTrace:
        rng = np.random.default_rng(self.seed)
        batches = objective.get_batches()
        num_stages = objective.num_stages
        num_batches = len(batches)

        if self.training_batch_indices is None:
            available_batch_indices = list(range(num_batches))
        else:
            available_batch_indices = list(self.training_batch_indices)
            if len(available_batch_indices) == 0:
                raise ValueError("training_batch_indices must not be empty")
            if min(available_batch_indices) < 0 or max(available_batch_indices) >= num_batches:
                raise ValueError("training_batch_indices contains invalid batch ids")

        pipedream_backward_ops = None
        if self.stage_sampling == "pipedream" or self.batch_sampling == "pipedream":
            if self.timeline is None:
                raise ValueError(
                    "timeline must be provided when sampling_order='pipedream' "
                    "or blocks_update_order='pipedream'"
                )
            pipedream_backward_ops = extract_pipedream_backward_ops(self.timeline)
            if len(pipedream_backward_ops) == 0:
                raise ValueError("timeline contains no backward operations")


        # Initialize stage weights. The stage_weights is a list of parameters for every stage.
        if self.init_stage_weights is None:
            stage_weights = objective.initial_stage_weights(mode="zeros")
        else:
            stage_weights = clone_stage_weights(self.init_stage_weights)

        # History of stage weights for all iterations, used for simulating staleness.
        # Each entry is a list of stage weights at that iteration.
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

            mb_pd = None
            if self.stage_sampling == "uniform":
                s = int(rng.integers(0, num_stages))
            elif self.stage_sampling == "pipedream":
                assert pipedream_backward_ops is not None
                s, mb_pd = pipedream_backward_ops[k % len(pipedream_backward_ops)]
            else:
                raise ValueError(f"Unknown stage sampling: {self.stage_sampling}")

            if self.batch_sampling == "uniform":
                idx = int(rng.integers(0, len(available_batch_indices)))
                m = int(available_batch_indices[idx])
            elif self.batch_sampling == "cyclic":
                m = int(available_batch_indices[k % len(available_batch_indices)])
            elif self.batch_sampling == "pipedream":
                assert pipedream_backward_ops is not None
                if self.training_batch_indices is None:
                    raise ValueError(
                        "training_batch_indices must be provided when sampling_order='pipedream'"
                    )
                if mb_pd is None:
                    _, mb_pd = pipedream_backward_ops[k % len(pipedream_backward_ops)]
                if mb_pd >= len(self.training_batch_indices):
                    raise ValueError(
                        f"PipeDream microbatch index {mb_pd} is out of range for "
                        f"training_batch_indices of length {len(self.training_batch_indices)}"
                    )
                m = int(self.training_batch_indices[mb_pd])
            else:
                raise ValueError(f"Unknown batch sampling: {self.batch_sampling}")

            # Simulate staleness by sampling read indices for each stage.
            # The read index determines which iteration's weights are read for that stage.
            if self.stale_sampling == "uniform":
                read_indices, delays = sample_stale_read_indices(
                    k=k,
                    num_stages=num_stages,
                    delta=self.delta,
                    rng=rng,
                    mode="uniform",
                )
            elif self.stale_sampling == "pipedream":
                delays = np.array(
                    [min((num_stages - 1) - s, k) for s in range(num_stages)],
                    dtype=int,
                )
                read_indices = np.array([k - d for d in delays], dtype=int)

            # form our weird z weights for all stages based on the sampled read indices
            z_stage_weights = [history[read_indices[ss]][ss].copy() for ss in range(num_stages)]

            batch = batches[m]

            activations: list[np.ndarray | None] = [None] * (num_stages + 1)
            caches: list[dict] = [{} for _ in range(num_stages)]
            activations[0] = objective.initial_activation(batch)

            # Perform a forward pass through all stages using the z_stage_weights.
            # We also cache any intermediate values needed for the backward pass.
            for stage in range(num_stages):
                activation_in = activations[stage]
                if activation_in is None:
                    raise RuntimeError(f"Missing activation for stage {stage} in GPD forward pass")
                activation_out, cache = objective.forward_stage(
                    batch=batch,
                    stage=stage,
                    w_stage=z_stage_weights[stage],
                    activation_in=activation_in,
                )
                activations[stage + 1] = activation_out
                caches[stage] = cache

            final_activation = activations[num_stages]
            if final_activation is None:
                raise RuntimeError("Missing final activation in GPD")

            _, grad_out = objective.loss_and_output_grad(batch, final_activation)

            grad_s: np.ndarray | None = None
            current_grad = grad_out
            # Perform a backward pass starting from the last stage down to stage s (which was randomly sampled),
            # using the cached values and z_stage_weights.
            for stage in range(num_stages - 1, s - 1, -1):
                grad_w, grad_in = objective.backward_stage(
                    batch=batch,
                    stage=stage,
                    w_stage=z_stage_weights[stage],
                    cache=caches[stage],
                    grad_out=current_grad,
                )
                if stage == s:
                    grad_s = grad_w
                    break
                current_grad = grad_in

            if grad_s is None:
                raise RuntimeError(f"Failed to compute gradient for stage {s} in GPD")

            # Update the stage weights for stage s using the computed gradient.
            # This simulates a block update for that stage.
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