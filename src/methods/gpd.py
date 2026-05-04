from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.methods.base import Method
from src.methods.utils import sample_stale_read_indices
from src.objectives.base import Objective
from src.state import Timeline
from src.state.trace import SimulationTrace
from src.utils.progress import ProgressBar
from src.utils.partitioning import (
    clone_stage_weights,
    clone_weight,
    combine_stage_weights,
    stage_weight_norm,
    sum_squared_stage_weights,
)
from src.state.versions import VersionTracker

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


def build_pipedream_exact_delays(timeline, forward_history_indices):
    backward_ops = extract_pipedream_backward_ops(timeline)
    K = len(backward_ops)
    num_stages = forward_history_indices.shape[1]

    exact_delays = np.zeros((K, num_stages), dtype=int)

    for k, (_, mb) in enumerate(backward_ops):
        exact_delays[k] = k - forward_history_indices[mb]

    return exact_delays

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
    pipedream_exact_delays: np.ndarray | None = None

    init_stage_weights: list[np.ndarray] | None = None
    store_final_weight: bool = True
    show_progress: bool = False
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

        def compute_full_grad_norm_sq(current_stage_weights: list[np.ndarray]) -> float:
            full_grad = objective.full_gradient(current_stage_weights)
            return sum_squared_stage_weights(full_grad)

        def compute_theory_bound(
            K_value: int,
            G_est: float,
        ) -> float:
            S = float(num_stages)
            gamma = float(self.learning_rate)

            return float(
                2.0 * S * (f0 - f_star) / (gamma * K_value)
                + gamma * S * L * (G_est ** 2)
                + (gamma ** 2) * (L ** 2) * (self.delta ** 2) * (G_est ** 2)
            )

        versions = VersionTracker(num_stages=num_stages)

        # History of stage weights for all iterations, used for simulating staleness.
        # Each entry is a list of stage weights at that iteration.
        history = [clone_stage_weights(stage_weights)]

        objective_trace = [objective.full_objective(stage_weights)]
        stage_version_history = [versions.snapshot()]

        f0 = float(objective_trace[0])
        f_star = float(objective.optimal_objective_value)
        L = float(objective.smoothness_constant)

        initial_full_grad_norm_sq = compute_full_grad_norm_sq(stage_weights)
        full_grad_norm_sq_trace = [initial_full_grad_norm_sq]
        avg_full_grad_norm_sq_trace = [initial_full_grad_norm_sq]
        estimated_G = 0.0
        estimated_G_trace = [estimated_G]
        theory_bound_trace = [compute_theory_bound(
            K_value=1,
            G_est=estimated_G,
        )]

        block_update_objective: list[float] = []
        sampled_stages = []
        sampled_batches = []
        sampled_delays = []
        stale_distance_trace = []
        grad_norm_trace = []
        stage_update_counts = np.zeros(num_stages, dtype=int)

        progress = ProgressBar(self.num_iterations, label=self.name, enabled=self.show_progress)
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
            if self.pipedream_exact_delays is not None:
                delays = np.asarray(self.pipedream_exact_delays[k], dtype=int)
                if np.any(delays < 0):
                    raise ValueError("pipedream_exact_delays contains negative values")
                if np.any(delays > k):
                    raise ValueError(
                        f"pipedream_exact_delays at iteration {k} asks for delay > k"
                    )
                read_indices = np.array([k - int(d) for d in delays], dtype=int)
            elif self.stale_sampling == "uniform":
                read_indices, delays = sample_stale_read_indices(
                    k=k,
                    num_stages=num_stages,
                    delta=self.delta,
                    rng=rng,
                    mode="uniform",
                )
            elif self.stale_sampling == "pipedream":
                '''
                delays = np.array(
                    [min((num_stages - 1) - s, k) for s in range(num_stages)],
                    dtype=int,
                )
                read_indices = np.array([k - d for d in delays], dtype=int)
                '''
                active_stage = s

                delays = np.array(
                    [min(max(0, stage_idx - active_stage), k) for stage_idx in range(num_stages)],
                    dtype=int,
                )
                read_indices = np.array([k - d for d in delays], dtype=int)


            # form our weird z weights for all stages based on the sampled read indices
            z_stage_weights = [clone_weight(history[read_indices[ss]][ss]) for ss in range(num_stages)]

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

            estimated_G = max(estimated_G, stage_weight_norm(grad_s))

            # Update the stage weights for stage s using the computed gradient.
            # This simulates a block update for that stage.
            stage_weights[s] = stage_weights[s] - self.learning_rate * grad_s
            versions.increment(s)
            stage_update_counts[s] += 1
            history.append(clone_stage_weights(stage_weights))

            obj_now = objective.full_objective(stage_weights)
            objective_trace.append(obj_now)
            stage_version_history.append(versions.snapshot())
            block_update_objective.append(obj_now)



            full_grad_norm_sq_now = compute_full_grad_norm_sq(stage_weights)
            full_grad_norm_sq_trace.append(full_grad_norm_sq_now)
            avg_full_grad_norm_sq_trace.append(float(np.mean(full_grad_norm_sq_trace)))
            estimated_G_trace.append(estimated_G)
            theory_bound_trace.append(
                compute_theory_bound(
                    K_value=len(full_grad_norm_sq_trace),
                    G_est=estimated_G,
                )
            )


            sampled_stages.append(s)
            sampled_batches.append(m)
            sampled_delays.append(delays.copy())

            current_full = combine_stage_weights(stage_weights)
            stale_full = combine_stage_weights(z_stage_weights)
            stale_distance_trace.append(stage_weight_norm(current_full - stale_full))
            grad_norm_trace.append(stage_weight_norm(grad_s))
            progress.update(k + 1)
        progress.close()

        metadata = {
            "delta": self.delta,
            "stage_update_counts": stage_update_counts,
        }
        if self.store_final_weight:
            metadata["final_weight"] = combine_stage_weights(stage_weights)

        return SimulationTrace(
            method_name=self.name,
            objective_trace=np.array(objective_trace),
            stage_version_history=np.array(stage_version_history),
            block_update_objective=np.array(block_update_objective),
            sampled_stages=np.array(sampled_stages),
            sampled_batches=np.array(sampled_batches),
            sampled_delays=np.array(sampled_delays),
            stale_distance_trace=np.array(stale_distance_trace),
            grad_norm_trace=np.array(grad_norm_trace),

            full_grad_norm_sq_trace=np.array(full_grad_norm_sq_trace),
            avg_full_grad_norm_sq_trace=np.array(avg_full_grad_norm_sq_trace),
            estimated_G_trace=np.array(estimated_G_trace),
            theory_bound_trace=np.array(theory_bound_trace),

            metadata=metadata,
        )
