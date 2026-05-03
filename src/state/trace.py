from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SimulationTrace:
    method_name: str
    objective_trace: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    block_update_objective: np.ndarray | None = None
    stage_version_history: np.ndarray | None = None
    forward_versions: np.ndarray | None = None
    backward_versions: np.ndarray | None = None
    backward_staleness: np.ndarray | None = None
    sampled_stages: np.ndarray | None = None
    sampled_batches: np.ndarray | None = None
    sampled_delays: np.ndarray | None = None
    stale_distance_trace: np.ndarray | None = None
    grad_norm_trace: np.ndarray | None = None
    forward_loss_trace: np.ndarray | None = None
    forward_loss_time_trace: np.ndarray | None = None

    full_grad_norm_sq_trace: np.ndarray | None = None
    avg_full_grad_norm_sq_trace: np.ndarray | None = None
    estimated_G_trace: np.ndarray | None = None
    theory_bound_trace: np.ndarray | None = None

    @property
    def final_objective(self) -> float:
        return float(self.objective_trace[-1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "objective_trace": self.objective_trace,
            "metadata": self.metadata,
            "block_update_objective": self.block_update_objective,
            "stage_version_history": self.stage_version_history,
            "forward_versions": self.forward_versions,
            "backward_versions": self.backward_versions,
            "backward_staleness": self.backward_staleness,
            "sampled_stages": self.sampled_stages,
            "sampled_batches": self.sampled_batches,
            "sampled_delays": self.sampled_delays,
            "stale_distance_trace": self.stale_distance_trace,
            "grad_norm_trace": self.grad_norm_trace,
            "forward_loss_trace": self.forward_loss_trace,
            "forward_loss_time_trace": self.forward_loss_time_trace,

            "full_grad_norm_sq_trace": self.full_grad_norm_sq_trace,
            "avg_full_grad_norm_sq_trace": self.avg_full_grad_norm_sq_trace,
            "estimated_G_trace": self.estimated_G_trace,
            "theory_bound_trace": self.theory_bound_trace,
        }


@dataclass
class MethodComparison:
    traces: dict[str, SimulationTrace]
    aligned_curves: dict[str, np.ndarray]
    x_label: str = "number of block updates"
