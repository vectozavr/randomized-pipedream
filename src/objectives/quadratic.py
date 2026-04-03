from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.objectives.base import Objective
from src.utils.batching import Batch, build_batches
from src.utils.partitioning import combine_stage_weights, make_stage_slices


@dataclass
class QuadraticObjective(Objective):
    X: np.ndarray
    y: np.ndarray
    num_pipeline_stages: int
    batch_size: int

    def __post_init__(self) -> None:
        self._stage_slices = make_stage_slices(self.X.shape[1], self.num_pipeline_stages)
        self._batches = build_batches(self.X, self.y, self.batch_size)

    @classmethod
    def synthetic(
        cls,
        num_examples: int,
        num_parameters: int,
        num_stages: int,
        batch_size: int,
        seed: int = 0,
        noise_std: float = 0.0,
    ) -> "QuadraticObjective":
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(num_examples, num_parameters))
        true_w = rng.normal(size=(num_parameters,))
        y = X @ true_w
        if noise_std > 0.0:
            y = y + noise_std * rng.normal(size=y.shape)
        return cls(X=X, y=y, num_pipeline_stages=num_stages, batch_size=batch_size)

    @property
    def num_stages(self) -> int:
        return self.num_pipeline_stages

    @property
    def num_parameters(self) -> int:
        return self.X.shape[1]

    @property
    def stage_slices(self) -> list[slice]:
        return self._stage_slices

    @property
    def smoothness_constant(self) -> float:
        xtx = (self.X.T @ self.X) / len(self.X)
        return float(np.linalg.eigvalsh(xtx).max())

    def initial_activation(self, batch: Batch) -> np.ndarray:
        _, yb = batch
        return np.zeros(len(yb))

    def initial_stage_weights(
        self,
        mode: str = "zeros",
        seed: int = 0,
        scale: float = 1e-2,
    ) -> list[np.ndarray]:
        rng = np.random.default_rng(seed)
        result: list[np.ndarray] = []
        for sl in self.stage_slices:
            d = sl.stop - sl.start
            if mode == "zeros":
                result.append(np.zeros(d))
            elif mode == "random":
                result.append(scale * rng.normal(size=d))
            else:
                raise ValueError(f"Unknown init mode: {mode}")
        return result

    def get_batches(self) -> list[Batch]:
        return self._batches

    def full_objective(self, stage_weights: list[np.ndarray]) -> float:
        w = combine_stage_weights(stage_weights)
        residual = self.X @ w - self.y
        return float(0.5 * np.mean(residual ** 2))

    def full_gradient(self, stage_weights: list[np.ndarray]) -> list[np.ndarray]:
        w = combine_stage_weights(stage_weights)
        residual = self.X @ w - self.y
        grad = self.X.T @ (residual / len(self.X))
        return [grad[sl].copy() for sl in self.stage_slices]

    def forward_stage(
        self,
        batch: Batch,
        stage: int,
        w_stage: np.ndarray,
        activation_in: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        Xb, _ = batch
        sl = self.stage_slices[stage]
        contribution = Xb[:, sl] @ w_stage
        activation_out = activation_in + contribution
        cache = {
            "activation_in": activation_in.copy(),
            "activation_out": activation_out.copy(),
            "contribution": contribution.copy(),
        }
        return activation_out, cache

    def loss_and_output_grad(
        self,
        batch: Batch,
        final_activation: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        _, yb = batch
        residual = final_activation - yb
        loss = float(0.5 * np.mean(residual ** 2))
        grad_out = residual / len(yb)
        return loss, grad_out

    def backward_stage(
        self,
        batch: Batch,
        stage: int,
        w_stage: np.ndarray,
        cache: dict,
        grad_out: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        Xb, _ = batch
        sl = self.stage_slices[stage]
        grad_w = Xb[:, sl].T @ grad_out
        grad_in = grad_out.copy()
        return grad_w, grad_in
