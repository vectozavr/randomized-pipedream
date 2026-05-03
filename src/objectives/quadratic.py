from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
    kind: Literal["random", "simple", "tridiagonal"] = "random"
    true_w: np.ndarray | None = None
    analytic_L: float | None = None
    gradient_noise_std: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        self._stage_slices = make_stage_slices(self.X.shape[1], self.num_pipeline_stages)
        self._batches = build_batches(self.X, self.y, self.batch_size)
        self._rng = np.random.default_rng(self.seed)

    @classmethod
    def synthetic(
            cls,
            num_examples: int,
            num_parameters: int,
            num_stages: int,
            batch_size: int,
            seed: int = 0,
            noise_std: float = 0.0, # labeling noise
            gradient_noise_std: float = 0.0,  # gradient noise (Stochastic Oracle)
            kind: Literal["random", "simple", "tridiagonal"] = "random",
            condition_number: float | None = None,
    ) -> "QuadraticObjective":
        rng = np.random.default_rng(seed)

        if kind == "random":
            X = rng.normal(size=(num_examples, num_parameters))
            true_w = rng.normal(size=(num_parameters,))
            y = X @ true_w
            if noise_std > 0.0:
                y = y + noise_std * rng.normal(size=y.shape)

            return cls(
                X=X, y=y, num_pipeline_stages=num_stages, batch_size=batch_size,
                kind="random", true_w=true_w, analytic_L=None,
                gradient_noise_std=gradient_noise_std, seed=seed,
            )

        if kind == "simple":
            n, d = num_examples, num_parameters
            r = min(n, d)

            lambdas = np.linspace(1.0, float(r), r)
            analytic_L = float(lambdas.max())

            U, _ = np.linalg.qr(rng.normal(size=(n, r)))
            V, _ = np.linalg.qr(rng.normal(size=(d, r)))
            X = np.sqrt(n) * U @ np.diag(np.sqrt(lambdas)) @ V.T

            true_w = rng.normal(size=(d,))
            y = X @ true_w
            if noise_std > 0.0:
                y = y + noise_std * rng.normal(size=y.shape)

            return cls(
                X=X, y=y, num_pipeline_stages=num_stages, batch_size=batch_size,
                kind="simple", true_w=true_w, analytic_L=analytic_L,
                gradient_noise_std=gradient_noise_std, seed=seed,
            )

        if kind == "tridiagonal":
            n, d = num_examples, num_parameters
            if n < d:
                raise ValueError("For exact tridiagonal Hessian, num_examples must be >= num_parameters.")

            # 1. Create Base Nesterov's Worst-Case Tridiagonal Matrix T
            main_diag = 2.0 * np.ones(d)
            side_diag = -1.0 * np.ones(d - 1)
            T = np.diag(main_diag) + np.diag(side_diag, k=1) + np.diag(side_diag, k=-1)

            # Calculate base eigenvalues
            eigvals = np.linalg.eigvalsh(T)
            l_min, l_max = eigvals[0], eigvals[-1]
            base_kappa = l_max / l_min

            # 2. Determine mu to hit the target condition number
            if condition_number is not None:
                if condition_number <= 1.0:
                    raise ValueError("Target condition_number must be > 1.0.")
                if condition_number >= base_kappa:
                    raise ValueError(
                        f"Requested condition number ({condition_number}) must be strictly less "
                        f"than the base unregularized condition number ({base_kappa:.2f})."
                    )

                # Algebra trick: solve for mu to get exactly the desired condition number
                mu = (l_max - condition_number * l_min) / (condition_number - 1.0)
            else:
                mu = 0.01  # Fallback to the old default

            # Add the computed strong convexity shift
            A = T + mu * np.eye(d)

            # 3. Factorize A = L L^T using Cholesky
            L_mat = np.linalg.cholesky(A)

            # 4. Create X such that the empirical Hessian (1/n) X^T X = A exactly!
            Z = rng.normal(size=(n, d))
            Q, _ = np.linalg.qr(Z)  # Q has orthonormal columns: Q^T Q = I
            X = np.sqrt(n) * Q @ L_mat.T

            # 5. Generate targets
            true_w = rng.normal(size=(d,))
            y = X @ true_w
            if noise_std > 0.0:
                y = y + noise_std * rng.normal(size=y.shape)

            analytic_L = float(np.linalg.eigvalsh(A).max())

            return cls(
                X=X, y=y, num_pipeline_stages=num_stages, batch_size=batch_size,
                kind="tridiagonal", true_w=true_w, analytic_L=analytic_L,
                gradient_noise_std=gradient_noise_std, seed=seed,
            )

        raise ValueError(f"Unknown quadratic kind: {kind}")

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

    @property
    def optimal_objective_value(self) -> float:
        w_star, *_ = np.linalg.lstsq(self.X, self.y, rcond=None)
        residual = self.X @ w_star - self.y
        return float(0.5 * np.mean(residual ** 2))

    @property
    def true_smoothness_constant(self) -> float | None:
        return self.analytic_L

    def check_smoothness_constant(self, atol: float = 1e-10, rtol: float = 1e-10) -> bool:
        if self.analytic_L is None:
            raise ValueError("Analytic smoothness constant is available only for kind='simple' or 'tridiagonal'.")
        return np.isclose(self.smoothness_constant, self.analytic_L, atol=atol, rtol=rtol)

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

        if self.gradient_noise_std > 0.0:
            grad_w += self._rng.normal(scale=self.gradient_noise_std, size=grad_w.shape)

        grad_in = grad_out.copy()
        return grad_w, grad_in
