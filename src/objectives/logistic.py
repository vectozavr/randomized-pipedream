from __future__ import annotations
import scipy.optimize

from dataclasses import dataclass

import numpy as np

from src.objectives.base import Objective
from src.utils.batching import Batch, build_batches
from src.utils.partitioning import combine_stage_weights, make_stage_slices


# Helper for numerically stable sigmoid
def expit(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


@dataclass
class LogisticRegressionObjective(Objective):
    X: np.ndarray
    y: np.ndarray
    num_pipeline_stages: int
    batch_size: int
    l2_reg: float = 0.0
    true_w: np.ndarray | None = None
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
            l2_reg: float = 0.0,
            gradient_noise_std: float = 0.0,
    ) -> "LogisticRegressionObjective":
        rng = np.random.default_rng(seed)

        # Generate random features
        X = rng.normal(size=(num_examples, num_parameters))

        # Generate true weights
        true_w = rng.normal(size=(num_parameters,))

        # Generate binary labels based on true probabilities
        logits = X @ true_w
        probs = expit(logits)
        y = rng.binomial(1, probs).astype(np.float32)

        return cls(
            X=X,
            y=y,
            num_pipeline_stages=num_stages,
            batch_size=batch_size,
            l2_reg=l2_reg,
            true_w=true_w,
            gradient_noise_std=gradient_noise_std,
            seed=seed,
        )

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
        # For Logistic Regression with L2 regularization, the Hessian is bounded by:
        # H <= (1/4N) * X^T X + l2_reg * I
        xtx = (self.X.T @ self.X) / len(self.X)
        max_eig = float(np.linalg.eigvalsh(xtx).max())
        return (0.25 * max_eig) + self.l2_reg

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
        logits = self.X @ w

        # Numerically stable Binary Cross Entropy
        # Loss = -y*log(p) - (1-y)*log(1-p)
        # Using softplus equivalent: max(x, 0) - x * y + log(1 + exp(-abs(x)))
        loss_vec = np.maximum(logits, 0) - logits * self.y + np.log1p(np.exp(-np.abs(logits)))

        data_loss = float(np.mean(loss_vec))
        reg_loss = 0.5 * self.l2_reg * float(np.sum(w ** 2))
        return data_loss + reg_loss

    def full_gradient(self, stage_weights: list[np.ndarray]) -> list[np.ndarray]:
        w = combine_stage_weights(stage_weights)
        logits = self.X @ w
        preds = expit(logits)

        # grad = X^T (preds - y) / N + l2_reg * w
        residual = preds - self.y
        grad = self.X.T @ (residual / len(self.X)) + self.l2_reg * w
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
        logits = final_activation
        preds = expit(logits)

        # Calculate mini-batch BCE loss
        loss_vec = np.maximum(logits, 0) - logits * yb + np.log1p(np.exp(-np.abs(logits)))
        loss = float(np.mean(loss_vec))

        # The gradient of BCE w.r.t logits is simply (preds - targets)
        grad_out = (preds - yb) / len(yb)

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

        # gradient w.r.t weights for this stage, plus L2 regularization term
        grad_w = Xb[:, sl].T @ grad_out + self.l2_reg * w_stage

        if self.gradient_noise_std > 0.0:
            grad_w += self._rng.normal(scale=self.gradient_noise_std, size=grad_w.shape)

        # gradient to pass to the previous stage is just grad_out
        # (since derivative of addition is 1)
        grad_in = grad_out.copy()

        return grad_w, grad_in

    @property
    def optimal_objective_value(self) -> float:
        # Cache the result so we don't re-run the solver multiple times
        if hasattr(self, '_optimal_val'):
            return self._optimal_val

        # Define the objective and gradient for the scipy solver
        def cost_and_grad(w):
            logits = self.X @ w
            # Loss
            loss_vec = np.maximum(logits, 0) - logits * self.y + np.log1p(np.exp(-np.abs(logits)))
            loss = float(np.mean(loss_vec)) + 0.5 * self.l2_reg * float(np.sum(w ** 2))

            # Gradient
            preds = expit(logits)
            grad = self.X.T @ (preds - self.y) / len(self.X) + self.l2_reg * w

            return loss, grad

        # Use L-BFGS-B to find the true mathematical minimum
        w0 = np.zeros(self.X.shape[1])
        res = scipy.optimize.minimize(cost_and_grad, w0, jac=True, method='L-BFGS-B', tol=1e-12)

        self._optimal_val = float(res.fun)
        return self._optimal_val
