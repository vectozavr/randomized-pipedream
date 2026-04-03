from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.utils.batching import Batch


class Objective(ABC):
    @property
    @abstractmethod
    def num_stages(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def stage_slices(self) -> list[slice]:
        raise NotImplementedError

    @abstractmethod
    def initial_stage_weights(
        self,
        mode: str = "zeros",
        seed: int = 0,
        scale: float = 1e-2,
    ) -> list[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def initial_activation(self, batch: Batch) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_batches(self) -> list[Batch]:
        raise NotImplementedError

    @abstractmethod
    def full_objective(self, stage_weights: list[np.ndarray]) -> float:
        raise NotImplementedError

    @abstractmethod
    def forward_stage(
        self,
        batch: Batch,
        stage: int,
        w_stage: np.ndarray,
        activation_in: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        raise NotImplementedError

    @abstractmethod
    def loss_and_output_grad(
        self,
        batch: Batch,
        final_activation: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def backward_stage(
        self,
        batch: Batch,
        stage: int,
        w_stage: np.ndarray,
        cache: dict,
        grad_out: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
