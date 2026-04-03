from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Batch = tuple[np.ndarray, np.ndarray]


def build_batches(X: np.ndarray, y: np.ndarray, batch_size: int) -> list[Batch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of examples")
    if len(X) % batch_size != 0:
        raise ValueError("For now this project assumes len(X) is divisible by batch_size.")
    batches: list[Batch] = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        batches.append((X[start:end], y[start:end]))
    return batches


@dataclass
class BatchSampler:
    num_batches: int
    mode: str = "sequential"
    seed: int = 0
    position: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def next_index(self) -> int:
        if self.mode == "sequential":
            index = self.position % self.num_batches
            self.position += 1
            return index
        if self.mode == "random":
            return int(self.rng.integers(0, self.num_batches))
        raise ValueError(f"Unknown sampling mode: {self.mode}")

    def sample_many(self, count: int) -> list[int]:
        return [self.next_index() for _ in range(count)]
