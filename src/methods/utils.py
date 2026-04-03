from __future__ import annotations

import numpy as np

from src.utils.partitioning import clone_stage_weights


def build_prediction_from_stage_weights(
    Xb: np.ndarray,
    stage_slices: list[slice],
    stage_weights: list[np.ndarray],
) -> np.ndarray:
    pred = np.zeros(Xb.shape[0])
    for s, sl in enumerate(stage_slices):
        pred += Xb[:, sl] @ stage_weights[s]
    return pred


def sample_stale_read_indices(
    k: int,
    num_stages: int,
    delta: int,
    rng: np.random.Generator,
    mode: str = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    max_delay = min(delta, k)

    if mode == "uniform":
        delays = rng.integers(0, max_delay + 1, size=num_stages)
    elif mode == "max":
        delays = np.full(num_stages, max_delay, dtype=int)
    elif mode == "zero":
        delays = np.zeros(num_stages, dtype=int)
    else:
        raise ValueError(f"Unknown stale sampling mode: {mode}")

    read_indices = np.array([k - int(d) for d in delays], dtype=int)
    return read_indices, delays


def clone_history(history_entry: list[np.ndarray]) -> list[np.ndarray]:
    return clone_stage_weights(history_entry)
