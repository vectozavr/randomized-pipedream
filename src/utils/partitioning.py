from __future__ import annotations

import numpy as np


def make_stage_slices(total_size: int, num_parts: int) -> list[slice]:
    if num_parts <= 0:
        raise ValueError("num_parts must be positive")
    if total_size < num_parts:
        raise ValueError("total_size must be at least num_parts")
    base = total_size // num_parts
    remainder = total_size % num_parts
    result: list[slice] = []
    start = 0
    for part in range(num_parts):
        extra = 1 if part < remainder else 0
        end = start + base + extra
        result.append(slice(start, end))
        start = end
    return result


def slice_sizes(stage_slices: list[slice]) -> list[int]:
    return [sl.stop - sl.start for sl in stage_slices]


def combine_stage_weights(stage_weights: list[np.ndarray]) -> np.ndarray:
    if not stage_weights:
        return np.array([], dtype=float)
    return np.concatenate(stage_weights)


def clone_stage_weights(stage_weights: list[np.ndarray]) -> list[np.ndarray]:
    return [w.copy() for w in stage_weights]
