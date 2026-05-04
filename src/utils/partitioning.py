from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for non-LLM experiments.
    torch = None


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


def is_torch_tensor(value) -> bool:
    return torch is not None and torch.is_tensor(value)


def clone_weight(weight):
    if is_torch_tensor(weight):
        return weight.detach().clone()
    return weight.copy()


def combine_stage_weights(stage_weights: list[np.ndarray]) -> np.ndarray:
    if not stage_weights:
        return np.array([], dtype=float)
    if is_torch_tensor(stage_weights[0]):
        return torch.cat([w.reshape(-1) for w in stage_weights])
    return np.concatenate(stage_weights)


def clone_stage_weights(stage_weights: list[np.ndarray]) -> list[np.ndarray]:
    return [clone_weight(w) for w in stage_weights]


def stage_weight_norm(weight) -> float:
    if is_torch_tensor(weight):
        return float(torch.linalg.norm(weight).detach().cpu())
    return float(np.linalg.norm(weight))


def sum_squared_stage_weights(stage_weights) -> float:
    total = 0.0
    for weight in stage_weights:
        if is_torch_tensor(weight):
            total += float(torch.sum(weight.detach() ** 2).cpu())
        else:
            total += float(np.sum(weight ** 2))
    return total
