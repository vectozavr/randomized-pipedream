from __future__ import annotations

from itertools import product
from typing import Any, Callable

from src.experiments.runner import ExperimentRunner
from src.methods.base import Method
from src.state.trace import SimulationTrace


def parameter_sweep(
    runner: ExperimentRunner,
    method_factory: Callable[..., Method],
    grid: dict[str, list[Any]],
) -> list[tuple[dict[str, Any], SimulationTrace]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    results: list[tuple[dict[str, Any], SimulationTrace]] = []
    for combo in product(*values):
        kwargs = dict(zip(keys, combo))
        method = method_factory(**kwargs)
        trace = runner.run(method)
        results.append((kwargs, trace))
    return results
