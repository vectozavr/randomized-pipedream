from __future__ import annotations

import numpy as np

from src.experiments.runner import ExperimentRunner
from src.methods.base import Method
from src.state.trace import MethodComparison, SimulationTrace


def align_block_update_curves(traces: dict[str, SimulationTrace]) -> dict[str, np.ndarray]:
    lengths = []
    for trace in traces.values():
        curve = trace.block_update_objective
        if curve is None:
            raise ValueError(f"Trace {trace.method_name} does not expose block_update_objective.")
        lengths.append(len(curve))
    T = min(lengths)
    return {name: trace.block_update_objective[:T] for name, trace in traces.items()}


def compare_methods(runner: ExperimentRunner, methods: list[Method]) -> MethodComparison:
    traces = runner.run_many(methods)
    aligned = align_block_update_curves(traces)
    return MethodComparison(traces=traces, aligned_curves=aligned)
