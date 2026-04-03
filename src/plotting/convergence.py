from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from src.state.trace import MethodComparison, SimulationTrace


def plot_objective_trace(trace: SimulationTrace, log_scale: bool = False, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(trace.objective_trace))
    if log_scale:
        ax.semilogy(x, np.maximum(trace.objective_trace, 1e-16), label=trace.method_name)
    else:
        ax.plot(x, trace.objective_trace, label=trace.method_name)
    ax.set_xlabel("iteration")
    ax.set_ylabel("full objective")
    ax.legend()
    return ax


def plot_block_update_comparison(comparison: MethodComparison, log_scale: bool = False, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    for name, curve in comparison.aligned_curves.items():
        y = np.maximum(curve, 1e-16) if log_scale else curve
        if log_scale:
            ax.semilogy(y, label=name)
        else:
            ax.plot(y, label=name)
    ax.set_xlabel(comparison.x_label)
    ax.set_ylabel("full objective")
    ax.legend()
    return ax
