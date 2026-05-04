from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.experiments.compare_methods import compare_methods
from src.experiments.runner import ExperimentRunner
from src.methods.gpd import GPDMethod
from src.methods.pipedream import PipeDreamMethod
from src.objectives.simple_llm import SimpleLLMObjective
from src.plotting.convergence import plot_block_update_comparison
from src.schedulers.pipedream_1f1b import PipeDream1F1BScheduler, plot_schedule
from src.utils.batching import build_training_batch_schedule
from src.utils.llm_data import load_text_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare PipeDream and GPD/RPD on the SimpleLLMObjective."
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=Path, default=Path("results/llm_experiments"))

    parser.add_argument("--dataset", choices=["toy", "tiny_shakespeare", "file"], default="toy")
    parser.add_argument("--dataset-file", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data/llm"))
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--num-data-batches", type=int, default=8)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Optional corpus cap. Defaults to exactly the text needed for num-data-batches.",
    )

    parser.add_argument("--num-stages", type=int, default=4)
    parser.add_argument("--num-microbatches", type=int, default=16)
    parser.add_argument("--noam", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--require-cuda", action="store_true")

    parser.add_argument("--pd-lr", type=float, default=1e-3)
    parser.add_argument("--gpd-lr", type=float, default=1e-3)
    parser.add_argument("--gpd-delta", type=int, default=None)

    parser.add_argument("--gpd-seed", type=int, default=123)
    parser.add_argument("--shuffle-batches", action="store_true")

    parser.add_argument("--tune-stepsizes", action="store_true")
    parser.add_argument("--tuning-seeds", type=str, default="0")
    parser.add_argument("--pd-lr-grid", type=str, default="pow2:-5:5")
    parser.add_argument("--gpd-lr-grid", type=str, default="pow2:-5:5")
    parser.add_argument(
        "--lr-selection",
        choices=["best-final", "stable"],
        default="best-final",
        help="Choose the swept learning rate by final loss or by the conservative stable selector.",
    )
    parser.add_argument("--stable-tail-frac", type=float, default=0.2)
    parser.add_argument("--no-progress", action="store_true")

    return parser


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.set_float32_matmul_precision("high")
    except ImportError:
        pass


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_lr_grid(value: str) -> np.ndarray:
    value = value.strip()
    if value.startswith("pow2:"):
        _, start, stop = value.split(":")
        return 2.0 ** np.arange(int(start), int(stop))
    return np.array([float(part.strip()) for part in value.split(",") if part.strip()], dtype=float)


def count_backward_ops(timeline) -> int:
    return sum(1 for step in timeline for op in step if op is not None and op[0] == "B")


def make_pd_method(
    *,
    lr: float,
    timeline,
    training_batch_indices: list[int],
    init_stage_weights: list[np.ndarray],
    show_progress: bool,
    name: str,
) -> PipeDreamMethod:
    return PipeDreamMethod(
        timeline=timeline,
        learning_rate=lr,
        training_batch_indices=training_batch_indices,
        init_stage_weights=init_stage_weights,
        log_grad_norms=False,
        store_final_weight=False,
        show_progress=show_progress,
        name=name,
    )


def make_gpd_method(
    *,
    lr: float,
    k_budget: int,
    delta: int,
    seed: int,
    training_batch_indices: list[int],
    init_stage_weights: list[np.ndarray],
    show_progress: bool,
    name: str,
) -> GPDMethod:
    return GPDMethod(
        num_iterations=k_budget,
        learning_rate=lr,
        delta=delta,
        seed=seed,
        training_batch_indices=training_batch_indices,
        init_stage_weights=init_stage_weights,
        store_final_weight=False,
        show_progress=show_progress,
        name=name,
    )


def release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_trials(objective, method_factory: Callable[[int], object], seeds: list[int]):
    traces = []
    for seed in seeds:
        method = method_factory(seed)
        traces.append(method.run(objective))
        del method
        release_cuda_cache()
    return traces


def aggregate_block_curves(traces, k_budget: int, tail_frac: float = 0.2) -> dict[str, object]:
    curves = []
    for trace in traces:
        curve = trace.block_update_objective
        if curve is None:
            raise ValueError(f"{trace.method_name} has no block_update_objective.")
        if len(curve) < k_budget:
            raise ValueError(f"{trace.method_name} curve is too short: {len(curve)} < {k_budget}")
        curves.append(curve[:k_budget])

    stacked = np.stack(curves, axis=0)
    tail_len = max(1, int(np.ceil(tail_frac * k_budget)))
    tail_values = stacked[:, -tail_len:].mean(axis=1)

    return {
        "mean": stacked.mean(axis=0),
        "std": stacked.std(axis=0),
        "final_mean": float(stacked[:, -1].mean()),
        "final_std": float(stacked[:, -1].std()),
        "tail_mean": float(tail_values.mean()),
        "tail_std": float(tail_values.std()),
    }


def sweep_learning_rates(
    *,
    objective,
    method_name: str,
    learning_rates: np.ndarray,
    method_factory_builder: Callable[[float, int], object],
    seeds: list[int],
    k_budget: int,
    tail_frac: float,
) -> dict[str, object]:
    results = {}
    for lr in learning_rates:
        print(f"Running {method_name} lr={lr:.6e}")
        traces = run_trials(
            objective=objective,
            method_factory=lambda seed, lr=float(lr): method_factory_builder(lr, seed),
            seeds=seeds,
        )
        agg = aggregate_block_curves(traces, k_budget=k_budget, tail_frac=tail_frac)
        print(
            f"Finished {method_name} lr={lr:.6e} | "
            f"final loss={agg['final_mean']:.6e} +/- {agg['final_std']:.2e} | "
            f"tail loss={agg['tail_mean']:.6e} +/- {agg['tail_std']:.2e}",
            flush=True,
        )
        results[float(lr)] = {
            "traces": traces,
            "mean_curve": agg["mean"],
            "std_curve": agg["std"],
            "final_mean": agg["final_mean"],
            "final_std": agg["final_std"],
            "tail_mean": agg["tail_mean"],
            "tail_std": agg["tail_std"],
        }

    best_lr = min(results.keys(), key=lambda lr: results[lr]["final_mean"])
    return {
        "method_name": method_name,
        "results": results,
        "best_lr": best_lr,
        "best_curve": results[best_lr]["mean_curve"],
        "best_std": results[best_lr]["std_curve"],
    }


def curve_stability_metrics(curve, tail_frac: float = 0.2, eps: float = 1e-16) -> dict[str, float | bool]:
    y = np.asarray(curve, dtype=float)
    if len(y) < 3 or not np.all(np.isfinite(y)):
        return {
            "stable": False,
            "tail_mean": np.inf,
            "tail_cv": np.inf,
            "up_fraction": 1.0,
            "max_up_rel": np.inf,
            "tv_ratio": np.inf,
            "tail_log_slope": np.inf,
        }

    y = np.maximum(y, eps)
    tail_len = max(1, int(np.ceil(tail_frac * len(y))))
    tail = y[-tail_len:]
    diffs = np.diff(tail)

    tail_mean = float(tail.mean())
    tail_cv = float(tail.std() / (abs(tail_mean) + eps))
    positive = diffs[diffs > 0]
    max_up_rel = float(positive.max() / (abs(tail_mean) + eps)) if len(positive) else 0.0
    up_fraction = float(np.mean(diffs > 0)) if len(diffs) else 0.0
    net_change = abs(tail[0] - tail[-1])
    tv_ratio = float(np.sum(np.abs(diffs)) / (net_change + eps)) if len(diffs) else 0.0
    tail_log_slope = float(np.polyfit(np.arange(len(tail)), np.log(tail), 1)[0]) if len(tail) > 1 else 0.0

    return {
        "stable": True,
        "tail_mean": tail_mean,
        "tail_cv": tail_cv,
        "up_fraction": up_fraction,
        "max_up_rel": max_up_rel,
        "tv_ratio": tv_ratio,
        "tail_log_slope": tail_log_slope,
    }


def select_stable_learning_rate(
    sweep_result,
    *,
    tail_frac: float,
    relative_tol: float = 0.05,
    max_tail_cv: float = 0.25,
    max_up_fraction: float = 0.55,
    max_up_rel: float = 0.10,
    max_tv_ratio: float = 10.0,
    max_tail_log_slope: float = 1e-3,
) -> tuple[float, list[dict[str, float | bool]]]:
    rows = []
    for lr, info in sorted(sweep_result["results"].items(), key=lambda item: item[0]):
        metrics = curve_stability_metrics(info["mean_curve"], tail_frac=tail_frac)
        stable = (
            bool(metrics["stable"])
            and metrics["tail_cv"] <= max_tail_cv
            and metrics["up_fraction"] <= max_up_fraction
            and metrics["max_up_rel"] <= max_up_rel
            and metrics["tv_ratio"] <= max_tv_ratio
            and metrics["tail_log_slope"] <= max_tail_log_slope
        )
        rows.append({"lr": lr, "stable": stable, **metrics})

    stable_rows = [row for row in rows if row["stable"]]
    if not stable_rows:
        best = min(rows, key=lambda row: row["tail_mean"])
        return float(best["lr"]), rows

    best_tail_mean = min(row["tail_mean"] for row in stable_rows)
    near_best = [
        row for row in stable_rows if row["tail_mean"] <= best_tail_mean * (1.0 + relative_tol)
    ]
    return float(min(near_best, key=lambda row: row["lr"])["lr"]), rows


def plot_lr_sweep_curves(sweep_result, path: Path, *, log_scale: bool) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    eps = 1e-16
    for lr, info in sorted(sweep_result["results"].items(), key=lambda item: item[0]):
        y = np.maximum(info["mean_curve"], eps) if log_scale else info["mean_curve"]
        label = f"lr={lr:.1e}"
        linewidth = 2.8 if np.isclose(lr, sweep_result["best_lr"]) else 1.5
        if np.isclose(lr, sweep_result["best_lr"]):
            label += " [best]"
        if log_scale:
            ax.semilogy(y, label=label, linewidth=linewidth)
        else:
            ax.plot(y, label=label, linewidth=linewidth)
    ax.set_xlabel("number of block updates")
    ax.set_ylabel("full objective")
    ax.set_title(f"{sweep_result['method_name']} learning-rate sweep")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lr_sweep_summary(sweep_result, path: Path) -> None:
    lrs = np.array(sorted(sweep_result["results"].keys()))
    means = np.array([sweep_result["results"][lr]["final_mean"] for lr in lrs])
    stds = np.array([sweep_result["results"][lr]["final_std"] for lr in lrs])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(lrs, means, yerr=stds, marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("final objective at fixed budget")
    ax.set_title(f"{sweep_result['method_name']}: final objective vs lr")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_trace_curves(path: Path, traces: dict[str, object]) -> None:
    arrays = {}
    for name, trace in traces.items():
        key = name.lower().replace(" ", "_").replace("/", "_")
        arrays[f"{key}_objective_trace"] = trace.objective_trace
        if trace.block_update_objective is not None:
            arrays[f"{key}_block_update_objective"] = trace.block_update_objective
    np.savez(path, **arrays)


def main() -> None:
    args = build_parser().parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    try:
        import torch

        if args.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("--require-cuda was set, but PyTorch does not see a CUDA device.")
    except ImportError as exc:
        raise RuntimeError("SimpleLLMObjective requires PyTorch to be installed.") from exc

    target_chars = args.num_data_batches * args.batch_size * args.seq_len + 1
    max_chars = args.max_chars if args.max_chars is not None else target_chars
    text, data_info = load_text_dataset(
        dataset=args.dataset,
        dataset_file=args.dataset_file,
        data_dir=args.data_dir,
        min_chars=target_chars,
        max_chars=max_chars,
        force_download=args.force_download,
    )

    objective = SimpleLLMObjective(
        text_data=text,
        num_stages=args.num_stages,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
    )
    init_stage_weights = objective.initial_stage_weights(mode="random", seed=args.seed)
    show_progress = not args.no_progress

    scheduler = PipeDream1F1BScheduler(noam=args.noam or args.num_stages)
    timeline = scheduler.generate(
        num_stages=objective.num_stages,
        num_microbatches=args.num_microbatches,
    )
    k_budget = count_backward_ops(timeline)
    gpd_delta = args.gpd_delta if args.gpd_delta is not None else int(args.num_stages**2 - args.num_stages / 2)

    training_batch_indices = build_training_batch_schedule(
        num_dataset_batches=len(objective.get_batches()),
        num_microbatches=args.num_microbatches,
        shuffle_each_epoch=args.shuffle_batches,
        seed=args.seed + 1,
    )

    print("=" * 80)
    print("LLM EXPERIMENT CONFIG")
    print(f"dataset             = {data_info.source}")
    print(f"dataset path        = {data_info.path}")
    print(f"chars / vocab       = {data_info.num_chars} / {data_info.vocab_size}")
    print(
        f"model parameters    = {objective.num_parameters:,} "
        f"({objective.parameter_mebibytes:.2f} MiB of weights)"
    )
    print(f"num data batches    = {len(objective.get_batches())}")
    print(f"num stages          = {args.num_stages}")
    print(f"num microbatches    = {args.num_microbatches}")
    print(f"block-update budget = {k_budget}")
    print(f"batch size / seq    = {args.batch_size} / {args.seq_len}")
    print(f"embed dim / heads   = {args.embed_dim} / {args.num_heads}")
    print(f"GPD delta           = {gpd_delta}")
    print(f"tune stepsizes      = {args.tune_stepsizes}")
    print("=" * 80)
    sys.stdout.flush()

    fig_sched, _ = plot_schedule(timeline, startup_boundary=None, reduce_text=True, max_xtick_labels=24)
    fig_sched.savefig(args.save_dir / "pipedream_schedule.png", dpi=200, bbox_inches="tight")
    plt.close(fig_sched)

    pd_lr = args.pd_lr
    gpd_lr = args.gpd_lr
    tuning_summary = {}

    if args.tune_stepsizes:
        tuning_seeds = parse_int_list(args.tuning_seeds)
        pd_lrs = parse_lr_grid(args.pd_lr_grid)
        gpd_lrs = parse_lr_grid(args.gpd_lr_grid)

        pd_sweep = sweep_learning_rates(
            objective=objective,
            method_name="PipeDream",
            learning_rates=pd_lrs,
            method_factory_builder=lambda lr, seed: make_pd_method(
                lr=lr,
                timeline=timeline,
                training_batch_indices=training_batch_indices,
                init_stage_weights=init_stage_weights,
                show_progress=show_progress,
                name=f"PipeDream lr={lr:.6e}",
            ),
            seeds=tuning_seeds,
            k_budget=k_budget,
            tail_frac=args.stable_tail_frac,
        )
        gpd_sweep = sweep_learning_rates(
            objective=objective,
            method_name="GPD",
            learning_rates=gpd_lrs,
            method_factory_builder=lambda lr, seed: make_gpd_method(
                lr=lr,
                k_budget=k_budget,
                delta=gpd_delta,
                seed=seed,
                training_batch_indices=training_batch_indices,
                init_stage_weights=init_stage_weights,
                show_progress=show_progress,
                name=f"GPD lr={lr:.6e}",
            ),
            seeds=tuning_seeds,
            k_budget=k_budget,
            tail_frac=args.stable_tail_frac,
        )

        stable_pd_lr, pd_lr_table = select_stable_learning_rate(
            pd_sweep,
            tail_frac=args.stable_tail_frac,
        )
        stable_gpd_lr, gpd_lr_table = select_stable_learning_rate(
            gpd_sweep,
            tail_frac=args.stable_tail_frac,
        )

        if args.lr_selection == "best-final":
            pd_lr = float(pd_sweep["best_lr"])
            gpd_lr = float(gpd_sweep["best_lr"])
        else:
            pd_lr = stable_pd_lr
            gpd_lr = stable_gpd_lr

        plot_lr_sweep_curves(pd_sweep, args.save_dir / "pd_lr_sweep_log.png", log_scale=True)
        plot_lr_sweep_summary(pd_sweep, args.save_dir / "pd_lr_sweep_summary.png")
        plot_lr_sweep_curves(gpd_sweep, args.save_dir / "gpd_lr_sweep_log.png", log_scale=True)
        plot_lr_sweep_summary(gpd_sweep, args.save_dir / "gpd_lr_sweep_summary.png")

        tuning_summary = {
            "pd_final_point_best_lr": float(pd_sweep["best_lr"]),
            "pd_stable_lr": float(stable_pd_lr),
            "pd_selected_lr": float(pd_lr),
            "pd_lr_table": pd_lr_table,
            "gpd_final_point_best_lr": float(gpd_sweep["best_lr"]),
            "gpd_stable_lr": float(stable_gpd_lr),
            "gpd_selected_lr": float(gpd_lr),
            "gpd_lr_table": gpd_lr_table,
            "selection_policy": args.lr_selection,
        }
        print(f"LR selection policy  = {args.lr_selection}")
        print(f"PipeDream best-final = {float(pd_sweep['best_lr']):.6e}, stable = {stable_pd_lr:.6e}")
        print(f"GPD best-final       = {float(gpd_sweep['best_lr']):.6e}, stable = {stable_gpd_lr:.6e}")
        print(f"Selected PipeDream lr = {pd_lr:.6e}")
        print(f"Selected GPD lr       = {gpd_lr:.6e}")

    methods = [
        make_pd_method(
            lr=pd_lr,
            timeline=timeline,
            training_batch_indices=training_batch_indices,
            init_stage_weights=init_stage_weights,
            show_progress=show_progress,
            name=f"PipeDream lr={pd_lr:.1e}",
        ),
        make_gpd_method(
            lr=gpd_lr,
            k_budget=k_budget,
            delta=gpd_delta,
            seed=args.gpd_seed,
            training_batch_indices=training_batch_indices,
            init_stage_weights=init_stage_weights,
            show_progress=show_progress,
            name=f"GPD delta={gpd_delta} lr={gpd_lr:.1e}",
        ),
    ]

    comparison = compare_methods(ExperimentRunner(objective), methods)

    fig_lin, ax_lin = plt.subplots(figsize=(8, 4.5))
    plot_block_update_comparison(comparison, log_scale=False, ax=ax_lin)
    ax_lin.set_title("PipeDream vs GPD on SimpleLLM")
    ax_lin.grid(True, alpha=0.25)
    fig_lin.tight_layout()
    fig_lin.savefig(args.save_dir / "comparison_linear.png", dpi=200, bbox_inches="tight")
    plt.close(fig_lin)

    fig_log, ax_log = plt.subplots(figsize=(8, 4.5))
    plot_block_update_comparison(comparison, log_scale=True, ax=ax_log)
    ax_log.set_title("PipeDream vs GPD on SimpleLLM")
    ax_log.grid(True, alpha=0.25)
    fig_log.tight_layout()
    fig_log.savefig(args.save_dir / "comparison_log.png", dpi=200, bbox_inches="tight")
    plt.close(fig_log)

    save_trace_curves(args.save_dir / "curves.npz", comparison.traces)

    summary = {
        "dataset": {
            "source": data_info.source,
            "path": str(data_info.path) if data_info.path is not None else None,
            "num_chars": data_info.num_chars,
            "vocab_size": data_info.vocab_size,
        },
        "num_data_batches": len(objective.get_batches()),
        "num_stages": args.num_stages,
        "num_microbatches": args.num_microbatches,
        "k_budget": k_budget,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "pd_lr": pd_lr,
        "gpd_lr": gpd_lr,
        "gpd_delta": gpd_delta,
        "final_objectives": {
            name: trace.final_objective for name, trace in comparison.traces.items()
        },
        "tuning": tuning_summary,
    }
    (args.save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFINAL OBJECTIVES")
    for name, trace in comparison.traces.items():
        print(f"{name:>24s}: {trace.final_objective:.6e}")

    print(f"\nSaved outputs to: {args.save_dir.resolve()}")
    print("  - pipedream_schedule.png")
    print("  - comparison_linear.png")
    print("  - comparison_log.png")
    print("  - curves.npz")
    print("  - summary.json")


if __name__ == "__main__":
    main()
