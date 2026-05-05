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

from src.methods.localSGD import LocalMinibatchSGD1F1BMethod
from src.methods.pipedream import PipeDreamMethod
from src.objectives.simple_llm import SimpleLLMObjective
from src.schedulers.independent_local_sgd_pipeline import IndependentLocalSGDScheduler
from src.schedulers.pipedream_1f1b import PipeDream1F1BScheduler, plot_schedule
from src.utils.batching import build_training_batch_schedule
from src.utils.llm_data import load_text_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare PipeDream and LocalSGD on SimpleLLMObjective using wall-clock time."
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=Path, default=Path("results/llm_pd_vs_sgd"))

    parser.add_argument("--dataset", choices=["toy", "tiny_shakespeare", "file"], default="toy")
    parser.add_argument("--dataset-file", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data/llm"))
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--num-data-batches", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=None)

    parser.add_argument("--num-stages", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--require-cuda", action="store_true")

    parser.add_argument(
        "--pd-num-microbatches",
        type=int,
        default=16,
        help="Used to define the default time budget when --target-time-steps is not set.",
    )
    parser.add_argument(
        "--target-time-steps",
        type=int,
        default=None,
        help="If set, both methods choose the smallest microbatch count whose timeline reaches this many steps.",
    )
    parser.add_argument("--pd-noam", type=int, default=None)
    parser.add_argument("--local-num-runs", type=int, default=None)
    parser.add_argument("--local-steps", type=int, default=5)
    parser.add_argument("--shuffle-batches", action="store_true")

    parser.add_argument("--pd-lr", type=float, default=6.25e-02)
    parser.add_argument("--local-sgd-lr", type=float, default=6.25e-02)
    parser.add_argument("--tune-stepsizes", action="store_true")
    parser.add_argument("--tuning-seeds", type=str, default="0")
    parser.add_argument("--pd-lr-grid", type=str, default="pow2:-5:0")
    parser.add_argument("--local-sgd-lr-grid", type=str, default="pow2:-5:0")
    parser.add_argument(
        "--lr-selection",
        choices=["best-final", "stable"],
        default="stable",
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


def find_optimal_microbatches(scheduler_gen_fn: Callable[[int], object], target_length: int) -> tuple[int, object]:
    high = 1
    while len(scheduler_gen_fn(high)) < target_length:
        high *= 2

    low = max(1, high // 2)
    best_n = high
    best_timeline = scheduler_gen_fn(high)

    while low <= high:
        mid = (low + high) // 2
        timeline = scheduler_gen_fn(mid)
        if len(timeline) >= target_length:
            best_n = mid
            best_timeline = timeline
            high = mid - 1
        else:
            low = mid + 1

    return best_n, best_timeline


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
        log_full_objective=False,
        log_forward_loss=True,
        log_grad_norms=False,
        store_final_weight=False,
        show_progress=show_progress,
        name=name,
    )


def make_local_sgd_method(
    *,
    lr: float,
    timeline,
    training_batch_indices: list[int],
    init_stage_weights: list[np.ndarray],
    num_runs: int,
    local_steps: int,
    show_progress: bool,
    name: str,
) -> LocalMinibatchSGD1F1BMethod:
    return LocalMinibatchSGD1F1BMethod(
        timeline=timeline,
        learning_rate=lr,
        training_batch_indices=training_batch_indices,
        num_runs=num_runs,
        local_steps=local_steps,
        init_stage_weights=init_stage_weights,
        log_full_objective=False,
        log_forward_loss=True,
        log_grad_norms=False,
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


def time_curve_from_trace(trace) -> np.ndarray:
    curve = np.asarray(trace.objective_trace, dtype=float)
    if np.any(np.isfinite(curve)):
        return curve

    times = np.asarray(trace.forward_loss_time_trace, dtype=int)
    losses = np.asarray(trace.forward_loss_trace, dtype=float)
    if len(times) == 0:
        return curve

    rebuilt = np.full(len(curve), np.nan, dtype=float)
    latest = np.nan
    loss_by_time = {int(t): float(loss) for t, loss in zip(times, losses)}
    for t in range(len(rebuilt)):
        if t in loss_by_time:
            latest = loss_by_time[t]
        rebuilt[t] = latest
    return rebuilt


def final_finite_value(curve: np.ndarray) -> float:
    finite = curve[np.isfinite(curve)]
    if len(finite) == 0:
        return np.inf
    return float(finite[-1])


def aggregate_time_curves(traces, tail_frac: float = 0.2) -> dict[str, object]:
    curves = [time_curve_from_trace(trace) for trace in traces]
    length = min(len(curve) for curve in curves)
    stacked = np.stack([curve[:length] for curve in curves], axis=0)

    finite_counts = np.sum(np.isfinite(stacked), axis=0)
    sums = np.nansum(stacked, axis=0)
    means = np.full(length, np.nan, dtype=float)
    means[finite_counts > 0] = sums[finite_counts > 0] / finite_counts[finite_counts > 0]

    stds = np.full(length, np.nan, dtype=float)
    for idx in np.where(finite_counts > 0)[0]:
        vals = stacked[:, idx]
        stds[idx] = float(np.std(vals[np.isfinite(vals)]))
    tail_len = max(1, int(np.ceil(tail_frac * length)))
    tail = means[-tail_len:]

    return {
        "mean": means,
        "std": stds,
        "final_mean": final_finite_value(means),
        "final_std": float(np.nanstd(stacked[:, -1])) if np.any(np.isfinite(stacked[:, -1])) else np.inf,
        "tail_mean": float(np.nanmean(tail)) if np.any(np.isfinite(tail)) else np.inf,
        "tail_std": float(np.nanstd(tail)) if np.any(np.isfinite(tail)) else np.inf,
    }


def sweep_learning_rates(
    *,
    objective,
    method_name: str,
    learning_rates: np.ndarray,
    method_factory_builder: Callable[[float, int], object],
    seeds: list[int],
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
        agg = aggregate_time_curves(traces, tail_frac=tail_frac)
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
    y = y[np.isfinite(y)]
    if len(y) < 3:
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


def plot_time_curves(curves: dict[str, np.ndarray], path: Path, *, log_scale: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    eps = 1e-16
    for name, curve in curves.items():
        x = np.arange(len(curve))
        y = np.maximum(curve, eps) if log_scale else curve
        if log_scale:
            ax.semilogy(x, y, linewidth=2.2, label=name)
        else:
            ax.plot(x, y, linewidth=2.2, label=name)
    ax.set_xlabel("time step")
    ax.set_ylabel("last-stage forward loss")
    ax.set_title("PipeDream vs LocalSGD on SimpleLLM")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lr_sweep_curves(sweep_result, path: Path, *, log_scale: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    eps = 1e-16
    for lr, info in sorted(sweep_result["results"].items(), key=lambda item: item[0]):
        curve = info["mean_curve"]
        y = np.maximum(curve, eps) if log_scale else curve
        label = f"lr={lr:.1e}"
        linewidth = 2.8 if np.isclose(lr, sweep_result["best_lr"]) else 1.5
        if np.isclose(lr, sweep_result["best_lr"]):
            label += " [best]"
        if log_scale:
            ax.semilogy(y, label=label, linewidth=linewidth)
        else:
            ax.plot(y, label=label, linewidth=linewidth)
    ax.set_xlabel("time step")
    ax.set_ylabel("last-stage forward loss")
    ax.set_title(f"{sweep_result['method_name']} learning-rate sweep")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lr_sweep_summary(sweep_result, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lrs = np.array(sorted(sweep_result["results"].keys()))
    means = np.array([sweep_result["results"][lr]["final_mean"] for lr in lrs])
    stds = np.array([sweep_result["results"][lr]["final_std"] for lr in lrs])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(lrs, means, yerr=stds, marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("final forward loss")
    ax.set_title(f"{sweep_result['method_name']}: final loss vs lr")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_trace_curves(path: Path, traces: dict[str, object], curves: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, trace in traces.items():
        key = name.lower().replace(" ", "_").replace("/", "_")
        arrays[f"{key}_time_curve"] = curves[name]
        arrays[f"{key}_forward_loss"] = trace.forward_loss_trace
        arrays[f"{key}_forward_loss_time"] = trace.forward_loss_time_trace
    np.savez(path, **arrays)


def save_sweep_summary(path: Path, sweeps: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, sweep in sweeps.items():
        summary[name] = {
            "best_lr": float(sweep["best_lr"]),
            "results": {
                f"{lr:.16e}": {
                    "final_mean": float(info["final_mean"]),
                    "final_std": float(info["final_std"]),
                    "tail_mean": float(info["tail_mean"]),
                    "tail_std": float(info["tail_std"]),
                }
                for lr, info in sweep["results"].items()
            },
        }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


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

    local_num_runs = args.local_num_runs if args.local_num_runs is not None else args.num_stages
    pd_scheduler = PipeDream1F1BScheduler(noam=args.pd_noam or args.num_stages)
    local_scheduler = IndependentLocalSGDScheduler(
        num_runs=local_num_runs,
        local_steps=args.local_steps,
    )

    if args.target_time_steps is None:
        pd_num_microbatches = args.pd_num_microbatches
        pd_timeline = pd_scheduler.generate(args.num_stages, pd_num_microbatches)
        target_time_steps = len(pd_timeline)
    else:
        target_time_steps = args.target_time_steps
        pd_num_microbatches, pd_timeline = find_optimal_microbatches(
            lambda n: pd_scheduler.generate(args.num_stages, n),
            target_time_steps,
        )

    local_num_microbatches, local_timeline = find_optimal_microbatches(
        lambda n: local_scheduler.generate(args.num_stages, n),
        target_time_steps,
    )

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
    num_dataset_batches = len(objective.get_batches())
    show_progress = not args.no_progress

    pd_training_batch_indices = build_training_batch_schedule(
        num_dataset_batches=num_dataset_batches,
        num_microbatches=pd_num_microbatches,
        shuffle_each_epoch=args.shuffle_batches,
        seed=args.seed + 1,
    )
    local_training_batch_indices = build_training_batch_schedule(
        num_dataset_batches=num_dataset_batches,
        num_microbatches=local_num_microbatches,
        shuffle_each_epoch=args.shuffle_batches,
        seed=args.seed + 1,
    )

    print("=" * 80)
    print("LLM PIPE DREAM VS LOCAL SGD CONFIG")
    print(f"dataset                  = {data_info.source}")
    print(f"dataset path             = {data_info.path}")
    print(f"chars / vocab            = {data_info.num_chars} / {data_info.vocab_size}")
    print(
        f"model parameters         = {objective.num_parameters:,} "
        f"({objective.parameter_mebibytes:.2f} MiB of weights)"
    )
    print(f"num data batches         = {num_dataset_batches}")
    print(f"num stages               = {args.num_stages}")
    print(f"target time steps        = {target_time_steps}")
    print(f"PipeDream timeline       = {len(pd_timeline)} steps, N={pd_num_microbatches}")
    print(f"LocalSGD timeline        = {len(local_timeline)} steps, N={local_num_microbatches}")
    print(f"LocalSGD M / K           = {local_num_runs} / {args.local_steps}")
    print(f"batch size / seq         = {args.batch_size} / {args.seq_len}")
    print(f"embed dim / heads        = {args.embed_dim} / {args.num_heads}")
    print(f"tune stepsizes           = {args.tune_stepsizes}")
    print("=" * 80)
    sys.stdout.flush()

    args.save_dir.mkdir(parents=True, exist_ok=True)

    fig_pd_sched, _ = plot_schedule(pd_timeline, startup_boundary=None, reduce_text=True, max_xtick_labels=24)
    (args.save_dir / "pipedream_schedule.png").parent.mkdir(parents=True, exist_ok=True)
    fig_pd_sched.savefig(args.save_dir / "pipedream_schedule.png", dpi=200, bbox_inches="tight")
    plt.close(fig_pd_sched)

    fig_local_sched, _ = plot_schedule(local_timeline, startup_boundary=None, reduce_text=True, max_xtick_labels=24)
    (args.save_dir / "local_sgd_schedule.png").parent.mkdir(parents=True, exist_ok=True)
    fig_local_sched.savefig(args.save_dir / "local_sgd_schedule.png", dpi=200, bbox_inches="tight")
    plt.close(fig_local_sched)

    pd_lr = args.pd_lr
    local_lr = args.local_sgd_lr
    tuning_summary = {}

    if args.tune_stepsizes:
        tuning_seeds = parse_int_list(args.tuning_seeds)
        pd_lrs = parse_lr_grid(args.pd_lr_grid)
        local_lrs = parse_lr_grid(args.local_sgd_lr_grid)

        pd_sweep = sweep_learning_rates(
            objective=objective,
            method_name="PipeDream",
            learning_rates=pd_lrs,
            method_factory_builder=lambda lr, seed: make_pd_method(
                lr=lr,
                timeline=pd_timeline,
                training_batch_indices=pd_training_batch_indices,
                init_stage_weights=init_stage_weights,
                show_progress=show_progress,
                name=f"PipeDream lr={lr:.6e}",
            ),
            seeds=tuning_seeds,
            tail_frac=args.stable_tail_frac,
        )
        local_sweep = sweep_learning_rates(
            objective=objective,
            method_name="LocalSGD",
            learning_rates=local_lrs,
            method_factory_builder=lambda lr, seed: make_local_sgd_method(
                lr=lr,
                timeline=local_timeline,
                training_batch_indices=local_training_batch_indices,
                init_stage_weights=init_stage_weights,
                num_runs=local_num_runs,
                local_steps=args.local_steps,
                show_progress=show_progress,
                name=f"LocalSGD lr={lr:.6e}",
            ),
            seeds=tuning_seeds,
            tail_frac=args.stable_tail_frac,
        )

        stable_pd_lr, pd_lr_table = select_stable_learning_rate(
            pd_sweep,
            tail_frac=args.stable_tail_frac,
        )
        stable_local_lr, local_lr_table = select_stable_learning_rate(
            local_sweep,
            tail_frac=args.stable_tail_frac,
        )

        if args.lr_selection == "best-final":
            pd_lr = float(pd_sweep["best_lr"])
            local_lr = float(local_sweep["best_lr"])
        else:
            pd_lr = stable_pd_lr
            local_lr = stable_local_lr

        save_sweep_summary(
            args.save_dir / "stepsize_sweeps_summary.json",
            {"PipeDream": pd_sweep, "LocalSGD": local_sweep},
        )

        plot_lr_sweep_curves(pd_sweep, args.save_dir / "pd_lr_sweep_log.png", log_scale=True)
        plot_lr_sweep_summary(pd_sweep, args.save_dir / "pd_lr_sweep_summary.png")
        plot_lr_sweep_curves(local_sweep, args.save_dir / "local_sgd_lr_sweep_log.png", log_scale=True)
        plot_lr_sweep_summary(local_sweep, args.save_dir / "local_sgd_lr_sweep_summary.png")

        tuning_summary = {
            "pd_final_point_best_lr": float(pd_sweep["best_lr"]),
            "pd_stable_lr": float(stable_pd_lr),
            "pd_selected_lr": float(pd_lr),
            "pd_lr_table": pd_lr_table,
            "local_sgd_final_point_best_lr": float(local_sweep["best_lr"]),
            "local_sgd_stable_lr": float(stable_local_lr),
            "local_sgd_selected_lr": float(local_lr),
            "local_sgd_lr_table": local_lr_table,
            "selection_policy": args.lr_selection,
        }
        print(f"LR selection policy  = {args.lr_selection}")
        print(f"PipeDream best-final = {float(pd_sweep['best_lr']):.6e}, stable = {stable_pd_lr:.6e}")
        print(f"LocalSGD best-final  = {float(local_sweep['best_lr']):.6e}, stable = {stable_local_lr:.6e}")
        print(f"Selected PipeDream lr = {pd_lr:.6e}")
        print(f"Selected LocalSGD lr  = {local_lr:.6e}")

    methods = [
        make_pd_method(
            lr=pd_lr,
            timeline=pd_timeline,
            training_batch_indices=pd_training_batch_indices,
            init_stage_weights=init_stage_weights,
            show_progress=show_progress,
            name=f"PipeDream lr={pd_lr:.1e}",
        ),
        make_local_sgd_method(
            lr=local_lr,
            timeline=local_timeline,
            training_batch_indices=local_training_batch_indices,
            init_stage_weights=init_stage_weights,
            num_runs=local_num_runs,
            local_steps=args.local_steps,
            show_progress=show_progress,
            name=f"LocalSGD M={local_num_runs} K={args.local_steps} lr={local_lr:.1e}",
        ),
    ]

    traces = {method.name: method.run(objective) for method in methods}
    curves = {name: time_curve_from_trace(trace) for name, trace in traces.items()}

    plot_time_curves(curves, args.save_dir / "comparison_time_linear.png", log_scale=False)
    plot_time_curves(curves, args.save_dir / "comparison_time_log.png", log_scale=True)
    save_trace_curves(args.save_dir / "curves.npz", traces, curves)

    summary = {
        "dataset": {
            "source": data_info.source,
            "path": str(data_info.path) if data_info.path is not None else None,
            "num_chars": data_info.num_chars,
            "vocab_size": data_info.vocab_size,
        },
        "num_data_batches": num_dataset_batches,
        "num_stages": args.num_stages,
        "target_time_steps": target_time_steps,
        "pd_num_microbatches": pd_num_microbatches,
        "pd_timeline_steps": len(pd_timeline),
        "local_sgd_num_microbatches": local_num_microbatches,
        "local_sgd_timeline_steps": len(local_timeline),
        "local_sgd_num_runs": local_num_runs,
        "local_sgd_local_steps": args.local_steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "pd_lr": pd_lr,
        "local_sgd_lr": local_lr,
        "final_forward_losses": {
            name: final_finite_value(curve) for name, curve in curves.items()
        },
        "tuning": tuning_summary,
    }
    (args.save_dir / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFINAL FORWARD LOSSES")
    for name, curve in curves.items():
        print(f"{name:>34s}: {final_finite_value(curve):.6e}")

    print(f"\nSaved outputs to: {args.save_dir.resolve()}")
    print("  - pipedream_schedule.png")
    print("  - local_sgd_schedule.png")
    print("  - comparison_time_linear.png")
    print("  - comparison_time_log.png")
    print("  - curves.npz")
    print("  - summary.json")


if __name__ == "__main__":
    main()
