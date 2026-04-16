from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.methods.localSGD import LocalMinibatchSGD1F1BMethod
from src.schedulers.independent_local_sgd_pipeline import IndependentLocalSGDScheduler
from src.utils.batching import build_training_batch_schedule
from src.objectives.quadratic import QuadraticObjective
from src.schedulers.pipedream_1f1b import PipeDream1F1BScheduler, print_schedule, plot_schedule
from src.methods.pipedream import PipeDreamMethod
from src.methods.gpd import GPDMethod, build_pipedream_exact_delays
from src.methods.sgd import SGDMethod
from src.experiments.runner import ExperimentRunner
from src.experiments.compare_methods import compare_methods
from src.plotting.convergence import plot_block_update_comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PipeDream / GPD / SGD debug experiment.")
    parser.add_argument("--num-parameters", type=int, default=100)
    parser.add_argument("--num-stages", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-microbatches", type=int, default=300)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.0)

    parser.add_argument("--pd-lr", type=float, default=None)
    parser.add_argument("--gpd-lr", type=float, default=None)
    parser.add_argument("--sgd-lr", type=float, default=None)

    parser.add_argument("--gpd-delta", type=int, default=None)
    parser.add_argument("--gpd-seed", type=int, default=123)
    parser.add_argument("--sgd-seed", type=int, default=456)

    parser.add_argument("--save-dir", type=str, default="results/debug_main")
    parser.add_argument("--no-show", action="store_true", default=True)
    return parser



def extract_pipedream_backward_ops(timeline):
    backward_ops = []
    for ops_this_step in timeline:
        for stage, op in enumerate(ops_this_step):
            if op is None:
                continue
            kind, mb = op
            if kind == "B":
                backward_ops.append((stage, mb))
    return backward_ops


def compute_forward_history_indices_from_timeline(timeline, num_microbatches, num_stages):
    """
    Compute forward_history_indices[mb, stage] using only the timeline.

    Interpretation:
    forward_history_indices[mb, stage] = number of backward updates that
    have already happened before the forward event (stage, mb).
    """
    forward_history_indices = -np.ones((num_microbatches, num_stages), dtype=int)

    backward_count = 0
    for ops_this_step in timeline:
        for stage, op in enumerate(ops_this_step):
            if op is None:
                continue

            kind, mb = op
            if kind == "F":
                forward_history_indices[mb, stage] = backward_count
            elif kind == "B":
                backward_count += 1
            else:
                raise ValueError(f"Unknown op kind: {kind}")

    if np.any(forward_history_indices < 0):
        bad = np.argwhere(forward_history_indices < 0)
        raise RuntimeError(f"Missing forward history indices for entries: {bad[:10]}")

    return forward_history_indices


def build_pipedream_exact_delays_from_timeline(timeline, num_microbatches, num_stages):
    """
    Build exact global-history delays for GPD directly from the timeline.
    No PipeDream optimization run is needed.
    """
    forward_history_indices = compute_forward_history_indices_from_timeline(
        timeline=timeline,
        num_microbatches=num_microbatches,
        num_stages=num_stages,
    )

    backward_ops = extract_pipedream_backward_ops(timeline)
    K = len(backward_ops)

    exact_delays = np.zeros((K, num_stages), dtype=int)
    for k, (_, mb) in enumerate(backward_ops):
        exact_delays[k] = k - forward_history_indices[mb]

    return exact_delays


def plot_delay_stats_vs_num_stages(args, max_num_stages=100) -> None:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    scheduler = PipeDream1F1BScheduler(noam=None)

    stage_values = []
    mean_delay_values = []
    p10_delay_values = []
    max_delay_values = []

    for num_stages in range(2, max_num_stages + 1):
        num_microbatches = num_stages * 10

        timeline = scheduler.generate(
            num_stages=num_stages,
            num_microbatches=num_microbatches,
        )

        pipedream_exact_delays = build_pipedream_exact_delays_from_timeline(
            timeline=timeline,
            num_microbatches=num_microbatches,
            num_stages=num_stages,
        )

        pipedream_exact_delays = pipedream_exact_delays[pipedream_exact_delays.shape[0]//2:]

        flat_delays = pipedream_exact_delays.reshape(-1)

        stage_values.append(num_stages)
        mean_delay_values.append(float(np.mean(flat_delays)))
        p10_delay_values.append(float(np.percentile(flat_delays, 10)))
        max_delay_values.append(float(np.max(flat_delays)))

        print(
            f"num_stages={num_stages:4d} | "
            f"mean={mean_delay_values[-1]:10.4f} | "
            f"p10={p10_delay_values[-1]:10.4f} | "
            f"max={max_delay_values[-1]:10.4f}"
        )

    stage_values = np.asarray(stage_values)
    mean_delay_values = np.asarray(mean_delay_values)
    p10_delay_values = np.asarray(p10_delay_values) + 1
    max_delay_values = np.asarray(max_delay_values)

    eps = 1e-12

    low_plot = np.maximum(p10_delay_values, eps)
    mean_plot = np.maximum(mean_delay_values, eps)
    max_plot = np.maximum(max_delay_values, eps)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(stage_values, mean_plot, label=r"mean $k-j_k^{(s)}$")
    ax.fill_between(stage_values, low_plot, max_plot, alpha=0.2, label=r"10th percentile - max range of $k-j_k^{(s)}$")

    ax.set_yscale("log")
    ax.set_xlabel(r"Number of stages $S$")
    ax.set_ylabel(r"Staleness $(k-j_k^{(s)})$")
    ax.set_title(r"PipeDream exact delay statistics vs number of stages")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_path = save_dir / "delay_stats_vs_num_stages_log_only_stable.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    stats_path = save_dir / "delay_stats_vs_num_stages_only_stable.npz"
    np.savez(
        stats_path,
        stage_values=stage_values,
        mean_delay_values=mean_delay_values,
        p10_delay_values=p10_delay_values,
        max_delay_values=max_delay_values,
    )

    print(f"\nSaved plot to: {out_path.resolve()}")
    print(f"Saved arrays to: {stats_path.resolve()}")



def plot_delay_stats_vs_num_stages_from_file_with_quadratic_fit(
    args,
    filename: str | None = None,
) -> None:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        stats_path = save_dir / "delay_stats_vs_num_stages_only_stable.npz"
    else:
        stats_path = Path(filename)

    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    data = np.load(stats_path)

    stage_values = data["stage_values"]
    mean_delay_values = data["mean_delay_values"]
    p10_delay_values = data["p10_delay_values"]
    max_delay_values = data["max_delay_values"]

    coeffs1 = np.polyfit(stage_values, max_delay_values, deg=2)
    a1, b1, c1 = coeffs1
    max_quadratic_fit = np.polyval(coeffs1, stage_values)

    coeffs2 = np.polyfit(stage_values, mean_delay_values, deg=2)
    a2, b2, c2 = coeffs2
    mean_quadratic_fit = np.polyval(coeffs2, stage_values)

    eps = 1e-12
    low_plot = np.maximum(p10_delay_values, eps)
    mean_plot = np.maximum(mean_delay_values, eps)
    max_plot = np.maximum(max_delay_values, eps)
    fit_plot1 = np.maximum(max_quadratic_fit, eps)
    fit_plot2 = np.maximum(mean_quadratic_fit, eps)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    ax.fill_between(
        stage_values,
        low_plot,
        max_plot,
        alpha=0.18,
        label=r"10th percentile – max range",
    )

    ax.plot(
        stage_values,
        mean_plot,
        linewidth=2.2,
        label=r"Mean staleness",
    )

    ax.plot(
        stage_values,
        fit_plot2,
        linestyle="--",
        linewidth=2.2,
        label=rf"Quadratic fit (${a2:.1f} \cdot S^2$)",
    )

    ax.plot(
        stage_values,
        fit_plot1,
        linestyle="--",
        linewidth=2.2,
        label=rf"Quadratic fit (${a1:.1f} \cdot S^2 {b1:.1f} \cdot S$)",
    )

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(r"Number of stages $S$")
    ax.set_ylabel(r"Staleness $(k-j_k^{(s)})$")
    ax.set_title(r"PipeDream exact delay statistics vs number of stages")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=True)



    fig.tight_layout()
    out_path = save_dir / "delay_stats_vs_num_stages_log_with_quadratic_fit.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    fit_path = save_dir / "delay_stats_quadratic_fit.npz"
    '''
    np.savez(
        fit_path,
        coeffs=coeffs,
        stage_values=stage_values,
        mean_delay_values=mean_delay_values,
        mean_quadratic_fit=mean_quadratic_fit,
    )
    '''

    print(f"Loaded arrays from: {stats_path.resolve()}")
    print(f"Saved plot to: {out_path.resolve()}")
    print(f"Saved fit data to: {fit_path.resolve()}")
    print(f"S^2 coefficient: {a1:.8e}")
    print(f"S^2 coefficient: {a2:.8e}")


def main() -> None:
    args = build_parser().parse_args()
    #plot_delay_stats_vs_num_stages(args, max_num_stages=150)
    #plot_delay_stats_vs_num_stages_from_file_with_quadratic_fit(args)

    num_examples = args.batch_size * args.num_microbatches // args.num_epochs

    objective = QuadraticObjective.synthetic(
        num_examples=num_examples,
        num_parameters=args.num_parameters,
        num_stages=args.num_stages,
        batch_size=args.batch_size,
        seed=args.seed,
        noise_std=args.noise_std,
    )

    init_stage_weights = objective.initial_stage_weights(mode="zeros", seed=args.seed)

    scheduler = IndependentLocalSGDScheduler(num_runs=4, local_steps=5)
    timeline = scheduler.generate(
        num_stages=args.num_stages,
        num_microbatches=args.num_microbatches,
    )

    num_dataset_batches = len(objective.get_batches())
    training_batch_indices = build_training_batch_schedule(
        num_dataset_batches=num_dataset_batches,
        num_microbatches=args.num_microbatches,
        shuffle_each_epoch=False,  # or True if you want shuffled epochs
        seed=args.seed + 1,
    )

    local_sgd_method = LocalMinibatchSGD1F1BMethod(
        timeline=timeline,
        learning_rate=1e-2,
        training_batch_indices=training_batch_indices,
        init_stage_weights=init_stage_weights,
        name="Local SGD",
        num_runs=4,
        local_steps=5,
    )

    runner = ExperimentRunner(objective=objective)
    sgd_trace = runner.run(local_sgd_method)

    print(sgd_trace)

    '''
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    num_examples = args.batch_size * args.num_microbatches // args.num_epochs

    objective = QuadraticObjective.synthetic(
        num_examples=num_examples,
        num_parameters=args.num_parameters,
        num_stages=args.num_stages,
        batch_size=args.batch_size,
        seed=args.seed,
        noise_std=args.noise_std,
    )

    L = objective.smoothness_constant
    init_stage_weights = objective.initial_stage_weights(mode="zeros", seed=args.seed)

    pd_lr = args.pd_lr if args.pd_lr is not None else 0.5 / L
    gpd_lr = args.gpd_lr if args.gpd_lr is not None else 0.5 / L
    sgd_lr = args.sgd_lr if args.sgd_lr is not None else 0.8 / L
    gpd_delta = args.gpd_delta if args.gpd_delta is not None else args.num_stages

    print("=" * 80)
    print("DEBUG RUN CONFIG")
    print(f"num_examples      = {num_examples}")
    print(f"num_parameters    = {args.num_parameters}")
    print(f"num_stages        = {args.num_stages}")
    print(f"batch_size        = {args.batch_size}")
    print(f"num_microbatches  = {args.num_microbatches}")
    print(f"num_epochs         = {args.num_epochs}")
    print(f"seed              = {args.seed}")
    print(f"noise_std         = {args.noise_std}")
    print(f"L                 = {L:.6e}")
    print(f"PipeDream lr      = {pd_lr:.6e}")
    print(f"GPD lr            = {gpd_lr:.6e}")
    print(f"SGD lr            = {sgd_lr:.6e}")
    print(f"GPD delta         = {gpd_delta}")
    print("=" * 80)

    scheduler = PipeDream1F1BScheduler(noam=args.num_stages)
    timeline = scheduler.generate(
        num_stages=args.num_stages,
        num_microbatches=args.num_microbatches,
    )

    print("\nPipeDream schedule:")
    print_schedule(timeline)
    fig_sched, ax_sched = plot_schedule(timeline, startup_boundary=args.num_stages, reduce_text=True)
    fig_sched.savefig(save_dir / "schedule.png", dpi=200, bbox_inches="tight")

    num_dataset_batches = len(objective.get_batches())
    training_batch_indices = build_training_batch_schedule(
        num_dataset_batches=num_dataset_batches,
        num_microbatches=args.num_microbatches,
        shuffle_each_epoch=False,  # or True if you want shuffled epochs
        seed=args.seed + 1,
    )

    pipedream_method = PipeDreamMethod(
        timeline=timeline,
        learning_rate=pd_lr,
        training_batch_indices=training_batch_indices,
        init_stage_weights=init_stage_weights,
        name="PipeDream",
    )

    # Use the PipeDream block-update budget as the common budget.
    runner = ExperimentRunner(objective=objective)
    pd_trace = runner.run(pipedream_method)
    K_budget = len(pd_trace.block_update_objective)

    forward_history_indices = pd_trace.metadata["forward_history_indices"]
    pipedream_exact_delays = build_pipedream_exact_delays(
        timeline=timeline,
        forward_history_indices=forward_history_indices,
    )

    print("forward_history_indices shape:", forward_history_indices.shape)
    print("pipedream_exact_delays shape:", pipedream_exact_delays.shape)
    print("first 5 exact delay rows:\n", pipedream_exact_delays[:30])

    gpd_method = GPDMethod(
        num_iterations=K_budget,
        learning_rate=gpd_lr,
        delta=20,
        seed=args.gpd_seed,
        stage_sampling="pipedream",
        #batch_sampling="pipedream",
        #stale_sampling="pipedream",
        timeline=timeline,
        training_batch_indices=training_batch_indices,
        #pipedream_exact_delays=pipedream_exact_delays,
        init_stage_weights=init_stage_weights,
        name="GPD",
    )

    sgd_method = SGDMethod(
        num_iterations=int(np.ceil(K_budget / args.num_stages)),
        learning_rate=sgd_lr,
        seed=args.sgd_seed,
        batch_sampling="uniform",
        training_batch_indices=training_batch_indices,
        init_stage_weights=init_stage_weights,
        name="SGD",
    )

    methods = [pipedream_method, gpd_method, sgd_method]
    comparison = compare_methods(runner, methods)

    print("\nFINAL OBJECTIVES")
    for name, trace in comparison.traces.items():
        print(f"{name:>10s}: final objective = {trace.final_objective:.6e}")

    print("\nBLOCK-UPDATE CURVE LENGTHS")
    for name, trace in comparison.traces.items():
        curve_len = 0 if trace.block_update_objective is None else len(trace.block_update_objective)
        print(f"{name:>10s}: {curve_len}")

    pd_trace = comparison.traces["PipeDream"]
    if pd_trace.backward_staleness is not None:
        vals = pd_trace.backward_staleness[pd_trace.backward_staleness >= 0]
        if len(vals) > 0:
            print("\nPipeDream staleness stats")
            print(f"  mean = {vals.mean():.4f}")
            print(f"  max  = {vals.max():.4f}")

    gpd_trace = comparison.traces["GPD"]
    if gpd_trace.stale_distance_trace is not None and len(gpd_trace.stale_distance_trace) > 0:
        print("\nGPD stale-distance stats")
        print(f"  mean = {gpd_trace.stale_distance_trace.mean():.6e}")
        print(f"  max  = {gpd_trace.stale_distance_trace.max():.6e}")

    fig_cmp_lin, ax_cmp_lin = plt.subplots(figsize=(8, 4))
    plot_block_update_comparison(comparison, log_scale=False, ax=ax_cmp_lin)
    ax_cmp_lin.set_title("PipeDream vs GPD vs SGD")
    fig_cmp_lin.tight_layout()
    fig_cmp_lin.savefig(save_dir / "comparison_linear.png", dpi=200, bbox_inches="tight")

    fig_cmp_log, ax_cmp_log = plt.subplots(figsize=(8, 4))
    plot_block_update_comparison(comparison, log_scale=True, ax=ax_cmp_log)
    ax_cmp_log.set_title("PipeDream vs GPD vs SGD (log scale)")
    fig_cmp_log.tight_layout()
    fig_cmp_log.savefig(save_dir / "comparison_log.png", dpi=200, bbox_inches="tight")

    print(f"\nSaved figures to: {save_dir.resolve()}")
    print("  - schedule.png")
    print("  - comparison_linear.png")
    print("  - comparison_log.png")

    if not args.no_show:
        plt.show()
        
    '''


if __name__ == "__main__":
    main()