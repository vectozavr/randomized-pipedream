from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.batching import build_training_batch_schedule
from src.objectives.quadratic import QuadraticObjective
from src.schedulers.pipedream_1f1b import PipeDream1F1BScheduler, print_schedule, plot_schedule
from src.methods.pipedream import PipeDreamMethod
from src.methods.gpd import GPDMethod
from src.methods.sgd import SGDMethod
from src.experiments.runner import ExperimentRunner
from src.experiments.compare_methods import compare_methods
from src.plotting.convergence import plot_block_update_comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PipeDream / GPD / SGD debug experiment.")
    parser.add_argument("--num-parameters", type=int, default=64)
    parser.add_argument("--num-stages", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-microbatches", type=int, default=100)
    parser.add_argument("--num-epochs", type=int, default=10)
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


def main() -> None:
    args = build_parser().parse_args()

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

    pd_lr = args.pd_lr if args.pd_lr is not None else 0.2 / L
    gpd_lr = args.gpd_lr if args.gpd_lr is not None else 0.2 / L
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
    fig_sched, ax_sched = plot_schedule(timeline, startup_boundary=args.num_stages)
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

    gpd_method = GPDMethod(
        num_iterations=K_budget,
        learning_rate=gpd_lr,
        delta=gpd_delta,
        seed=args.gpd_seed,
        stage_sampling="pipedream",
        batch_sampling="pipedream",
        stale_sampling="pipedream",
        timeline=timeline,
        training_batch_indices=training_batch_indices,
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


if __name__ == "__main__":
    main()