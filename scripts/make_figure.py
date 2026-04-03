from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from src.experiments.compare_methods import compare_methods
from src.experiments.runner import ExperimentRunner
from src.methods.gpd import GPDMethod
from src.methods.pipedream import PipeDreamMethod
from src.methods.sgd import SGDMethod
from src.objectives.quadratic import QuadraticObjective
from src.plotting.convergence import plot_block_update_comparison
from src.schedulers.pipedream_1f1b import PipeDream1F1BScheduler
from src.utils.batching import BatchSampler
from src.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="comparison yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    obj_cfg = cfg["objective"]
    cmp_cfg = cfg["comparison"]

    objective = QuadraticObjective.synthetic(
        num_examples=obj_cfg["num_examples"],
        num_parameters=obj_cfg["num_parameters"],
        num_stages=obj_cfg["num_stages"],
        batch_size=obj_cfg["batch_size"],
        seed=cfg["seed"],
        noise_std=obj_cfg.get("noise_std", 0.0),
    )
    L = objective.smoothness_constant
    init_stage_weights = objective.initial_stage_weights(
        mode=cmp_cfg.get("init_mode", "zeros"),
        seed=cfg["seed"],
    )

    scheduler = PipeDream1F1BScheduler(noam=cmp_cfg.get("noam"))
    timeline = scheduler.generate(
        num_stages=objective.num_stages,
        num_microbatches=cmp_cfg["num_microbatches"],
    )
    sampler = BatchSampler(
        num_batches=len(objective.get_batches()),
        mode=cmp_cfg.get("batch_sampler_mode", "random"),
        seed=cfg["seed"],
    )
    selected_batch_indices = sampler.sample_many(cmp_cfg["num_microbatches"])

    pipedream = PipeDreamMethod(
        timeline=timeline,
        learning_rate=cmp_cfg["learning_rate_scale_pipedream"] / L,
        selected_batch_indices=selected_batch_indices,
        init_stage_weights=init_stage_weights,
    )
    gpd = GPDMethod(
        num_iterations=objective.num_stages * cmp_cfg["num_microbatches"],
        learning_rate=cmp_cfg["learning_rate_scale_gpd"] / L,
        delta=cmp_cfg["gpd_delta"],
        seed=cfg["seed"],
        init_stage_weights=init_stage_weights,
    )
    sgd = SGDMethod(
        num_iterations=cmp_cfg["num_microbatches"],
        learning_rate=cmp_cfg["learning_rate_scale_sgd"] / L,
        seed=cfg["seed"],
        init_stage_weights=init_stage_weights,
    )

    comparison = compare_methods(ExperimentRunner(objective), [pipedream, gpd, sgd])
    plot_block_update_comparison(comparison, log_scale=False)
    plt.show()


if __name__ == "__main__":
    main()
