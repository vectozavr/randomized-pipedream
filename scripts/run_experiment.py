from __future__ import annotations

import argparse

from src.experiments.runner import ExperimentRunner
from src.methods.gpd import GPDMethod
from src.methods.pipedream import PipeDreamMethod
from src.methods.sgd import SGDMethod
from src.objectives.quadratic import QuadraticObjective
from src.schedulers.pipedream_1f1b import PipeDream1F1BScheduler
from src.utils.batching import BatchSampler
from src.utils.io import load_yaml


def build_objective(cfg: dict) -> QuadraticObjective:
    obj = cfg["objective"]
    if obj["kind"] != "quadratic":
        raise ValueError("Only quadratic objective is wired into the scripts for now.")
    return QuadraticObjective.synthetic(
        num_examples=obj["num_examples"],
        num_parameters=obj["num_parameters"],
        num_stages=obj["num_stages"],
        batch_size=obj["batch_size"],
        seed=cfg["seed"],
        noise_std=obj.get("noise_std", 0.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    objective = build_objective(cfg)
    runner = ExperimentRunner(objective=objective)

    L = objective.smoothness_constant
    init_mode = cfg["method"].get("init_mode", "zeros")
    init_stage_weights = objective.initial_stage_weights(mode=init_mode, seed=cfg["seed"])

    method_cfg = cfg["method"]
    kind = method_cfg["kind"]

    if kind == "pipedream":
        scheduler = PipeDream1F1BScheduler(noam=method_cfg.get("noam"))
        timeline = scheduler.generate(
            num_stages=objective.num_stages,
            num_microbatches=method_cfg["num_microbatches"],
        )
        sampler = BatchSampler(
            num_batches=len(objective.get_batches()),
            mode=method_cfg.get("batch_sampler_mode", "sequential"),
            seed=cfg["seed"],
        )
        selected_batch_indices = sampler.sample_many(method_cfg["num_microbatches"])
        method = PipeDreamMethod(
            timeline=timeline,
            learning_rate=method_cfg["learning_rate_scale"] / L,
            selected_batch_indices=selected_batch_indices,
            init_stage_weights=init_stage_weights,
        )
    elif kind == "gpd":
        method = GPDMethod(
            num_iterations=method_cfg["num_iterations"],
            learning_rate=method_cfg["learning_rate_scale"] / L,
            delta=method_cfg["delta"],
            seed=cfg["seed"],
            stage_sampling=method_cfg.get("stage_sampling", "uniform"),
            batch_sampling=method_cfg.get("batch_sampling", "uniform"),
            stale_sampling=method_cfg.get("stale_sampling", "uniform"),
            init_stage_weights=init_stage_weights,
        )
    elif kind == "sgd":
        method = SGDMethod(
            num_iterations=method_cfg["num_iterations"],
            learning_rate=method_cfg["learning_rate_scale"] / L,
            seed=cfg["seed"],
            batch_sampling=method_cfg.get("batch_sampling", "uniform"),
            init_stage_weights=init_stage_weights,
        )
    else:
        raise ValueError(f"Unknown method kind: {kind}")

    trace = runner.run(method)
    print(f"method={trace.method_name}")
    print(f"final_objective={trace.final_objective:.8f}")


if __name__ == "__main__":
    main()
