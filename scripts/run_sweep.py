from __future__ import annotations

import argparse

from src.experiments.runner import ExperimentRunner
from src.experiments.sweep import parameter_sweep
from src.methods.gpd import GPDMethod
from src.objectives.quadratic import QuadraticObjective
from src.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="YAML config with objective and sweep grid")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    objective_cfg = cfg["objective"]
    objective = QuadraticObjective.synthetic(
        num_examples=objective_cfg["num_examples"],
        num_parameters=objective_cfg["num_parameters"],
        num_stages=objective_cfg["num_stages"],
        batch_size=objective_cfg["batch_size"],
        seed=cfg["seed"],
        noise_std=objective_cfg.get("noise_std", 0.0),
    )
    runner = ExperimentRunner(objective=objective)
    L = objective.smoothness_constant
    init_stage_weights = objective.initial_stage_weights(mode="zeros", seed=cfg["seed"])

    sweep_cfg = cfg["sweep"]
    grid = sweep_cfg["grid"]

    def factory(**kwargs):
        return GPDMethod(
            learning_rate=sweep_cfg.get("learning_rate_scale", 1.0) / L,
            seed=cfg["seed"],
            init_stage_weights=init_stage_weights,
            **kwargs,
        )

    results = parameter_sweep(runner, factory, grid)
    for params, trace in results:
        print(params, trace.final_objective)


if __name__ == "__main__":
    main()
