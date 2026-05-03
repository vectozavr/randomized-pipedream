python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
PY# PipeDream / GPD research project

This project packages the exploratory notebook code into a small research codebase.

## What is included

- `src/objectives/` — objective definitions
- `src/schedulers/` — schedule generation
- `src/methods/` — PipeDream, GPD, and SGD simulations
- `src/plotting/` — plotting helpers
- `src/experiments/` — small orchestration helpers
- `notebooks/experiments.ipynb` — thin notebook driver

The current implementation centers on a block-partitioned quadratic objective and compares:

- PipeDream-style 1F1B with weight stashing
- Generalized PipeDream (GPD)
- standard minibatch SGD

## Quick start

From the project root:

```bash
python -m scripts.run_experiment configs/quadratic_pipedream.yaml
python -m scripts.run_experiment configs/quadratic_gpd.yaml
python -m scripts.run_experiment configs/quadratic_sgd.yaml
python -m scripts.make_figure configs/comparison.yaml
```

Or open `notebooks/experiments.ipynb`.

## Notes

- The GPD abstraction follows the stale block-gradient formulation in the draft text:
  a random stage-batch pair is sampled, a stale mixed model is read, and only the active block is updated.
- The PipeDream implementation replays a static 1F1B timeline and enforces weight stashing by verifying that each microbatch-stage pair uses the same local version on forward and backward.

## Minimal design choice

This repository keeps the code modular enough for research, but avoids heavy configuration systems or large frameworks.
