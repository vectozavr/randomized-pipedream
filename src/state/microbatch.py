from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MicrobatchRuntime:
    batch_id: int
    activations: list[np.ndarray | None]
    grad_to_left: np.ndarray | None
    stashed_weights: list[np.ndarray | None]
    stashed_versions: list[int | None]
    loss_on_forward: float | None = None
