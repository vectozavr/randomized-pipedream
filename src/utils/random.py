from __future__ import annotations

import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)
