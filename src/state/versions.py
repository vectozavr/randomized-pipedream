from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VersionTracker:
    num_stages: int
    versions: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.versions = np.zeros(self.num_stages, dtype=int)

    def increment(self, stage: int) -> None:
        self.versions[stage] += 1

    def snapshot(self) -> np.ndarray:
        return self.versions.copy()
