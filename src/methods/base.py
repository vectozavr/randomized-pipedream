from __future__ import annotations

from abc import ABC, abstractmethod

from src.objectives.base import Objective
from src.state.trace import SimulationTrace


class Method(ABC):
    name: str

    @abstractmethod
    def run(self, objective: Objective) -> SimulationTrace:
        raise NotImplementedError
