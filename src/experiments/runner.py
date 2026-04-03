from __future__ import annotations

from dataclasses import dataclass

from src.methods.base import Method
from src.objectives.base import Objective
from src.state.trace import SimulationTrace


@dataclass
class ExperimentRunner:
    objective: Objective

    def run(self, method: Method) -> SimulationTrace:
        return method.run(self.objective)

    def run_many(self, methods: list[Method]) -> dict[str, SimulationTrace]:
        return {method.name: method.run(self.objective) for method in methods}
