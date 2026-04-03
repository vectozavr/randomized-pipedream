from __future__ import annotations

from dataclasses import dataclass

from src.objectives.quadratic import QuadraticObjective


@dataclass
class LinearRegressionObjective(QuadraticObjective):
    """
    Thin alias around QuadraticObjective.
    """
    pass
