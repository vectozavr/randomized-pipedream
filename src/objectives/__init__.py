from src.objectives.base import Objective
from src.objectives.quadratic import QuadraticObjective

try:
    from src.objectives.logistic import LogisticRegressionObjective
except ModuleNotFoundError as exc:
    if exc.name != "scipy":
        raise
    LogisticRegressionObjective = None
