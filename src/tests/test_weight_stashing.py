import numpy as np

from src.methods.pipedream import PipeDreamMethod
from src.objectives.quadratic import QuadraticObjective
from src.schedulers.pipedream_1f1b import simulate_pipedream_1f1b


def test_weight_stashing_versions_match():
    objective = QuadraticObjective.synthetic(
        num_examples=200,
        num_parameters=20,
        num_stages=4,
        batch_size=20,
        seed=0,
    )
    timeline = simulate_pipedream_1f1b(4, 8, 4)
    selected_batch_indices = list(range(8))
    method = PipeDreamMethod(
        timeline=timeline,
        learning_rate=0.01,
        selected_batch_indices=selected_batch_indices,
        init_stage_weights=objective.initial_stage_weights(mode="zeros"),
    )
    trace = method.run(objective)
    assert np.all(trace.forward_versions == trace.backward_versions)
