from src.methods.gpd import GPDMethod
from src.objectives.quadratic import QuadraticObjective


def test_gpd_runs():
    objective = QuadraticObjective.synthetic(
        num_examples=200,
        num_parameters=20,
        num_stages=4,
        batch_size=20,
        seed=0,
    )
    method = GPDMethod(
        num_iterations=25,
        learning_rate=0.01,
        delta=3,
        seed=0,
        init_stage_weights=objective.initial_stage_weights(mode="zeros"),
    )
    trace = method.run(objective)
    assert len(trace.objective_trace) == 26
    assert trace.block_update_objective is not None
