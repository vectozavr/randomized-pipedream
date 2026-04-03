from src.objectives.quadratic import QuadraticObjective


def test_quadratic_gradient_shape():
    objective = QuadraticObjective.synthetic(
        num_examples=200,
        num_parameters=21,
        num_stages=4,
        batch_size=20,
        seed=0,
    )
    weights = objective.initial_stage_weights(mode="zeros")
    grads = objective.full_gradient(weights)
    assert len(grads) == objective.num_stages
    assert sum(g.size for g in grads) == objective.num_parameters
