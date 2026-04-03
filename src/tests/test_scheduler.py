from src.schedulers.pipedream_1f1b import simulate_pipedream_1f1b


def test_pipedream_schedule_nonempty():
    timeline = simulate_pipedream_1f1b(num_stages=4, num_microbatches=8, noam=4)
    assert len(timeline) > 0
    assert len(timeline[0]) == 4
