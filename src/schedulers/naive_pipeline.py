from __future__ import annotations

from dataclasses import dataclass

from src.schedulers.base import Scheduler
from src.state.timeline import Timeline


def simulate_naive_fill_drain(num_stages: int, num_microbatches: int) -> Timeline:
    num_steps = 2 * (num_microbatches + num_stages - 1)
    timeline: Timeline = [[None] * num_stages for _ in range(num_steps)]

    for mb in range(num_microbatches):
        for stage in range(num_stages):
            t = mb + stage
            timeline[t][stage] = ("F", mb)

    offset = num_microbatches + num_stages - 1
    for mb in range(num_microbatches):
        for stage in reversed(range(num_stages)):
            t = offset + mb + (num_stages - 1 - stage)
            timeline[t][stage] = ("B", mb)

    return timeline


@dataclass
class NaivePipelineScheduler(Scheduler):
    def generate(self, num_stages: int, num_microbatches: int) -> Timeline:
        return simulate_naive_fill_drain(num_stages=num_stages, num_microbatches=num_microbatches)
