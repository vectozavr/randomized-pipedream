from __future__ import annotations

from abc import ABC, abstractmethod

from src.state.timeline import Timeline


class Scheduler(ABC):
    @abstractmethod
    def generate(self, num_stages: int, num_microbatches: int) -> Timeline:
        raise NotImplementedError
