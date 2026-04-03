from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

Operation = tuple[str, int]
TimeStep = list[Optional[Operation]]
Timeline = list[TimeStep]


@dataclass(frozen=True)
class OpRecord:
    time: int
    stage: int
    kind: str
    microbatch: int


def num_microbatches_from_timeline(timeline: Timeline) -> int:
    mx = -1
    for ops in timeline:
        for op in ops:
            if op is not None:
                _, mb = op
                mx = max(mx, mb)
    return mx + 1


def iter_operations(timeline: Timeline) -> list[OpRecord]:
    records: list[OpRecord] = []
    for t, ops in enumerate(timeline):
        for stage, op in enumerate(ops):
            if op is None:
                continue
            kind, mb = op
            records.append(OpRecord(time=t, stage=stage, kind=kind, microbatch=mb))
    return records
