from __future__ import annotations

from src.state.timeline import Timeline


def validate_timeline(timeline: Timeline) -> None:
    if not timeline:
        raise ValueError("Timeline is empty.")
    num_stages = len(timeline[0])
    for t, ops in enumerate(timeline):
        if len(ops) != num_stages:
            raise ValueError(f"Inconsistent number of stages at time {t}.")
        for op in ops:
            if op is None:
                continue
            kind, mb = op
            if kind not in {"F", "B"}:
                raise ValueError(f"Unknown operation type: {kind}")
            if mb < 0:
                raise ValueError("Microbatch id must be non-negative.")
