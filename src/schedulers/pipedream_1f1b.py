from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

from src.schedulers.base import Scheduler
from src.state.timeline import Timeline


def simulate_pipedream_1f1b(num_stages: int, num_microbatches: int, noam: int | None = None) -> Timeline:
    if noam is None:
        noam = num_stages

    f_done = [[None] * num_microbatches for _ in range(num_stages)]
    b_done = [[None] * num_microbatches for _ in range(num_stages)]

    launched = 0
    steady_state = [False] * num_stages
    next_preference = ["F"] * num_stages
    timeline: Timeline = []

    def active_microbatches() -> int:
        return sum(
            1
            for mb in range(num_microbatches)
            if f_done[0][mb] is not None and b_done[0][mb] is None
        )

    while b_done[0][num_microbatches - 1] is None:
        ops_this_step = [None] * num_stages

        for stage in range(num_stages):
            ready_f: int | None = None
            ready_b: int | None = None

            for mb in range(num_microbatches):
                if f_done[stage][mb] is not None:
                    continue

                if stage == 0:
                    if mb == launched and launched < num_microbatches and active_microbatches() < noam:
                        ready_f = mb
                    break

                if f_done[stage - 1][mb] is not None and f_done[stage - 1][mb] < len(timeline):
                    ready_f = mb
                break

            for mb in range(num_microbatches):
                if f_done[stage][mb] is None:
                    continue
                if f_done[stage][mb] >= len(timeline):
                    continue
                if b_done[stage][mb] is not None:
                    continue

                if stage == num_stages - 1:
                    ready_b = mb
                    break

                if b_done[stage + 1][mb] is not None and b_done[stage + 1][mb] < len(timeline):
                    ready_b = mb
                    break

            chosen = None

            if not steady_state[stage]:
                if ready_b is not None:
                    chosen = ("B", ready_b)
                    steady_state[stage] = True
                    next_preference[stage] = "F"
                elif ready_f is not None:
                    chosen = ("F", ready_f)
            else:
                pref = next_preference[stage]
                if pref == "B" and ready_b is not None:
                    chosen = ("B", ready_b)
                    next_preference[stage] = "F"
                elif pref == "F" and ready_f is not None:
                    chosen = ("F", ready_f)
                    next_preference[stage] = "B"
                elif ready_b is not None:
                    chosen = ("B", ready_b)
                    next_preference[stage] = "F"
                elif ready_f is not None:
                    chosen = ("F", ready_f)
                    next_preference[stage] = "B"

            ops_this_step[stage] = chosen

        for stage, op in enumerate(ops_this_step):
            if op is None:
                continue
            kind, mb = op
            if kind == "F":
                f_done[stage][mb] = len(timeline)
                if stage == 0:
                    launched += 1
            else:
                b_done[stage][mb] = len(timeline)

        timeline.append(ops_this_step)

    return timeline


def print_schedule(timeline: Timeline) -> None:
    num_stages = len(timeline[0])
    for stage in range(num_stages):
        row = []
        for ops in timeline:
            op = ops[stage]
            if op is None:
                row.append(" . ")
            else:
                kind, mb = op
                row.append(f"{kind}{mb + 1}".rjust(3))
        print(f"stage {stage}: " + " ".join(row))


def plot_schedule(
    timeline: Timeline,
    startup_boundary: int | None = None,
    figsize: tuple[float, float] = (14.0, 4.0),
):
    num_steps = len(timeline)
    num_stages = len(timeline[0])

    fig, ax = plt.subplots(figsize=figsize)

    colors = {'F': '#4C78A8', 'B': '#8BC17C'}

    for t in range(num_steps):
        for stage in range(num_stages):
            y = num_stages - 1 - stage
            op = timeline[t][stage]

            if op is None:
                rect = Rectangle(
                    (t, y), 1, 1,
                    facecolor='white',
                    edgecolor='black',
                    hatch='////',
                    linewidth=0.8
                )
                ax.add_patch(rect)
                continue

            kind, mb = op
            rect = Rectangle(
                (t, y), 1, 1,
                facecolor=colors[kind],
                edgecolor='black',
                linewidth=0.8
            )
            ax.add_patch(rect)

            ax.text(
                t + 0.5,
                y + 0.5,
                str(mb + 1),
                ha='center',
                va='center',
                fontsize=15,
                color='white',
                fontweight='bold'
            )

    if startup_boundary is not None:
        ax.axvline(startup_boundary, color='black', linestyle='--', linewidth=1.2)
        ax.text(startup_boundary / 2, -0.45, 'Startup', ha='center', va='center', fontsize=11)
        ax.text((startup_boundary + num_steps) / 2, -0.45, 'Steady / transition', ha='center', va='center', fontsize=11)

    ax.set_xlim(0, num_steps)
    ax.set_ylim(0, num_stages)
    ax.set_xticks(np.arange(num_steps) + 0.5)
    ax.set_xticklabels(np.arange(num_steps))
    ax.set_yticks(np.arange(num_stages) + 0.5)
    ax.set_yticklabels([f'Machine {i + 1}' for i in range(num_stages)][::-1])
    ax.set_xlabel('Time step')
    ax.set_ylabel('')
    ax.set_title('PipeDream-style 1F1B schedule')
    ax.grid(False)

    forward_patch = Rectangle((0, 0), 1, 1, facecolor=colors['F'], edgecolor='black')
    backward_patch = Rectangle((0, 0), 1, 1, facecolor=colors['B'], edgecolor='black')
    idle_patch = Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', hatch='////')

    legend = ax.legend(
        [forward_patch, backward_patch, idle_patch],
        ['Forward', 'Backward', 'Idle'],
        loc='upper left',
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        facecolor='white',
        edgecolor='black',
        framealpha=0.85
    )
    legend.set_zorder(1000)

    plt.tight_layout()
    return fig, ax


@dataclass
class PipeDream1F1BScheduler(Scheduler):
    noam: Optional[int] = None

    def generate(self, num_stages: int, num_microbatches: int) -> Timeline:
        return simulate_pipedream_1f1b(
            num_stages=num_stages,
            num_microbatches=num_microbatches,
            noam=self.noam,
        )
