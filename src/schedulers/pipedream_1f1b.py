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

    # We only need to store the completion time of each microbatch to know if it's ready.
    # -1 indicates that the microbatch has not completed yet.
    f_done = [[-1] * num_microbatches for _ in range(num_stages)]
    b_done = [[-1] * num_microbatches for _ in range(num_stages)]

    steady_state = [False] * num_stages
    next_preference = ["F"] * num_stages
    timeline: Timeline = []

    # O(1) pointers to the next candidate microbatch for each stage
    next_f = [0] * num_stages
    next_b = [0] * num_stages

    active_mbs = 0
    current_time = 0

    # cycle until the last microbatch finishes backward on the first stage
    while b_done[0][num_microbatches - 1] == -1:
        ops_this_step = [None] * num_stages

        # iterate over stages to find ready forward and backward microbatches
        for stage in range(num_stages):
            ready_f: int | None = None
            ready_b: int | None = None

            # 1. Check if the earliest unfinished FORWARD microbatch is ready
            mb_f = next_f[stage]
            if mb_f < num_microbatches:
                if stage == 0:
                    if active_mbs < noam:
                        ready_f = mb_f
                else:
                    if f_done[stage - 1][mb_f] != -1 and f_done[stage - 1][mb_f] < current_time:
                        ready_f = mb_f

            # 2. Check if the earliest unfinished BACKWARD microbatch is ready
            mb_b = next_b[stage]
            if mb_b < num_microbatches:
                # backward can't start until forward is done on THIS stage in a previous step
                if f_done[stage][mb_b] != -1 and f_done[stage][mb_b] < current_time:
                    if stage == num_stages - 1:
                        ready_b = mb_b
                    else:
                        if b_done[stage + 1][mb_b] != -1 and b_done[stage + 1][mb_b] < current_time:
                            ready_b = mb_b

            chosen = None

            # 3. 1F1B picking logic (Simplified cleanly but functionally identical to original)
            if ready_b is not None and (not steady_state[stage] or next_preference[stage] == "B" or ready_f is None):
                chosen = ("B", ready_b)
                steady_state[stage] = True
                next_preference[stage] = "F"
            elif ready_f is not None:
                chosen = ("F", ready_f)
                if steady_state[stage]:
                    next_preference[stage] = "B"

            ops_this_step[stage] = chosen

        # 4. Update states efficiently at the end of the step
        for stage, op in enumerate(ops_this_step):
            if op is None:
                continue

            kind, mb = op
            if kind == "F":
                f_done[stage][mb] = current_time
                next_f[stage] += 1
                if stage == 0:
                    active_mbs += 1
            else:
                b_done[stage][mb] = current_time
                next_b[stage] += 1
                if stage == 0:
                    active_mbs -= 1

        timeline.append(ops_this_step)
        current_time += 1

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
        *,
        reduce_text: bool = False,
        max_xtick_labels: int = 30
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

            if not reduce_text:
                ax.text(
                    t + 0.5,
                    y + 0.5,
                    str(mb + 1),
                    ha='center',
                    va='center',
                    fontsize=21,
                    color='white',
                    fontweight='bold'
                )

    if startup_boundary is not None:
        ax.axvline(startup_boundary, color='black', linestyle='--', linewidth=1.2)

        phase_y = -0.75

        ax.text(
            startup_boundary / 2,
            phase_y,
            'Startup',
            ha='center',
            va='center',
            fontsize=11
        )
        ax.text(
            (startup_boundary + num_steps) / 2,
            phase_y,
            'Steady / transition',
            ha='center',
            va='center',
            fontsize=11
        )

    ax.set_xlim(0, num_steps)
    ax.set_ylim(0, num_stages)

    if num_steps <= max_xtick_labels:
        tick_positions = np.arange(num_steps) + 0.5
        tick_labels = np.arange(num_steps)
    else:
        tick_indices = np.linspace(0, num_steps - 1, max_xtick_labels, dtype=int)
        tick_indices = np.unique(tick_indices)
        tick_positions = tick_indices + 0.5
        tick_labels = tick_indices

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

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
    plt.subplots_adjust(bottom=0.22)

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