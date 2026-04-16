from __future__ import annotations
from dataclasses import dataclass
from src.schedulers.base import Scheduler
from src.state.timeline import Timeline


def simulate_local_sgd_1f1b(num_stages: int, num_microbatches: int, num_runs: int, local_steps: int) -> Timeline:
    M = num_runs
    K = local_steps
    round_size = M * K

    f_done = [[None] * num_microbatches for _ in range(num_stages)]
    b_done = [[None] * num_microbatches for _ in range(num_stages)]

    launched = 0
    steady_state = [False] * num_stages
    next_preference = ["F"] * num_stages
    timeline: Timeline = []

    while b_done[0][num_microbatches - 1] is None:
        ops_this_step = [None] * num_stages

        for stage in range(num_stages):
            ready_f: int | None = None
            ready_b: int | None = None

            # 1. Find earliest ready FORWARD microbatch
            for mb in range(num_microbatches):
                if f_done[stage][mb] is not None:
                    continue

                round_id = mb // round_size
                mb_in_round = mb % round_size

                # --- DEPENDENCY A: Intra-round Local SGD ---
                # Wait for the previous local step of THIS run to update local weights
                if mb_in_round >= M:
                    if b_done[stage][mb - M] is None or b_done[stage][mb - M] >= len(timeline):
                        break

                # --- DEPENDENCY B: Inter-round GLOBAL SYNC ---
                # If we are starting a new round, we must wait for the ENTIRE
                # previous round to finish its backward passes on Stage 0.
                if round_id > 0:
                    prev_round_start = (round_id - 1) * round_size
                    prev_round_end = round_id * round_size

                    prev_round_synced = True
                    for prev_mb in range(prev_round_start, prev_round_end):
                        # Stage 0 completing B means the pipeline is fully drained for that mb
                        if b_done[0][prev_mb] is None or b_done[0][prev_mb] >= len(timeline):
                            prev_round_synced = False
                            break

                    if not prev_round_synced:
                        break  # Halt forward search! The pipeline must drain and sync here.

                if stage == 0:
                    if mb == launched and launched < num_microbatches:
                        ready_f = mb
                    break

                if f_done[stage - 1][mb] is not None and f_done[stage - 1][mb] < len(timeline):
                    ready_f = mb
                break

            # 2. Find earliest ready BACKWARD microbatch
            for mb in range(num_microbatches):
                if f_done[stage][mb] is None or f_done[stage][mb] >= len(timeline) or b_done[stage][mb] is not None:
                    continue
                if stage == num_stages - 1:
                    ready_b = mb
                    break
                if b_done[stage + 1][mb] is not None and b_done[stage + 1][mb] < len(timeline):
                    ready_b = mb
                    break

            # 3. 1F1B Selection Logic
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
            if op is not None:
                kind, mb = op
                if kind == "F":
                    f_done[stage][mb] = len(timeline)
                    if stage == 0: launched += 1
                else:
                    b_done[stage][mb] = len(timeline)

        timeline.append(ops_this_step)

    return timeline


@dataclass
class IndependentLocalSGDScheduler(Scheduler):
    num_runs: int = 1  # M
    local_steps: int = 1  # K

    def generate(self, num_stages: int, num_microbatches: int) -> Timeline:
        return simulate_local_sgd_1f1b(
            num_stages=num_stages,
            num_microbatches=num_microbatches,
            num_runs=self.num_runs,
            local_steps=self.local_steps
        )