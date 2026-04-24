from __future__ import annotations
from dataclasses import dataclass
from src.schedulers.base import Scheduler
from src.state.timeline import Timeline


def simulate_local_sgd_1f1b(num_stages: int, num_microbatches: int, num_runs: int, local_steps: int) -> Timeline:
    M = num_runs
    K = local_steps
    round_size = M * K

    # -1 indicates that the microbatch has not completed yet.
    f_done = [[-1] * num_microbatches for _ in range(num_stages)]
    b_done = [[-1] * num_microbatches for _ in range(num_stages)]

    steady_state = [False] * num_stages
    next_preference = ["F"] * num_stages
    timeline: Timeline = []

    # O(1) pointers to the next candidate microbatch for each stage
    next_f = [0] * num_stages
    next_b = [0] * num_stages
    current_time = 0

    # cycle until the last microbatch finishes backward on the first stage
    while b_done[0][num_microbatches - 1] == -1:
        ops_this_step = [None] * num_stages

        for stage in range(num_stages):
            ready_f: int | None = None
            ready_b: int | None = None

            # 1. Find earliest ready FORWARD microbatch
            mb_f = next_f[stage]
            if mb_f < num_microbatches:
                round_id = mb_f // round_size
                mb_in_round = mb_f % round_size

                can_do_f = True

                # --- DEPENDENCY A: Intra-round Local SGD ---
                # Wait for the previous local step of THIS run to update local weights
                if mb_in_round >= M:
                    prev_dep = mb_f - M
                    if b_done[stage][prev_dep] == -1 or b_done[stage][prev_dep] >= current_time:
                        can_do_f = False

                # --- DEPENDENCY B: Inter-round GLOBAL SYNC ---
                # Because microbatches complete backwards strictly in order, we only
                # need to check if the LAST microbatch of the previous round finished! (O(1) vs O(N))
                if can_do_f and round_id > 0:
                    prev_round_last_mb = round_id * round_size - 1
                    if b_done[0][prev_round_last_mb] == -1 or b_done[0][prev_round_last_mb] >= current_time:
                        can_do_f = False

                if can_do_f:
                    if stage == 0:
                        ready_f = mb_f
                    else:
                        if f_done[stage - 1][mb_f] != -1 and f_done[stage - 1][mb_f] < current_time:
                            ready_f = mb_f

            # 2. Find earliest ready BACKWARD microbatch
            mb_b = next_b[stage]
            if mb_b < num_microbatches:
                # backward can't start until forward is done on THIS stage in a previous step
                if f_done[stage][mb_b] != -1 and f_done[stage][mb_b] < current_time:
                    if stage == num_stages - 1:
                        ready_b = mb_b
                    else:
                        if b_done[stage + 1][mb_b] != -1 and b_done[stage + 1][mb_b] < current_time:
                            ready_b = mb_b

            # 3. 1F1B Selection Logic
            chosen = None
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
            if op is not None:
                kind, mb = op
                if kind == "F":
                    f_done[stage][mb] = current_time
                    next_f[stage] += 1
                else:
                    b_done[stage][mb] = current_time
                    next_b[stage] += 1

        timeline.append(ops_this_step)
        current_time += 1

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
