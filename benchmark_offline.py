import argparse
import io
import sys

import numpy as np

# Ensure UTF-8 console output encoding on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dispatch import offline
from dispatch.exact_policy import ExactBatchPolicy
from dispatch.greedy_policy import GreedyInsertionPolicy
from dispatch.scenario import generate_kl_scenario
from dispatch.simulator import RollingSimulator


def row(label, km, makespan_s, latency_s, ref_km):
    gap = (km / ref_km - 1) * 100
    print(f"  {label:<34} {km:>9.1f} {gap:>+8.1f}% {makespan_s / 60:>9.1f} {latency_s.mean() / 60:>8.1f} "
          f"{np.percentile(latency_s, 95) / 60:>8.1f}")


def main():
    parser = argparse.ArgumentParser(description="Online rolling dispatch vs offline full-information reference")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2026])
    parser.add_argument("--time-limit", type=float, default=120.0, help="OR-Tools seconds per offline solve")
    parser.add_argument("--deadline-slack", type=float, default=60.0,
                        help="Seconds added to the online makespan for the offline deadline (absorbs integer rounding)")
    args = parser.parse_args()

    for seed in args.seeds:
        sc = generate_kl_scenario(seed=seed)
        exact = ExactBatchPolicy()
        exact.warm_up()
        online = RollingSimulator(sc, exact).run()
        greedy = RollingSimulator(sc, GreedyInsertionPolicy()).run()
        replay = offline.schedule(sc, online.courier_visits)
        lower = offline.assignment_lower_bound(sc)
        free = offline.solve_offline(sc, online.courier_visits, None, args.time_limit)
        deadline = offline.solve_offline(sc, online.courier_visits, online.makespan_s + args.deadline_slack,
                                         args.time_limit)

        ref = deadline.schedule.total_km
        print("=" * 100)
        print(f"seed {seed}: {len(sc.couriers)} couriers, {len(sc.orders)} orders | "
              f"OR-Tools {args.time_limit:.0f}s per offline solve | gap column is vs offline (same deadline)")
        print(f"  {'':<34} {'km':>9} {'gap':>9} {'makespan':>9} {'lat avg':>8} {'lat p95':>8}")
        row("online greedy (driven)", greedy.total_km, greedy.makespan_s, greedy.delivery_latency_s, ref)
        row("online exact (driven)", online.total_km, online.makespan_s, online.delivery_latency_s, ref)
        row("online exact (visit order, no detours)", replay.total_km, replay.makespan_s, replay.delivery_latency_s, ref)
        row("offline, same deadline", ref, deadline.schedule.makespan_s, deadline.schedule.delivery_latency_s, ref)
        row("offline, km only (no deadline)", free.schedule.total_km, free.schedule.makespan_s,
            free.schedule.delivery_latency_s, ref)
        print(f"  {'valid lower bound (assignment)':<34} {lower:>9.1f} {(lower / ref - 1) * 100:>+8.1f}%")
        print(f"  detours from re-planning mid-leg: {online.total_km - replay.total_km:.1f} km")


if __name__ == "__main__":
    main()
