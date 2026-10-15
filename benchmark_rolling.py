import argparse
import io
import sys
import time

import numpy as np

# Ensure UTF-8 console output encoding on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dispatch.exact_policy import ExactBatchPolicy, ExactConfig
from dispatch.greedy_policy import GreedyInsertionPolicy
from dispatch.scenario import generate_kl_scenario
from dispatch.simulator import RollingSimulator, SimulationResult

TIMING_KEYS = ("matrix_ms", "candidates_ms", "columns_ms", "dp_ms", "mip_ms", "decode_ms")


def build_policy(name: str, args):
    if name == "greedy":
        return GreedyInsertionPolicy()
    return ExactBatchPolicy(ExactConfig(
        max_add_per_batch=args.max_add,
        max_release_per_batch=args.max_release,
        max_add_when_releasing=args.max_add_releasing,
        candidates_per_courier=args.candidates,
        couriers_per_order=args.couriers_per_order,
        bb_node_limit=args.bb_nodes,
        rc_threshold_km=args.rc_window,
    ))


def print_batches(result: SimulationResult, show_all: bool):
    print(f"\n[{result.policy}] per-batch solve log")
    header = f"{'batch':>5} {'t(s)':>6} {'new':>4} {'open':>5} {'unasg':>5} {'cols':>7} {'solve ms':>9} {'gap':>6} {'reasg':>5}"
    if result.policy == "exact":
        header += "  " + " ".join(f"{k[:-3]:>8}" for k in TIMING_KEYS)
    print(header)
    for b in result.batches:
        if not show_all and b.released == 0 and b.batch % 10:
            continue
        cols = b.stats.get("columns", "-")
        gap = b.stats.get("mip_gap")
        line = (f"{b.batch:>5} {b.time_s:>6.0f} {b.released:>4} {b.open_orders:>5} {b.unassigned_after:>5} "
                f"{cols:>7} {b.solve_ms:>9.1f} {('%.4f' % gap) if gap is not None else '-':>6} {b.reassigned:>5}")
        if result.policy == "exact":
            line += "  " + " ".join(f"{b.stats.get(k, 0):>8.1f}" for k in TIMING_KEYS)
        print(line)


def summarize(result: SimulationResult, release_batches: int) -> dict:
    solve = result.solve_ms
    inject = solve[:release_batches]
    gaps = [b.stats.get("mip_gap", 0.0) for b in result.batches]
    optimal = [b.stats.get("mip_optimal", 1.0) for b in result.batches]
    return {
        "policy": result.policy,
        "total_km": result.total_km,
        "makespan_min": result.makespan_s / 60,
        "latency_avg_min": result.delivery_latency_s.mean() / 60,
        "latency_p95_min": np.percentile(result.delivery_latency_s, 95) / 60,
        "batches": len(result.batches),
        "inject_p50_ms": np.percentile(inject, 50),
        "inject_p95_ms": np.percentile(inject, 95),
        "all_p95_ms": np.percentile(solve, 95),
        "max_ms": solve.max(),
        "max_gap": max(gaps),
        "all_optimal": all(o == 1.0 for o in optimal),
        "reassignments": result.reassignments,
    }


def main():
    parser = argparse.ArgumentParser(description="Rolling 30 s batch dispatch benchmark (Klang Valley)")
    parser.add_argument("--policy", choices=["exact", "greedy", "both"], default="both")
    parser.add_argument("--couriers", type=int, default=50)
    parser.add_argument("--orders", type=int, default=500)
    parser.add_argument("--per-batch", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-add", type=int, default=2)
    parser.add_argument("--max-release", type=int, default=1)
    parser.add_argument("--max-add-releasing", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--couriers-per-order", type=int, default=3)
    parser.add_argument("--bb-nodes", type=int, default=ExactConfig.bb_node_limit)
    parser.add_argument("--rc-window", type=float, default=ExactConfig.rc_threshold_km)
    parser.add_argument("--all-batches", action="store_true", help="Print every batch, not just injection batches")
    args = parser.parse_args()

    scenario = generate_kl_scenario(num_couriers=args.couriers, num_orders=args.orders,
                                    orders_per_batch=args.per_batch, seed=args.seed)
    release_batches = -(-args.orders // args.per_batch)
    print("=" * 100)
    print(f"Rolling dispatch: {args.couriers} couriers, {args.orders} orders, {args.per_batch}/batch every "
          f"{scenario.batch_seconds:.0f}s, {scenario.speed_kmh:.0f} km/h, hold <= {scenario.max_orders_held}, seed {args.seed}")
    print("=" * 100)

    names = ["greedy", "exact"] if args.policy == "both" else [args.policy]
    summaries = []
    for name in names:
        policy = build_policy(name, args)
        t0 = time.perf_counter()
        policy.warm_up()
        print(f"[{name}] JIT warm-up: {(time.perf_counter() - t0) * 1000:.0f} ms (excluded from solve times)")
        result = RollingSimulator(scenario, policy).run()
        print_batches(result, args.all_batches)
        summaries.append(summarize(result, release_batches))

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    keys = list(summaries[0])
    for k in keys[1:]:
        cells = []
        for s in summaries:
            v = s[k]
            cells.append(f"{v:>14.2f}" if isinstance(v, float) else f"{str(v):>14}")
        print(f"{k:<18}" + "".join(cells))
    print(f"{'(policy)':<18}" + "".join(f"{s['policy']:>14}" for s in summaries))


if __name__ == "__main__":
    main()
