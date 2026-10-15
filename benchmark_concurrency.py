import sys
import io
import time
import argparse
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

# Ensure UTF-8 console output encoding on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from model.node import Node
from model.driver import Driver
from model.task import DeliveryTask, TaskStatus
from services.batch_dispatcher import BatchDispatcher
from services.adaptive_partitioner import AdaptiveH3Partitioner
from services.osrm_client import OSRMClient


def generate_fleet(num_drivers: int = 50, seed: int = 42) -> List[Driver]:
    """Generate num_drivers distributed across Klang Valley commercial districts, Malaysia."""
    random.seed(seed)
    drivers = []
    # Klang Valley bounds: Lng [101.55, 101.75], Lat [3.05, 3.25]
    for i in range(num_drivers):
        lng = 101.55 + random.random() * 0.20
        lat = 3.05 + random.random() * 0.20
        depot = Node(name=f"Depot-D{i+1}", Lng=lng, Lat=lat)
        driver = Driver(
            name=f"Courier-{i+1:02d}",
            current_location=depot,
            capacity=80.0  # Sufficient payload capacity for multi-order stacking
        )
        drivers.append(driver)
    return drivers


def generate_orders(num_orders: int = 500, seed: int = 42) -> List[DeliveryTask]:
    """Generate num_orders distributed across Klang Valley with pickup-delivery pairs."""
    random.seed(seed + 1000)
    orders = []
    for i in range(num_orders):
        p_lng = 101.55 + random.random() * 0.20
        p_lat = 3.05 + random.random() * 0.20
        d_lng = p_lng + (random.random() - 0.5) * 0.04
        d_lat = p_lat + (random.random() - 0.5) * 0.04

        pickup = Node(name=f"P-{i+1}", Lng=p_lng, Lat=p_lat)
        delivery = Node(name=f"D-{i+1}", Lng=d_lng, Lat=d_lat)
        weight = round(1.5 + random.random() * 3.0, 1)  # 1.5kg - 4.5kg

        task = DeliveryTask(
            task_name=f"Order-Stress-{i+1:03d}",
            pickup=pickup,
            delivery=delivery,
            weight=weight
        )
        orders.append(task)
    return orders


def run_benchmark_adaptive_osrm(
    worker_count: int,
    num_drivers: int = 50,
    num_orders: int = 500,
    enable_fallback: bool = True,
    routing_backend: str = "thread",
    use_osrm: bool = True
) -> Dict[str, Any]:
    """
    Execute spatial benchmark combining:
    1. Density-Aware Adaptive H3 Partitioning (hierarchical tree clustering).
    2. OSRM network road matrix client integration (with automated Haversine fallback).
    3. Parallel execution with worker_count workers: zone claiming in threads, then one ACO solve
       per courier in a thread pool or a GIL-free process pool (routing_backend).
    """
    drivers = generate_fleet(num_drivers=num_drivers, seed=42)
    orders = generate_orders(num_orders=num_orders, seed=42)
    osrm_client = OSRMClient(timeout=1.0) if use_osrm else None

    partitioner = AdaptiveH3Partitioner(
        base_resolution=5,
        max_resolution=7,
        max_orders_per_zone=45,
        enable_nearby_driver_fallback=enable_fallback
    )

    fleet_router = BatchDispatcher(
        drivers=drivers,
        enable_concurrent_routing=worker_count > 1,
        max_workers=worker_count,
        routing_client=osrm_client,
        routing_backend=routing_backend
    )
    # A production dispatcher keeps its worker processes alive across batches, so startup is reported separately
    t_pool = time.perf_counter()
    if worker_count > 1:
        fleet_router.warm_up()
    pool_startup_ms = (time.perf_counter() - t_pool) * 1000

    t_start = time.perf_counter()
    zones = partitioner.build_partition_zones(orders=orders, drivers=drivers)

    claimed_total = 0
    crashed = False
    crash_reason = None

    def _claim_sub_zone(zone):
        # Pool primary couriers; if insufficient, include 1-ring nearby fallback couriers
        couriers = list(zone.primary_drivers)
        if enable_fallback and len(couriers) < 3:
            for nd in zone.nearby_drivers:
                if nd not in couriers:
                    couriers.append(nd)

        if not couriers or not zone.orders:
            return 0

        disp = BatchDispatcher(
            drivers=couriers,
            batch_interval_seconds=30,
            max_h3_steps=6,
            enable_concurrent_claims=False,
            enable_concurrent_routing=False,
            routing_client=osrm_client
        )
        disp.add_orders(zone.orders)
        # Claim only: a courier shared by several zones is routed once after all zones finish claiming
        s = disp.claim_batch(batch_id=1, current_time_sec=0)
        return s.claimed_count

    aco_solves = 0
    claim_ms = 0.0
    try:
        if worker_count > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                zone_results = list(executor.map(_claim_sub_zone, zones))
                claimed_total = sum(zone_results)
        else:
            for z in zones:
                claimed_total += _claim_sub_zone(z)
        claim_ms = (time.perf_counter() - t_start) * 1000

        aco_solves = len(fleet_router.route_pending_drivers())
    except Exception as e:
        crashed = True
        crash_reason = str(e)
    finally:
        total_latency_ms = (time.perf_counter() - t_start) * 1000
        fleet_router.shutdown()

    return {
        "workers": worker_count,
        "backend": routing_backend,
        "total_latency_ms": total_latency_ms,
        "claim_ms": claim_ms,
        "routing_ms": total_latency_ms - claim_ms,
        "pool_startup_ms": pool_startup_ms,
        "aco_solves": aco_solves,
        "claimed_count": claimed_total,
        "unassigned_count": num_orders - claimed_total,
        "active_zones_count": len(zones),
        "crashed": crashed,
        "crash_reason": crash_reason,
    }


def main():
    parser = argparse.ArgumentParser(description="Adaptive H3 + ACO concurrency stress benchmark")
    parser.add_argument("--no-osrm", action="store_true",
                        help="Use local Haversine matrices only (isolates CPU scaling from public OSRM network jitter)")
    args = parser.parse_args()
    use_osrm = not args.no_osrm

    print("=" * 100)
    print("🔥 [STRESS BENCHMARK] Adaptive H3 Partitioning + Indexed ACO (Thread vs Process Pool)")
    print("=" * 100)
    print("Test Configuration:")
    print("  • Fleet Size: 50 Couriers across Klang Valley, Malaysia")
    print("  • Order Load: 500 Streaming Delivery Tasks (1,000 stops)")
    print("  • Architecture: Density-Aware Adaptive H3 Partitioning + 1-Ring Nearby Fallback")
    print(f"  • Routing Matrix Service: {'OSRM Table Service Client (w/ Auto-Fallback)' if use_osrm else 'Local Haversine (OSRM disabled)'}")
    print("  • Concurrency Suite: Thread pool [1, 2, 4, 8, 16] vs Process pool [2, 4, 8, 16]")
    print("-" * 100)

    configs = [("thread", w) for w in (1, 2, 4, 8, 16)] + [("process", w) for w in (2, 4, 8, 16)]
    results = []

    for backend, w in configs:
        print(f"⏳ Running {backend:<7} backend with {w:2d} worker(s) ...", end="", flush=True)
        res = run_benchmark_adaptive_osrm(
            worker_count=w, num_drivers=50, num_orders=500, enable_fallback=True,
            routing_backend=backend, use_osrm=use_osrm
        )
        results.append(res)
        status_str = "❌ CRASHED" if res["crashed"] else "✅ SUCCESS"
        print(f" -> {status_str} | Latency: {res['total_latency_ms']:8.2f} ms (claim {res['claim_ms']:.0f} ms, routing {res['routing_ms']:.0f} ms) | Claimed: {res['claimed_count']}/500 | ACO Solves: {res['aco_solves']}")
        if res["crashed"]:
            print(f"    Crash reason: {res['crash_reason']}")

    print("\n" + "=" * 100)
    print("📊 [BENCHMARK RESULTS TABLE]")
    print("=" * 100)
    print(f"{'Configuration':<22} | {'Total Latency':<14} | {'Speedup':<8} | {'Pool Startup*':<13} | {'Claimed':<8} | {'Fulfillment':<11} | {'Status'}")
    print("-" * 100)

    base_latency = results[0]["total_latency_ms"]

    for r in results:
        cfg_str = f"{r['backend']} x{r['workers']}"
        lat_str = f"{r['total_latency_ms']:.2f} ms"
        speedup = base_latency / r['total_latency_ms'] if r['total_latency_ms'] > 0 else 0.0
        startup_str = f"{r['pool_startup_ms']:.0f} ms" if r['backend'] == "process" else "-"
        claim_str = f"{r['claimed_count']}/500"
        fulfill_pct = f"{(r['claimed_count'] / 500.0) * 100:.1f}%"
        status_str = "CRASHED" if r["crashed"] else "PASSED"

        print(f"{cfg_str:<22} | {lat_str:<14} | {speedup:.2f}x{'':<3} | {startup_str:<13} | {claim_str:<8} | {fulfill_pct:<11} | {status_str}")

    print("=" * 100)
    print("* One-off worker process spawn, excluded from latency (a long-running dispatcher reuses its pool).")


if __name__ == "__main__":
    main()
