import time
from typing import List

import numpy as np

from dispatch import kernels
from dispatch.geo import DistanceCache
from dispatch.simulator import DELIVERY, PICKUP, BatchSnapshot, PlanResult, Stop


class GreedyInsertionPolicy:
    """
    Baseline: keep every existing assignment and itinerary, then take unassigned orders oldest
    first and insert each into the courier/slots with the cheapest feasible detour.
    """
    name = "greedy"

    def __init__(self):
        self.cache = DistanceCache()

    def warm_up(self):
        kernels.warm_up()

    def plan(self, snap: BatchSnapshot) -> PlanResult:
        t0 = time.perf_counter()
        self.cache.add_orders(snap.orders.values())
        node_dist = self.cache.node_dist
        courier_dist = self.cache.courier_rows(snap.courier_lat, snap.courier_lng)

        plans: List[List[Stop]] = [list(p) for p in snap.plans]
        held = [len(snap.onboard[d]) + sum(1 for _, kind in p if kind == PICKUP) for d, p in enumerate(plans)]
        unassigned = 0
        for oid in sorted(snap.pool, key=lambda o: (snap.orders[o].release_batch, o)):
            order = snap.orders[oid]
            best = (np.inf, -1, -1, -1)
            for d, plan in enumerate(plans):
                if held[d] >= snap.max_orders_held:
                    continue
                plan_node = np.array([2 * o + kind for o, kind in plan], dtype=np.int64)
                plan_dw = np.array([snap.orders[o].weight * (1 if kind == PICKUP else -1) for o, kind in plan])
                delta, i, j = kernels.best_pair_insertion(
                    d, plan_node, plan_dw, float(snap.courier_load[d]), float(snap.courier_capacity[d]),
                    2 * oid, 2 * oid + 1, order.weight, node_dist, courier_dist,
                )
                if delta < best[0]:
                    best = (delta, d, i, j)
            delta, d, i, j = best
            if d < 0:
                unassigned += 1
                continue
            plan = plans[d]
            plans[d] = plan[:i] + [(oid, PICKUP)] + plan[i:j] + [(oid, DELIVERY)] + plan[j:]
            held[d] += 1

        return PlanResult(plans=plans, stats={"unassigned": unassigned, "solve_ms": (time.perf_counter() - t0) * 1000})
