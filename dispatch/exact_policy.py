import time
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix

from dispatch import kernels
from dispatch.geo import DistanceCache
from dispatch.setpart import solve_set_partition
from dispatch.simulator import DELIVERY, PICKUP, BatchSnapshot, PlanResult, Stop


@dataclass
class ExactConfig:
    """
    Bounds of the column space the batch MIP is exact over. A courier's column is the set of
    orders it will hold after this batch: either its current held orders plus at most
    max_add_per_batch candidates, or its held orders minus one (max_release_per_batch) plus at most
    max_add_when_releasing candidates, never exceeding the scenario's holding limit.
    """
    max_add_per_batch: int = 2
    max_release_per_batch: int = 1
    max_add_when_releasing: int = 1    # A release column may take at most this many new orders (a swap)
    candidates_per_courier: int = 10   # Each courier considers its L closest open orders...
    couriers_per_order: int = 3        # ...and each open order is offered to its K closest couriers
    unassigned_penalty_km: float = 100.0   # Above any single-order detour inside the Klang Valley box
    age_penalty_km_per_min: float = 1.0    # Older orders win when holding limits leave some unassigned
    mip_time_limit_s: float = 25.0         # Only used if branch-and-bound falls back to HiGHS MIP
    rc_threshold_km: float = 2.0           # Initial reduced-cost window for LP reduced-cost fixing
    max_fixing_rounds: int = 6
    bb_node_limit: int = 60                # Own branch-and-bound budget before HiGHS MIP (with cuts) takes over


class ExactBatchPolicy:
    """
    Per-batch set-partitioning MIP:
      min  sum_c cost_c * x_c + sum_o penalty_o * u_o
      s.t. sum_{c of courier d} x_c = 1                  for every courier d
           sum_{c containing o} x_c + u_o = 1            for every open order o (pool + held)
    where cost_c is the exact optimal open-route length (DP) of courier d serving its on-board
    deliveries plus the pickup/delivery pairs of column c. The plan is optimal over the enumerated
    column space with a proof (stats["mip_optimal"]), and every route is exactly sequenced.
    """
    name = "exact"

    def __init__(self, config: Optional[ExactConfig] = None):
        self.config = config or ExactConfig()
        if self.config.max_release_per_batch not in (0, 1):
            raise ValueError("max_release_per_batch must be 0 or 1")
        self.cache = DistanceCache()

    def warm_up(self):
        kernels.warm_up()

    def plan(self, snap: BatchSnapshot) -> PlanResult:
        cfg = self.config
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()

        self.cache.add_orders(snap.orders.values())
        node_dist = self.cache.node_dist
        courier_dist = self.cache.courier_rows(snap.courier_lat, snap.courier_lng)
        t1 = time.perf_counter()
        timings["matrix_ms"] = (t1 - t0) * 1000

        n_couriers = len(snap.courier_lat)
        held = [[oid for oid, kind in plan if kind == PICKUP] for plan in snap.plans]
        holder = {oid: d for d, orders in enumerate(held) for oid in orders}
        open_orders = list(snap.pool) + [oid for orders in held for oid in orders]
        row_of = {oid: i for i, oid in enumerate(open_orders)}
        weight = {oid: snap.orders[oid].weight for oid in open_orders}
        for orders in snap.onboard:
            for oid in orders:
                weight[oid] = snap.orders[oid].weight

        candidates = self._candidates(snap, open_orders, holder, row_of, node_dist, courier_dist)
        t2 = time.perf_counter()
        timings["candidates_ms"] = (t2 - t1) * 1000

        cols = self._enumerate_columns(snap, held, candidates, weight)
        col_courier, col_orders, col_ptr, stop_node, stop_req, stop_dw, stop_order, stop_kind, keep_all, col_adds = cols
        t3 = time.perf_counter()
        timings["columns_ms"] = (t3 - t2) * 1000

        costs = kernels.route_costs(col_courier, col_ptr, stop_node, stop_req, stop_dw,
                                    snap.courier_load, snap.courier_capacity, node_dist, courier_dist,
                                    kernels.default_workers())
        t4 = time.perf_counter()
        timings["dp_ms"] = (t4 - t3) * 1000

        feasible = np.flatnonzero(np.isfinite(costs))
        if not np.isfinite(costs[keep_all]).all():
            raise RuntimeError("A courier's current itinerary became infeasible")
        release_s = np.array([snap.orders[oid].release_batch * snap.batch_seconds for oid in open_orders])
        penalty = cfg.unassigned_penalty_km + cfg.age_penalty_km_per_min * (snap.time_s - release_s) / 60.0
        chosen, mip_stats = self._solve_mip(n_couriers, open_orders, row_of, col_courier, col_orders,
                                            col_adds, costs, feasible, keep_all, penalty)
        t5 = time.perf_counter()
        timings["mip_ms"] = (t5 - t4) * 1000

        plans: List[List[Stop]] = [[] for _ in range(n_couriers)]
        for c in chosen:
            d = int(col_courier[c])
            a, b = col_ptr[c], col_ptr[c + 1]
            if a == b:
                continue
            _, path = kernels.route_path(d, stop_node[a:b], stop_req[a:b], stop_dw[a:b],
                                         float(snap.courier_load[d]), float(snap.courier_capacity[d]),
                                         node_dist, courier_dist)
            plans[d] = [(int(stop_order[a + i]), int(stop_kind[a + i])) for i in path]
        timings["decode_ms"] = (time.perf_counter() - t5) * 1000

        stats = {
            "open_orders": len(open_orders),
            "columns": int(len(col_courier)),
            "feasible_columns": int(len(feasible)),
            **mip_stats,
            **timings,
        }
        return PlanResult(plans=plans, stats=stats)

    def _candidates(self, snap, open_orders, holder, row_of, node_dist, courier_dist) -> List[List[int]]:
        """
        Candidate orders each courier may add: its L closest open orders plus every open order for
        which it is among the K closest couriers. Closeness is the distance from the order's pickup
        to the courier's position or to any stop already on its itinerary.
        """
        cfg = self.config
        n_couriers = len(snap.courier_lat)
        n_open = len(open_orders)
        if n_open == 0:
            return [[] for _ in range(n_couriers)]

        pick = np.array([2 * oid for oid in open_orders], dtype=np.int64)
        score = courier_dist[:, pick].copy()
        for d, plan in enumerate(snap.plans):
            if plan:
                nodes = np.array([2 * oid + kind for oid, kind in plan], dtype=np.int64)
                np.minimum(score[d], node_dist[np.ix_(nodes, pick)].min(axis=0), out=score[d])
        for oid, d in holder.items():
            score[d, row_of[oid]] = np.inf  # Already in the courier's keep set
        weights = np.array([snap.orders[oid].weight for oid in open_orders])
        score[weights[None, :] > snap.courier_capacity[:, None]] = np.inf

        chosen = np.zeros((n_couriers, n_open), dtype=bool)
        top_l = min(cfg.candidates_per_courier, n_open)
        if top_l > 0:
            idx = np.argpartition(score, top_l - 1, axis=1)[:, :top_l]
            chosen[np.arange(n_couriers)[:, None], idx] = True
        top_k = min(cfg.couriers_per_order, n_couriers)
        if top_k > 0:
            idx = np.argpartition(score, top_k - 1, axis=0)[:top_k, :]
            chosen[idx, np.arange(n_open)[None, :]] = True
        chosen &= np.isfinite(score)

        out = []
        for d in range(n_couriers):
            rows = np.flatnonzero(chosen[d])
            rows = rows[np.argsort(score[d, rows], kind="stable")]
            out.append([open_orders[r] for r in rows])
        return out

    def _enumerate_columns(self, snap, held, candidates, weight):
        cfg = self.config
        cap_orders = snap.max_orders_held
        col_courier: List[int] = []
        col_orders: List[Tuple[int, ...]] = []
        col_ptr: List[int] = [0]
        stop_node: List[int] = []
        stop_req: List[int] = []
        stop_dw: List[float] = []
        stop_order: List[int] = []
        stop_kind: List[int] = []
        keep_all: List[int] = []  # Column index of each courier's unchanged held set (always feasible)
        col_adds: List[int] = []  # Number of newly added orders per column

        for d in range(len(snap.courier_lat)):
            onboard = snap.onboard[d]
            # On-board deliveries are shared by every column of this courier
            base_node = [2 * oid + DELIVERY for oid in onboard]
            base_dw = [-weight[oid] for oid in onboard]
            keeps = [tuple(held[d])]
            if cfg.max_release_per_batch == 1:
                keeps += [tuple(o for o in held[d] if o != r) for r in held[d]]

            keep_all.append(len(col_courier))
            for keep in keeps:
                free = cap_orders - len(onboard) - len(keep)
                max_add = cfg.max_add_per_batch if len(keep) == len(held[d]) else cfg.max_add_when_releasing
                for k in range(min(max_add, max(free, 0)) + 1):
                    for added in combinations(candidates[d], k):
                        orders = keep + added
                        n_stops = len(onboard) + 2 * len(orders)
                        if n_stops > kernels.MAX_ROUTE_STOPS:
                            raise ValueError(f"Route of {n_stops} stops exceeds kernel limit")
                        stop_node.extend(base_node)
                        stop_req.extend([-1] * len(onboard))
                        stop_dw.extend(base_dw)
                        stop_order.extend(onboard)
                        stop_kind.extend([DELIVERY] * len(onboard))
                        local = len(onboard)
                        for oid in orders:
                            w = weight[oid]
                            stop_node.extend((2 * oid, 2 * oid + 1))
                            stop_req.extend((-1, local))
                            stop_dw.extend((w, -w))
                            stop_order.extend((oid, oid))
                            stop_kind.extend((PICKUP, DELIVERY))
                            local += 2
                        col_courier.append(d)
                        col_orders.append(orders)
                        col_adds.append(k)
                        col_ptr.append(len(stop_node))

        return (
            np.array(col_courier, dtype=np.int64),
            col_orders,
            np.array(col_ptr, dtype=np.int64),
            np.array(stop_node, dtype=np.int64),
            np.array(stop_req, dtype=np.int64),
            np.array(stop_dw, dtype=np.float64),
            np.array(stop_order, dtype=np.int64),
            np.array(stop_kind, dtype=np.int64),
            np.array(keep_all, dtype=np.int64),
            np.array(col_adds, dtype=np.int64),
        )

    def _solve_mip(self, n_couriers, open_orders, row_of, col_courier, col_orders, col_adds, costs,
                   feasible, keep_all, penalty):
        """
        Build the 0/1 matrix (courier rows, then open-order rows; column variables, then one
        unassigned slack per order) and solve it exactly with dispatch.setpart. The LP starts from
        each courier's keep-all column, every slack and every column adding at most one order.
        """
        cfg = self.config
        n_open = len(open_orders)
        n_x = len(feasible)

        rows: List[int] = []
        cols: List[int] = []
        for j, c in enumerate(feasible):
            rows.append(int(col_courier[c]))
            cols.append(j)
            for oid in col_orders[c]:
                rows.append(n_couriers + row_of[oid])
                cols.append(j)
        rows.extend(n_couriers + np.arange(n_open))
        cols.extend(n_x + np.arange(n_open))
        a = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_couriers + n_open, n_x + n_open)).tocsc()

        objective = np.concatenate([costs[feasible], penalty])

        always = np.zeros(n_x + n_open, dtype=bool)
        always[np.searchsorted(feasible, keep_all)] = True
        always[n_x:] = True
        initial = always.copy()
        initial[:n_x] |= col_adds[feasible] <= 1

        sp = solve_set_partition(
            objective, a, n_x, n_couriers, initial, always,
            tau=cfg.rc_threshold_km,
            node_limit=cfg.bb_node_limit,
            max_fixing_rounds=cfg.max_fixing_rounds,
            fallback_time_limit_s=cfg.mip_time_limit_s,
        )
        z = sp.z
        chosen = feasible[z[:n_x] > 0.5]
        gap = (sp.objective - sp.lower_bound) / max(abs(sp.objective), 1e-9)
        stats = {
            "mip_optimal": float(sp.proved_optimal),
            "mip_gap": 0.0 if sp.proved_optimal else gap,
            "lp_gap": gap,
            "objective": sp.objective,
            "route_km": float(costs[chosen].sum()),
            "unassigned": int(np.round(z[n_x:].sum())),
            "mip_columns": sp.restricted_columns,
            "fixing_rounds": sp.fixing_rounds,
            "lp_rounds": sp.lp_rounds,
            "bb_nodes": sp.bb_nodes,
            "fallback": float(sp.used_fallback),
            "lp_ms": sp.lp_ms,
            "bb_ms": sp.bb_ms,
        }
        return chosen, stats
