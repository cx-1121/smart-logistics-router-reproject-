import itertools
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispatch import kernels
from dispatch.exact_policy import ExactBatchPolicy, ExactConfig
from dispatch.geo import DistanceCache, haversine_km
from dispatch.greedy_policy import GreedyInsertionPolicy
from dispatch.scenario import Order, generate_kl_scenario
from dispatch.setpart import solve_set_partition
from dispatch.simulator import (
    DELIVERY, PICKUP, BatchSnapshot, PlanValidationError, RollingSimulator,
)


def brute_force_route(start_d, dist, req, dw, start_load, cap):
    """Cheapest precedence- and capacity-feasible visiting order, by trying every permutation."""
    n = len(req)
    best = np.inf
    for perm in itertools.permutations(range(n)):
        pos = {s: i for i, s in enumerate(perm)}
        if any(req[s] >= 0 and pos[req[s]] > pos[s] for s in range(n)):
            continue
        load, ok = start_load, True
        for s in perm:
            load += dw[s]
            if load > cap + 1e-9:
                ok = False
                break
        if not ok:
            continue
        cost = start_d[perm[0]] + sum(dist[perm[i], perm[i + 1]] for i in range(n - 1))
        best = min(best, cost)
    return best


def random_route(rng, n_onboard, n_pairs):
    """Random stops: on-board deliveries first, then pickup/delivery pairs (node i = stop i)."""
    n = n_onboard + 2 * n_pairs
    pts = rng.uniform(0, 10, size=(n + 1, 2))
    full = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    node_dist = np.ascontiguousarray(full[1:, 1:])
    courier_dist = np.ascontiguousarray(full[:1, 1:])
    req, dw = [], []
    onboard_w = rng.uniform(1, 5, size=n_onboard)
    for w in onboard_w:
        req.append(-1)
        dw.append(-w)
    for _ in range(n_pairs):
        w = rng.uniform(1, 5)
        req += [-1, len(req)]
        dw += [w, -w]
    start_load = float(onboard_w.sum())
    cap = start_load + rng.uniform(0, 8)  # Often binding, sometimes infeasible
    return (np.arange(n, dtype=np.int64), np.array(req, dtype=np.int64), np.array(dw),
            start_load, cap, node_dist, courier_dist)


class TestRouteDP(unittest.TestCase):
    """The DP kernel must match exhaustive search on precedence- and capacity-constrained routes."""

    def test_route_costs_match_brute_force(self):
        rng = np.random.default_rng(0)
        for _ in range(150):
            n_onboard, n_pairs = int(rng.integers(0, 3)), int(rng.integers(0, 4))
            nodes, req, dw, load, cap, node_dist, courier_dist = random_route(rng, n_onboard, n_pairs)
            n = len(nodes)
            got = kernels.route_costs(np.zeros(1, dtype=np.int64), np.array([0, n], dtype=np.int64), nodes, req, dw,
                                      np.array([load]), np.array([cap]), node_dist, courier_dist, 2)[0]
            want = brute_force_route(courier_dist[0], node_dist, req, dw, load, cap) if n else 0.0
            if np.isinf(want):
                self.assertTrue(np.isinf(got))
            else:
                self.assertAlmostEqual(got, want, places=9)

    def test_route_path_is_feasible_and_costs_what_it_claims(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            nodes, req, dw, load, cap, node_dist, courier_dist = random_route(rng, 2, 3)
            cost, path = kernels.route_path(0, nodes, req, dw, load, cap, node_dist, courier_dist)
            if np.isinf(cost):
                continue
            pos = {s: i for i, s in enumerate(path)}
            self.assertEqual(sorted(path), list(range(len(nodes))))
            for s in range(len(nodes)):
                if req[s] >= 0:
                    self.assertLess(pos[req[s]], pos[s], "Delivery visited before its pickup")
            running = load + np.cumsum(dw[path])
            self.assertLessEqual(running.max(), cap + 1e-9)
            walked = courier_dist[0, path[0]] + sum(node_dist[path[i], path[i + 1]] for i in range(len(path) - 1))
            self.assertAlmostEqual(cost, walked, places=9)

    def test_infeasible_capacity_returns_inf(self):
        node_dist = np.zeros((2, 2))
        courier_dist = np.zeros((1, 2))
        cost, _ = kernels.route_path(0, np.array([0, 1], dtype=np.int64), np.array([-1, 0], dtype=np.int64),
                                     np.array([6.0, -6.0]), 0.0, 5.0, node_dist, courier_dist)
        self.assertTrue(np.isinf(cost))


class TestPairInsertion(unittest.TestCase):
    """Greedy insertion kernel must find the cheapest feasible (pickup slot, delivery slot)."""

    def test_matches_exhaustive_slots(self):
        rng = np.random.default_rng(2)
        for _ in range(100):
            m = int(rng.integers(0, 6))
            pts = rng.uniform(0, 10, size=(m + 3, 2))
            full = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
            node_dist = np.ascontiguousarray(full[1:, 1:])  # nodes 0..m-1 plan, m pickup, m+1 delivery
            courier_dist = np.ascontiguousarray(full[:1, 1:])
            plan_node = np.arange(m, dtype=np.int64)
            plan_dw = rng.choice([-2.0, 3.0], size=m)
            start_load = 4.0
            cap = start_load + max(0.0, np.cumsum(plan_dw).max(initial=0.0)) + rng.uniform(0, 6)
            weight = float(rng.uniform(1, 5))
            delta, i, j = kernels.best_pair_insertion(0, plan_node, plan_dw, start_load, cap, m, m + 1, weight,
                                                      node_dist, courier_dist)

            def walk(seq):
                return courier_dist[0, seq[0]] + sum(node_dist[seq[k], seq[k + 1]] for k in range(len(seq) - 1)) if seq else 0.0
            base = walk(list(plan_node))
            best = np.inf
            for pi in range(m + 1):
                for dj in range(pi, m + 1):
                    seq = list(plan_node[:pi]) + [m] + list(plan_node[pi:dj]) + [m + 1] + list(plan_node[dj:])
                    dws = list(plan_dw[:pi]) + [weight] + list(plan_dw[pi:dj]) + [-weight] + list(plan_dw[dj:])
                    if (start_load + np.cumsum(dws)).max() > cap + 1e-9:
                        continue
                    best = min(best, walk(seq) - base)
            if np.isinf(best):
                self.assertTrue(np.isinf(delta))
            else:
                self.assertAlmostEqual(delta, best, places=9)


class TestSetPartition(unittest.TestCase):
    """The warm-started branch-and-bound must reach HiGHS' proven MIP optimum."""

    def test_matches_highs_mip_on_random_instances(self):
        rng = np.random.default_rng(3)
        for trial in range(40):
            n_groups, n_items = int(rng.integers(2, 6)), int(rng.integers(3, 9))
            cols, costs = [], []
            for g in range(n_groups):
                cols.append([g])          # Empty column keeps every group feasible
                costs.append(0.0)
                for _ in range(int(rng.integers(3, 12))):
                    items = rng.choice(n_items, size=int(rng.integers(1, 4)), replace=False)
                    cols.append([g] + [n_groups + int(i) for i in items])
                    costs.append(float(rng.uniform(1, 20)))
            n_x = len(cols)
            for o in range(n_items):
                cols.append([n_groups + o])
                costs.append(float(rng.uniform(20, 40)))   # Unassigned slack
            rows = [r for c in cols for r in c]
            col_idx = [j for j, c in enumerate(cols) for _ in c]
            a = csc_matrix((np.ones(len(rows)), (rows, col_idx)), shape=(n_groups + n_items, len(cols)))
            cost = np.array(costs)
            always = np.zeros(len(cols), dtype=bool)
            always[[j for j, c in enumerate(cols) if len(c) == 1]] = True
            res = solve_set_partition(cost, a, n_x, n_groups, always.copy(), always, tau=0.0)

            integ = np.r_[np.ones(n_x), np.zeros(n_items)]
            ref = milp(cost, integrality=integ, bounds=Bounds(0, 1), constraints=LinearConstraint(a, 1, 1),
                       options={"mip_rel_gap": 0.0})
            self.assertTrue(res.proved_optimal, f"trial {trial}")
            self.assertAlmostEqual(res.objective, ref.fun, places=6, msg=f"trial {trial}")
            np.testing.assert_allclose(a @ res.z, 1.0, atol=1e-7)
            self.assertTrue(np.all(np.abs(res.z[:n_x] - np.round(res.z[:n_x])) < 1e-6))

    def test_odd_cycle_is_cut_at_the_root(self):
        """Three couriers, three orders, each courier column holding two of them: the LP takes every
        pair at 0.5 (value 1.5) while any integer plan serves at most one pair. A clique cut must
        lift the root bound to the integer optimum."""
        # Rows: groups 0..2, items 3..5. Columns: a pair per group, an empty column per group, slacks.
        cols = [[0, 3, 4], [1, 4, 5], [2, 5, 3], [0], [1], [2], [3], [4], [5]]
        cost = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
        rows = [r for c in cols for r in c]
        col_idx = [j for j, c in enumerate(cols) for _ in c]
        a = csc_matrix((np.ones(len(rows)), (rows, col_idx)), shape=(6, len(cols)))
        mask = np.ones(len(cols), dtype=bool)
        res = solve_set_partition(cost, a, 6, 3, mask, mask, tau=100.0)
        self.assertTrue(res.proved_optimal)
        self.assertGreaterEqual(res.cuts, 1)
        self.assertAlmostEqual(res.objective, 11.0)       # One pair plus one unassigned order
        self.assertAlmostEqual(res.lower_bound, 11.0)     # Without the cut the root LP is 1.5


def make_order(oid, p, d, weight=2.0, release_batch=0):
    return Order(id=oid, pickup_lat=p[0], pickup_lng=p[1], delivery_lat=d[0], delivery_lng=d[1],
                 weight=weight, release_batch=release_batch)


class TestExactPolicyOptimality(unittest.TestCase):
    """With a column space covering every assignment, the batch plan must equal exhaustive search."""

    def test_small_batch_matches_exhaustive_assignment(self):
        rng = np.random.default_rng(4)
        for trial in range(6):
            n_couriers, n_orders = 2, 3
            c_lat = rng.uniform(3.05, 3.25, n_couriers)
            c_lng = rng.uniform(101.55, 101.75, n_couriers)
            orders = {}
            for i in range(n_orders):
                p = (rng.uniform(3.05, 3.25), rng.uniform(101.55, 101.75))
                d = (p[0] + rng.uniform(-0.02, 0.02), p[1] + rng.uniform(-0.02, 0.02))
                orders[i] = make_order(i, p, d, weight=float(rng.uniform(1, 5)))
            capacity = np.array([7.0, 30.0])  # Courier 0 cannot carry everything at once
            snap = BatchSnapshot(
                batch=0, time_s=0.0, orders=orders, pool=list(orders),
                courier_lat=c_lat, courier_lng=c_lng, courier_capacity=capacity,
                courier_load=np.zeros(n_couriers), onboard=[[], []], plans=[[], []],
                max_orders_held=6, batch_seconds=30.0,
            )
            cfg = ExactConfig(max_add_per_batch=3, candidates_per_courier=10, couriers_per_order=2)
            result = ExactBatchPolicy(cfg).plan(snap)
            self.assertEqual(result.stats["mip_optimal"], 1.0)

            cache = DistanceCache()
            cache.add_orders(orders.values())
            courier_dist = cache.courier_rows(c_lat, c_lng)

            def route_cost(d, oids):
                if not oids:
                    return 0.0
                nodes, req, dw = [], [], []
                for o in oids:
                    nodes += [2 * o, 2 * o + 1]
                    req += [-1, len(req)]
                    dw += [orders[o].weight, -orders[o].weight]
                nodes = np.array(nodes)
                return brute_force_route(courier_dist[d, nodes], cache.node_dist[np.ix_(nodes, nodes)],
                                         np.array(req), np.array(dw), 0.0, capacity[d])

            best = np.inf
            for assign in itertools.product([-1, 0, 1], repeat=n_orders):
                total = sum(cfg.unassigned_penalty_km for a in assign if a < 0)
                for d in range(n_couriers):
                    total += route_cost(d, [o for o in range(n_orders) if assign[o] == d])
                best = min(best, total)
            self.assertAlmostEqual(result.stats["objective"], best, places=6, msg=f"trial {trial}")


class TestRollingSimulator(unittest.TestCase):
    """End-to-end invariants of the rolling simulation for both policies."""

    def setUp(self):
        self.scenario = generate_kl_scenario(num_couriers=6, num_orders=40, orders_per_batch=10, seed=11)

    def _check(self, result):
        self.assertEqual(len(result.delivery_latency_s), 40)
        self.assertTrue(np.all(result.delivery_latency_s > 0))
        self.assertGreater(result.total_km, 0)
        # Every order needs at least its own pickup->delivery leg
        lower = sum(haversine_km(o.pickup_lat, o.pickup_lng, o.delivery_lat, o.delivery_lng)
                    for o in self.scenario.orders)
        self.assertGreaterEqual(result.total_km + 1e-6, lower)

    def test_exact_policy_delivers_everything(self):
        result = RollingSimulator(self.scenario, ExactBatchPolicy()).run()
        self._check(result)
        self.assertTrue(all(b.stats["mip_optimal"] == 1.0 for b in result.batches))

    def test_greedy_policy_delivers_everything(self):
        result = RollingSimulator(self.scenario, GreedyInsertionPolicy()).run()
        self._check(result)
        self.assertEqual(result.reassignments, 0)

    def test_scenario_is_deterministic(self):
        again = generate_kl_scenario(num_couriers=6, num_orders=40, orders_per_batch=10, seed=11)
        self.assertEqual(self.scenario, again)


class TestOfflineReference(unittest.TestCase):
    """Offline full-information solutions must be feasible and bracketed by bound and online run."""

    @classmethod
    def setUpClass(cls):
        cls.scenario = generate_kl_scenario(num_couriers=5, num_orders=30, orders_per_batch=10, seed=8)
        cls.online = RollingSimulator(cls.scenario, ExactBatchPolicy()).run()

    def _check_routes(self, routes):
        sc = self.scenario
        seen = []
        for d, route in enumerate(routes):
            load, held, picked = 0.0, 0, set()
            for oid, kind in route:
                w = sc.orders[oid].weight
                if kind == PICKUP:
                    picked.add(oid)
                    load += w
                    held += 1
                else:
                    self.assertIn(oid, picked, "Delivery before pickup or on another courier")
                    load -= w
                    held -= 1
                    seen.append(oid)
                self.assertLessEqual(load, sc.couriers[d].capacity + 1e-9)
                self.assertLessEqual(held, sc.max_orders_held)
        self.assertEqual(sorted(seen), list(range(len(sc.orders))))

    def test_replay_of_online_visits_never_exceeds_driven_km(self):
        from dispatch import offline
        replay = offline.schedule(self.scenario, self.online.courier_visits)
        self.assertLessEqual(replay.total_km, self.online.total_km + 1e-6)
        self.assertTrue(np.all(replay.delivery_latency_s > 0))

    def test_offline_solutions_are_feasible_and_bracketed(self):
        from dispatch import offline
        lower = offline.assignment_lower_bound(self.scenario)
        replay_km = offline.schedule(self.scenario, self.online.courier_visits).total_km
        free = offline.solve_offline(self.scenario, self.online.courier_visits, None, time_limit_s=2)
        deadline_s = self.online.makespan_s + 60
        timed = offline.solve_offline(self.scenario, self.online.courier_visits, deadline_s, time_limit_s=2)
        for res in (free, timed):
            self._check_routes(res.routes)
            self.assertLessEqual(lower, res.schedule.total_km + 1e-6)
            self.assertLessEqual(res.schedule.total_km, replay_km + 0.01)  # Warm-started from the online routes
        self.assertLessEqual(timed.schedule.makespan_s, deadline_s + 1e-6)


class _ScriptedPolicy:
    name = "scripted"

    def __init__(self, plans):
        self.plans = plans

    def plan(self, snapshot):
        from dispatch.simulator import PlanResult
        return PlanResult(plans=self.plans)


class TestPlanValidation(unittest.TestCase):
    """The simulator must reject itineraries that break PDP, capacity or holding rules."""

    def setUp(self):
        self.scenario = generate_kl_scenario(num_couriers=2, num_orders=3, orders_per_batch=3, seed=5,
                                             capacity_range=(5.0, 5.0), weight_range=(3.0, 3.0))

    def _run(self, plans):
        RollingSimulator(self.scenario, _ScriptedPolicy(plans), max_batches=1).run()

    def test_delivery_before_pickup_rejected(self):
        with self.assertRaises(PlanValidationError):
            self._run([[(0, DELIVERY), (0, PICKUP)], []])

    def test_order_on_two_couriers_rejected(self):
        with self.assertRaises(PlanValidationError):
            self._run([[(0, PICKUP), (0, DELIVERY)], [(0, PICKUP), (0, DELIVERY)]])

    def test_capacity_overflow_rejected(self):
        with self.assertRaises(PlanValidationError):
            self._run([[(0, PICKUP), (1, PICKUP), (0, DELIVERY), (1, DELIVERY)], []])

    def test_pickup_without_delivery_rejected(self):
        with self.assertRaises(PlanValidationError):
            self._run([[(0, PICKUP)], []])


if __name__ == "__main__":
    unittest.main()
