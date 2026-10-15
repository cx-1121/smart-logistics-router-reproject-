"""
Offline full-information reference: every order and its release time are known at t = 0.

A rolling dispatcher can never beat the offline optimum, so (online km) / (offline km) measures
what not knowing the future costs. The offline problem (50 couriers, 500 pickup/delivery pairs,
capacity and on-board limits) is far beyond exact solution, so it is bracketed:
  - upper bound: best solution OR-Tools' guided local search finds, warm-started from the routes
    the online run actually drove (so it is never worse than online);
  - lower bound: an assignment relaxation (every stop needs a distinct predecessor: another stop
    or a courier start), valid for any schedule.
Two offline variants are solved:
  - "km": distance only, couriers may wait as long as they like (the absolute km floor);
  - "deadline": pickups no earlier than release and everything delivered by the online run's
    makespan, i.e. the same service horizon as the online policy.
"""
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from scipy.optimize import linear_sum_assignment

from dispatch.geo import haversine_matrix
from dispatch.scenario import Scenario
from dispatch.simulator import DELIVERY, PICKUP, Stop

_FORBIDDEN = 1e9


@dataclass
class Schedule:
    """Timing of a set of routes driven at the scenario speed, waiting at pickups until release."""
    total_km: float
    makespan_s: float
    delivery_latency_s: np.ndarray


@dataclass
class OfflineResult:
    variant: str
    routes: List[List[Stop]]
    schedule: Schedule
    solve_s: float
    start_km: float        # Warm start (the online run's visit sequences) before the search


def _coords(scenario: Scenario):
    """Coordinates of courier starts (0..C-1) then stops (C + 2*o pickup, C + 2*o + 1 delivery)."""
    lat = [c.lat for c in scenario.couriers]
    lng = [c.lng for c in scenario.couriers]
    for o in scenario.orders:
        lat += [o.pickup_lat, o.delivery_lat]
        lng += [o.pickup_lng, o.delivery_lng]
    return np.array(lat), np.array(lng)


def distance_matrix_km(scenario: Scenario) -> np.ndarray:
    lat, lng = _coords(scenario)
    return haversine_matrix(lat, lng, lat, lng)


def schedule(scenario: Scenario, routes: List[List[Stop]], dist_km: Optional[np.ndarray] = None) -> Schedule:
    """Drive each courier's route in straight lines, waiting at a pickup until the order is released."""
    dist_km = distance_matrix_km(scenario) if dist_km is None else dist_km
    n_c = len(scenario.couriers)
    speed_km_s = scenario.speed_kmh / 3600.0
    release = np.array([o.release_batch * scenario.batch_seconds for o in scenario.orders])
    delivered = np.full(len(scenario.orders), np.nan)
    total_km = 0.0
    for d, route in enumerate(routes):
        node, t = d, 0.0
        for oid, kind in route:
            nxt = n_c + 2 * oid + kind
            leg = dist_km[node, nxt]
            total_km += leg
            t += leg / speed_km_s
            if kind == PICKUP:
                t = max(t, release[oid])
            else:
                delivered[oid] = t
            node = nxt
    if np.isnan(delivered).any():
        raise ValueError("Routes do not deliver every order")
    return Schedule(total_km=total_km, makespan_s=float(delivered.max()), delivery_latency_s=delivered - release)


def assignment_lower_bound(scenario: Scenario, dist_km: Optional[np.ndarray] = None) -> float:
    """
    Every stop is entered from exactly one distinct predecessor (another stop or a courier start),
    so the min-cost assignment of predecessors to stops bounds any schedule's driven km from below.
    Forbidden: a stop preceding itself, a delivery preceding its own pickup, a courier start
    preceding a delivery (each courier starts empty).
    """
    dist_km = distance_matrix_km(scenario) if dist_km is None else dist_km
    n_c = len(scenario.couriers)
    n_stops = 2 * len(scenario.orders)
    cost = dist_km[:, n_c:].T.copy()  # (stop, predecessor)
    stops = np.arange(n_stops)
    cost[stops, n_c + stops] = _FORBIDDEN
    pickups = stops[0::2]
    cost[pickups, n_c + pickups + 1] = _FORBIDDEN
    cost[1::2, :n_c] = _FORBIDDEN
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].sum())


def solve_offline(
    scenario: Scenario,
    warm_routes: List[List[Stop]],
    deadline_s: Optional[float] = None,
    time_limit_s: float = 60.0,
) -> OfflineResult:
    """
    OR-Tools model: one vehicle per courier from its start to a free dummy end (open routes),
    pickup/delivery pairs on the same vehicle in order, weight capacity and the on-board order limit.
    With deadline_s, a time dimension adds travel time at the scenario speed, waiting at pickups
    until release, and every route ending by deadline_s.
    """
    t0 = time.perf_counter()
    n_c = len(scenario.couriers)
    n_o = len(scenario.orders)
    dist_km = distance_matrix_km(scenario)
    n_nodes = n_c + 2 * n_o + 1
    end = n_nodes - 1

    meters = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    meters[:end, :end] = np.round(dist_km * 1000).astype(np.int64)
    meters[:, end] = 0  # Routes end wherever the last delivery is

    manager = pywrapcp.RoutingIndexManager(n_nodes, n_c, list(range(n_c)), [end] * n_c)
    routing = pywrapcp.RoutingModel(manager)
    dist_cb = routing.RegisterTransitMatrix(meters.tolist())
    routing.SetArcCostEvaluatorOfAllVehicles(dist_cb)
    routing.AddDimension(dist_cb, 0, int(meters.sum()), True, "Distance")
    distance = routing.GetDimensionOrDie("Distance")

    weight = np.zeros(n_nodes, dtype=np.int64)
    count = np.zeros(n_nodes, dtype=np.int64)
    for o in scenario.orders:
        w = int(round(o.weight * 10))
        weight[n_c + 2 * o.id], weight[n_c + 2 * o.id + 1] = w, -w
        count[n_c + 2 * o.id], count[n_c + 2 * o.id + 1] = 1, -1
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitVector(weight.tolist()), 0,
        [int(round(c.capacity * 10)) for c in scenario.couriers], True, "Weight")
    routing.AddDimension(routing.RegisterUnaryTransitVector(count.tolist()), 0,
                         scenario.max_orders_held, True, "Count")

    time_dim = None
    if deadline_s is not None:
        speed_m_s = scenario.speed_kmh / 3.6
        seconds = np.ceil(meters / speed_m_s).astype(np.int64)  # Rounding up keeps the deadline honest
        horizon = int(np.floor(deadline_s))
        routing.AddDimension(routing.RegisterTransitMatrix(seconds.tolist()), horizon, horizon, True, "Time")
        time_dim = routing.GetDimensionOrDie("Time")
        for v in range(n_c):
            time_dim.CumulVar(routing.End(v)).SetMax(horizon)

    solver = routing.solver()
    for o in scenario.orders:
        p = manager.NodeToIndex(n_c + 2 * o.id)
        d = manager.NodeToIndex(n_c + 2 * o.id + 1)
        routing.AddPickupAndDelivery(p, d)
        solver.Add(routing.VehicleVar(p) == routing.VehicleVar(d))
        solver.Add(distance.CumulVar(p) <= distance.CumulVar(d))
        if time_dim is not None:
            time_dim.CumulVar(p).SetMin(int(np.ceil(o.release_batch * scenario.batch_seconds)))

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromMilliseconds(int(time_limit_s * 1000))
    routing.CloseModelWithParameters(params)

    warm_nodes = [[n_c + 2 * oid + kind for oid, kind in route] for route in warm_routes]
    initial = routing.ReadAssignmentFromRoutes(warm_nodes, True)
    if initial is None:
        raise RuntimeError("The online routes are not a feasible warm start for the offline model")
    solution = routing.SolveFromAssignmentWithParameters(initial, params)
    if solution is None:
        raise RuntimeError("OR-Tools found no offline solution")

    routes: List[List[Stop]] = []
    for v in range(n_c):
        idx = solution.Value(routing.NextVar(routing.Start(v)))
        route: List[Stop] = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx) - n_c
            route.append((node // 2, DELIVERY if node % 2 else PICKUP))
            idx = solution.Value(routing.NextVar(idx))
        routes.append(route)

    return OfflineResult(
        variant="deadline" if deadline_s is not None else "km",
        routes=routes,
        schedule=schedule(scenario, routes, dist_km),
        solve_s=time.perf_counter() - t0,
        start_km=schedule(scenario, warm_routes, dist_km).total_km,
    )
