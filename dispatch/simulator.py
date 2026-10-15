import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np

from dispatch.geo import haversine_km
from dispatch.scenario import Order, Scenario

PICKUP, DELIVERY = 0, 1
Stop = Tuple[int, int]  # (order_id, PICKUP | DELIVERY); its matrix node id is 2 * order_id + kind


@dataclass
class BatchSnapshot:
    """Everything a policy may see at a batch boundary. Unreleased orders are never exposed."""
    batch: int
    time_s: float
    orders: Dict[int, Order]          # Every released order (including delivered ones)
    pool: List[int]                   # Released orders not assigned to any courier
    courier_lat: np.ndarray
    courier_lng: np.ndarray
    courier_capacity: np.ndarray
    courier_load: np.ndarray          # Weight currently on board
    onboard: List[List[int]]          # Picked-up orders per courier (locked to that courier)
    plans: List[List[Stop]]           # Current itinerary per courier; its pickups are the held orders
    max_orders_held: int
    batch_seconds: float


@dataclass
class PlanResult:
    plans: List[List[Stop]]           # New itinerary for every courier
    stats: Dict[str, float] = field(default_factory=dict)


class DispatchPolicy(Protocol):
    name: str

    def plan(self, snapshot: BatchSnapshot) -> PlanResult: ...


@dataclass
class BatchRecord:
    batch: int
    time_s: float
    released: int
    open_orders: int        # Pool + held orders the policy could (re)assign
    unassigned_after: int
    onboard: int
    solve_ms: float
    reassigned: int
    stats: Dict[str, float]


@dataclass
class SimulationResult:
    policy: str
    batches: List[BatchRecord]
    total_km: float
    makespan_s: float
    delivery_latency_s: np.ndarray    # Delivery time minus release time, per order
    reassignments: int
    courier_visits: List[List[Stop]]  # Stops each courier actually reached, in order

    @property
    def solve_ms(self) -> np.ndarray:
        return np.array([b.solve_ms for b in self.batches])


class PlanValidationError(Exception):
    pass


class RollingSimulator:
    """
    Time-stepped courier simulation. At each batch boundary the policy re-plans every courier;
    couriers then drive their itinerary in straight lines at speed_kmh for batch_seconds, picking
    up and delivering as they reach stops. Orders not yet picked up may move between couriers;
    picked-up orders stay with their courier until delivered.
    """
    def __init__(self, scenario: Scenario, policy: DispatchPolicy, max_batches: int = 2000):
        self.scenario = scenario
        self.policy = policy
        self.max_batches = max_batches

    def run(self) -> SimulationResult:
        sc = self.scenario
        n_couriers = len(sc.couriers)
        self.lat = np.array([c.lat for c in sc.couriers])
        self.lng = np.array([c.lng for c in sc.couriers])
        self.capacity = np.array([c.capacity for c in sc.couriers])
        self.load = np.zeros(n_couriers)
        self.onboard: List[List[int]] = [[] for _ in range(n_couriers)]
        self.plans: List[List[Stop]] = [[] for _ in range(n_couriers)]
        self.km = np.zeros(n_couriers)
        self.visits: List[List[Stop]] = [[] for _ in range(n_couriers)]
        self.released: Dict[int, Order] = {}
        self.delivered_at: Dict[int, float] = {}
        self.pool: List[int] = []

        by_batch: Dict[int, List[Order]] = {}
        for o in sc.orders:
            by_batch.setdefault(o.release_batch, []).append(o)
        last_release = max(by_batch) if by_batch else -1

        records: List[BatchRecord] = []
        reassignments = 0
        for k in range(self.max_batches):
            t = k * sc.batch_seconds
            arriving = by_batch.get(k, [])
            for o in arriving:
                self.released[o.id] = o
                self.pool.append(o.id)
            if k > last_release and len(self.delivered_at) == len(sc.orders):
                break

            snapshot = BatchSnapshot(
                batch=k,
                time_s=t,
                orders=dict(self.released),
                pool=list(self.pool),
                courier_lat=self.lat.copy(),
                courier_lng=self.lng.copy(),
                courier_capacity=self.capacity.copy(),
                courier_load=self.load.copy(),
                onboard=[list(x) for x in self.onboard],
                plans=[list(p) for p in self.plans],
                max_orders_held=sc.max_orders_held,
                batch_seconds=sc.batch_seconds,
            )
            open_orders = len(self.pool) + sum(1 for p in self.plans for _, kind in p if kind == PICKUP)

            t0 = time.perf_counter()
            result = self.policy.plan(snapshot)
            solve_ms = (time.perf_counter() - t0) * 1000

            reassigned = self._apply(result.plans)
            reassignments += reassigned
            records.append(BatchRecord(
                batch=k,
                time_s=t,
                released=len(arriving),
                open_orders=open_orders,
                unassigned_after=len(self.pool),
                onboard=sum(len(x) for x in self.onboard),
                solve_ms=solve_ms,
                reassigned=reassigned,
                stats=result.stats,
            ))
            self._advance(t)
        else:
            raise RuntimeError(
                f"Simulation did not finish within {self.max_batches} batches "
                f"({len(self.delivered_at)}/{len(sc.orders)} delivered)"
            )

        latency = np.array([self.delivered_at[o.id] - o.release_batch * sc.batch_seconds for o in sc.orders])
        return SimulationResult(
            policy=self.policy.name,
            batches=records,
            total_km=float(self.km.sum()),
            makespan_s=max(self.delivered_at.values()) if self.delivered_at else 0.0,
            delivery_latency_s=latency,
            reassignments=reassignments,
            courier_visits=self.visits,
        )

    def _apply(self, new_plans: List[List[Stop]]) -> int:
        """Validate the policy's itineraries, then adopt them. Returns how many held orders changed courier."""
        sc = self.scenario
        if len(new_plans) != len(self.plans):
            raise PlanValidationError(f"Expected {len(self.plans)} plans, got {len(new_plans)}")

        old_holder = {oid: d for d, plan in enumerate(self.plans) for oid, kind in plan if kind == PICKUP}
        onboard_owner = {oid: d for d, orders in enumerate(self.onboard) for oid in orders}
        claimed: Dict[int, int] = {}

        for d, plan in enumerate(new_plans):
            onboard = set(self.onboard[d])
            picked: set = set()
            delivered: set = set()
            load = self.load[d]
            for oid, kind in plan:
                order = self.released.get(oid)
                if order is None or oid in self.delivered_at:
                    raise PlanValidationError(f"Courier {d}: order {oid} is not open")
                if kind == PICKUP:
                    if oid in onboard_owner or oid in picked:
                        raise PlanValidationError(f"Courier {d}: order {oid} picked up twice")
                    if claimed.setdefault(oid, d) != d:
                        raise PlanValidationError(f"Order {oid} assigned to couriers {claimed[oid]} and {d}")
                    picked.add(oid)
                    load += order.weight
                elif kind == DELIVERY:
                    if oid in delivered or (oid not in onboard and oid not in picked):
                        raise PlanValidationError(f"Courier {d}: order {oid} delivered before pickup or twice")
                    delivered.add(oid)
                    load -= order.weight
                else:
                    raise PlanValidationError(f"Courier {d}: unknown stop kind {kind}")
                if load > self.capacity[d] + 1e-9:
                    raise PlanValidationError(f"Courier {d}: load {load:.1f} exceeds capacity {self.capacity[d]:.1f}")
            if delivered != onboard | picked:
                raise PlanValidationError(f"Courier {d}: itinerary does not deliver every order it carries")
            if len(onboard) + len(picked) > sc.max_orders_held:
                raise PlanValidationError(f"Courier {d}: holds {len(onboard) + len(picked)} > {sc.max_orders_held} orders")

        self.plans = [list(p) for p in new_plans]
        reassigned = sum(1 for oid, d in claimed.items() if oid in old_holder and old_holder[oid] != d)
        open_ids = [oid for oid in self.released if oid not in self.delivered_at and oid not in onboard_owner]
        self.pool = sorted(oid for oid in open_ids if oid not in claimed)
        return reassigned

    def _advance(self, t: float):
        """Drive every courier along its itinerary for one batch window."""
        sc = self.scenario
        speed_km_s = sc.speed_kmh / 3600.0
        window_km = speed_km_s * sc.batch_seconds
        for d, plan in enumerate(self.plans):
            budget = window_km
            while plan and budget > 0:
                oid, kind = plan[0]
                order = self.released[oid]
                s_lat, s_lng = (
                    (order.pickup_lat, order.pickup_lng) if kind == PICKUP
                    else (order.delivery_lat, order.delivery_lng)
                )
                dist = haversine_km(self.lat[d], self.lng[d], s_lat, s_lng)
                if dist <= budget:
                    budget -= dist
                    self.km[d] += dist
                    self.lat[d], self.lng[d] = s_lat, s_lng
                    plan.pop(0)
                    self.visits[d].append((oid, kind))
                    if kind == PICKUP:
                        self.onboard[d].append(oid)
                        self.load[d] += order.weight
                    else:
                        self.onboard[d].remove(oid)
                        self.load[d] -= order.weight
                        self.delivered_at[oid] = t + (window_km - budget) / speed_km_s
                else:
                    frac = budget / dist
                    self.lat[d] += frac * (s_lat - self.lat[d])
                    self.lng[d] += frac * (s_lng - self.lng[d])
                    self.km[d] += budget
                    budget = 0.0
