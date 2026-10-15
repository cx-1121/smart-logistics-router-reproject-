from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Klang Valley bounding box (KLCC, Bukit Bintang, Petaling Jaya, Subang), same as benchmark_concurrency.py
KL_LNG_RANGE = (101.55, 101.75)
KL_LAT_RANGE = (3.05, 3.25)


@dataclass(frozen=True)
class Order:
    id: int
    pickup_lat: float
    pickup_lng: float
    delivery_lat: float
    delivery_lng: float
    weight: float        # kg
    release_batch: int   # Batch index at which the order becomes visible to the dispatcher


@dataclass(frozen=True)
class Courier:
    id: int
    lat: float
    lng: float
    capacity: float      # kg


@dataclass(frozen=True)
class Scenario:
    couriers: Tuple[Courier, ...]
    orders: Tuple[Order, ...]
    batch_seconds: float = 30.0
    speed_kmh: float = 60.0
    max_orders_held: int = 6   # Orders on board plus orders assigned but not yet picked up


def generate_kl_scenario(
    num_couriers: int = 50,
    num_orders: int = 500,
    orders_per_batch: int = 50,
    seed: int = 42,
    capacity_range: Tuple[float, float] = (20.0, 30.0),
    weight_range: Tuple[float, float] = (1.0, 5.0),
    delivery_offset_deg: float = 0.02,
    batch_seconds: float = 30.0,
    speed_kmh: float = 60.0,
    max_orders_held: int = 6,
) -> Scenario:
    """
    Seeded Klang Valley scenario: couriers and pickups uniform over the bounding box, each delivery
    within +-delivery_offset_deg of its pickup, and orders released orders_per_batch at a time.
    """
    rng = np.random.default_rng(seed)
    couriers = tuple(
        Courier(
            id=i,
            lat=float(rng.uniform(*KL_LAT_RANGE)),
            lng=float(rng.uniform(*KL_LNG_RANGE)),
            capacity=float(np.round(rng.uniform(*capacity_range), 1)),
        )
        for i in range(num_couriers)
    )
    orders = []
    for i in range(num_orders):
        p_lat = float(rng.uniform(*KL_LAT_RANGE))
        p_lng = float(rng.uniform(*KL_LNG_RANGE))
        orders.append(Order(
            id=i,
            pickup_lat=p_lat,
            pickup_lng=p_lng,
            delivery_lat=p_lat + float(rng.uniform(-delivery_offset_deg, delivery_offset_deg)),
            delivery_lng=p_lng + float(rng.uniform(-delivery_offset_deg, delivery_offset_deg)),
            weight=float(np.round(rng.uniform(*weight_range), 1)),
            release_batch=i // orders_per_batch,
        ))
    return Scenario(
        couriers=couriers,
        orders=tuple(orders),
        batch_seconds=batch_seconds,
        speed_kmh=speed_kmh,
        max_orders_held=max_orders_held,
    )
