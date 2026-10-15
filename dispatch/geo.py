import math
from typing import Iterable

import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in kilometers between two coordinates."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = p2 - p1
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlng / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(a, 1.0)))


def haversine_matrix(lat1: np.ndarray, lng1: np.ndarray, lat2: np.ndarray, lng2: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distances in kilometers, shape (len(lat1), len(lat2))."""
    p1 = np.radians(lat1)[:, None]
    p2 = np.radians(lat2)[None, :]
    dlat = p2 - p1
    dlng = np.radians(lng2)[None, :] - np.radians(lng1)[:, None]
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlng / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


class DistanceCache:
    """
    Incremental node-to-node distance matrix for released orders.
    Node 2*o is order o's pickup and node 2*o+1 its delivery, so order ids index the matrix directly.
    Only rows/columns of newly released orders are computed on each call, so the per-batch cost
    is proportional to the new orders, not to everything seen so far.
    """
    def __init__(self, initial_orders: int = 512):
        self._alloc(2 * initial_orders)

    def _alloc(self, n_nodes: int):
        self.lat = np.zeros(n_nodes)
        self.lng = np.zeros(n_nodes)
        self.known = np.zeros(n_nodes, dtype=bool)
        self.node_dist = np.full((n_nodes, n_nodes), np.inf)

    def _grow(self, n_nodes: int):
        old_lat, old_lng, old_known, old_dist = self.lat, self.lng, self.known, self.node_dist
        n_old = len(old_lat)
        self._alloc(n_nodes)
        self.lat[:n_old], self.lng[:n_old], self.known[:n_old] = old_lat, old_lng, old_known
        self.node_dist[:n_old, :n_old] = old_dist

    def add_orders(self, orders: Iterable) -> int:
        """Register pickup/delivery nodes of orders not seen before. Returns the number of new orders."""
        new = [o for o in orders if 2 * o.id + 1 >= len(self.known) or not self.known[2 * o.id]]
        if not new:
            return 0
        top = 2 * max(o.id for o in new) + 2
        if top > len(self.known):
            self._grow(max(top, 2 * len(self.known)))

        new_idx = np.empty(2 * len(new), dtype=np.int64)
        for i, o in enumerate(new):
            p, d = 2 * o.id, 2 * o.id + 1
            self.lat[p], self.lng[p] = o.pickup_lat, o.pickup_lng
            self.lat[d], self.lng[d] = o.delivery_lat, o.delivery_lng
            new_idx[2 * i], new_idx[2 * i + 1] = p, d
        self.known[new_idx] = True

        known_idx = np.flatnonzero(self.known)
        block = haversine_matrix(self.lat[new_idx], self.lng[new_idx], self.lat[known_idx], self.lng[known_idx])
        self.node_dist[np.ix_(new_idx, known_idx)] = block
        self.node_dist[np.ix_(known_idx, new_idx)] = block.T
        return len(new)

    def courier_rows(self, courier_lat: np.ndarray, courier_lng: np.ndarray) -> np.ndarray:
        """Distances from each courier position to every known node (inf for unknown nodes)."""
        out = np.full((len(courier_lat), len(self.known)), np.inf)
        known_idx = np.flatnonzero(self.known)
        out[:, known_idx] = haversine_matrix(courier_lat, courier_lng, self.lat[known_idx], self.lng[known_idx])
        return out
