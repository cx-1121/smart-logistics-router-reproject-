from typing import Callable, List, Optional, Set, Tuple
from uuid import UUID
from model.node import Node
from model.task import DeliveryTask

CostFunc = Callable[[Node, Node], float]


class RouteInsertion:
    """
    Cheapest-insertion heuristic for in-transit order stacking.
    Operates on a closed-loop itinerary: origin -> stops... -> origin.
    A "slot" k means inserting between the k-th node of [origin] + stops and the next one.
    Runs in O(n) per insertion, so it is cheap enough to evaluate on every claim.
    """
    def __init__(self, cost_func: CostFunc):
        self.cost = cost_func

    def route_cost(self, origin: Node, stops: List[Node]) -> float:
        """Total closed-loop cost of origin -> stops -> origin."""
        path = [origin] + stops + [origin]
        return sum(self.cost(path[i], path[i + 1]) for i in range(len(path) - 1))

    def _slot_ends(self, origin: Node, stops: List[Node], k: int) -> Tuple[Node, Node]:
        prev_node = origin if k == 0 else stops[k - 1]
        next_node = stops[k] if k < len(stops) else origin
        return prev_node, next_node

    def _single_delta(self, origin: Node, stops: List[Node], k: int, node: Node) -> float:
        a, b = self._slot_ends(origin, stops, k)
        return self.cost(a, node) + self.cost(node, b) - self.cost(a, b)

    def insert_node(
        self,
        origin: Node,
        stops: List[Node],
        node: Node,
        min_slot: int = 0,
    ) -> Tuple[List[Node], float]:
        """Insert a single node at its cheapest slot >= min_slot. Returns (new_stops, delta_cost)."""
        best_k, best_delta = min_slot, float("inf")
        for k in range(min_slot, len(stops) + 1):
            delta = self._single_delta(origin, stops, k, node)
            if delta < best_delta:
                best_k, best_delta = k, delta
        return stops[:best_k] + [node] + stops[best_k:], best_delta

    def insert_pair(
        self,
        origin: Node,
        stops: List[Node],
        pickup: Node,
        delivery: Node,
    ) -> Tuple[List[Node], float]:
        """
        Insert a pickup-delivery pair at the cheapest feasible slots (pickup strictly before delivery).
        Returns (new_stops, delta_cost).
        """
        m = len(stops)
        best: Tuple[float, int, int] = (float("inf"), 0, 0)
        best_pickup_k, best_pickup_delta = -1, float("inf")  # Best pickup slot strictly before current j

        for j in range(m + 1):
            # Case 1: pickup and delivery placed back-to-back in the same slot j
            a, b = self._slot_ends(origin, stops, j)
            adjacent = self.cost(a, pickup) + self.cost(pickup, delivery) + self.cost(delivery, b) - self.cost(a, b)
            if adjacent < best[0]:
                best = (adjacent, j, j)

            # Case 2: pickup in an earlier slot i < j, delivery in slot j (edges are independent)
            if best_pickup_k >= 0:
                split = best_pickup_delta + self._single_delta(origin, stops, j, delivery)
                if split < best[0]:
                    best = (split, best_pickup_k, j)

            pickup_delta = self._single_delta(origin, stops, j, pickup)
            if pickup_delta < best_pickup_delta:
                best_pickup_k, best_pickup_delta = j, pickup_delta

        delta, i, j = best
        if i == j:
            return stops[:i] + [pickup, delivery] + stops[i:], delta
        return stops[:i] + [pickup] + stops[i:j] + [delivery] + stops[j:], delta

    @staticmethod
    def respects_precedence(stops: List[Node], required_ids: Set[UUID]) -> bool:
        """Every prerequisite that still needs a visit must appear earlier in the sequence."""
        seen: Set[UUID] = set()
        for node in stops:
            for pre_id in node.prerequisites:
                if pre_id in required_ids and pre_id not in seen:
                    return False
            seen.add(node.id)
        return True

    def build_seed_route(
        self,
        origin: Node,
        routing_nodes: List[Node],
        previous_route: List[Node],
        tasks: List[DeliveryTask],
    ) -> Optional[List[Node]]:
        """
        Build a warm-start closed-loop path for ACO:
        1. Keep the still-pending stops of the previous itinerary in their existing order.
        2. Insert every newly stacked task's stops at their cheapest feasible slots.
        Returns [origin, ...stops, origin], or None if a consistent seed cannot be built.
        """
        node_by_id = {n.id: n for n in routing_nodes if n.id != origin.id}
        required_ids = set(node_by_id)

        # Step 1: retain previous itinerary order for stops that are still pending
        stops: List[Node] = []
        placed: Set[UUID] = set()
        for n in previous_route:
            if n.id in node_by_id and n.id not in placed:
                stops.append(node_by_id[n.id])
                placed.add(n.id)
        if not self.respects_precedence(stops, required_ids):
            stops, placed = [], set()

        # Step 2: cheapest insertion for stops not yet in the sequence
        for task in tasks:
            pickup_pending = task.pickup.id in required_ids
            delivery_pending = task.delivery.id in required_ids
            pickup_missing = pickup_pending and task.pickup.id not in placed
            delivery_missing = delivery_pending and task.delivery.id not in placed

            if pickup_missing and delivery_missing:
                stops, _ = self.insert_pair(origin, stops, node_by_id[task.pickup.id], node_by_id[task.delivery.id])
            elif delivery_missing:
                # Pickup already completed (min slot 0) or already sequenced (must come after it)
                min_slot = 0
                if pickup_pending:
                    min_slot = next(i for i, n in enumerate(stops) if n.id == task.pickup.id) + 1
                stops, _ = self.insert_node(origin, stops, node_by_id[task.delivery.id], min_slot)
            elif pickup_missing:
                return None  # Delivery sequenced without its pickup: inconsistent, let ACO solve cold
            placed.update(n.id for n in stops)

        if len(stops) != len(required_ids):
            return None
        return [origin] + stops + [origin]
