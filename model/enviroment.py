import logging
import math
import random
from typing import Dict, List, Optional, Tuple
from uuid import UUID
from model.node import Node
from services.osrm_client import OSRMClient

logger = logging.getLogger(__name__)


class Environment:
    """
    Logistics environment: manages node topology, precomputed distance/cost caches,
    and pheromone networks. Delegates external routing queries to an injected routing_client.
    """
    def __init__(
        self,
        nodes: List[Node],
        routing_client: Optional[OSRMClient] = None,
        init_pheromone: float = 1.0,
    ):
        if not nodes:
            raise ValueError("Environment must have at least one node.")

        self.nodes = nodes
        self.node_map: Dict[UUID, Node] = {n.id: n for n in nodes}
        self.routing_client = routing_client
        self.init_pheromone = init_pheromone

        # Internal cache matrices for O(1) high-frequency distance and cost queries
        self._distance_cache: Dict[Tuple[UUID, UUID], float] = {}
        self._cost_cache: Dict[Tuple[UUID, UUID], float] = {}

        # Global pheromone matrix: (from_node_id, to_node_id) -> intensity
        self.pheromone_matrix: Dict[Tuple[UUID, UUID], float] = {}

        self._init_environment()

    def _init_environment(self):
        """
        Initialize the environment:
        1. Attempt matrix population using the injected routing client.
        2. Fall back to local Haversine geodesic calculations if unavailable.
        3. Initialize pheromone network.
        """
        matrix_loaded = False
        if self.routing_client and len(self.nodes) > 1:
            matrix_result = self.routing_client.get_matrix(self.nodes)
            if matrix_result is not None:
                self._distance_cache, self._cost_cache = matrix_result
                matrix_loaded = True

        if not matrix_loaded:
            logger.info("Using local Haversine calculations for distance and cost matrices.")
            self._compute_haversine_matrix()

        self._init_pheromones()

    def _compute_haversine_matrix(self):
        """Precompute distance and generalized cost matrices using local Haversine formula."""
        for n1 in self.nodes:
            for n2 in self.nodes:
                if n1.id != n2.id:
                    dist = self._calc_haversine_distance(n1, n2)
                    self._distance_cache[(n1.id, n2.id)] = dist
                    self._cost_cache[(n1.id, n2.id)] = self._calc_cost(n1, n2, dist)

    def _init_pheromones(self):
        """Initialize pheromone trail for all directed edges."""
        for n1 in self.nodes:
            for n2 in self.nodes:
                if n1.id != n2.id:
                    self.pheromone_matrix[(n1.id, n2.id)] = self.init_pheromone

    def _calc_haversine_distance(self, n1: Node, n2: Node) -> float:
        """
        Calculate great-circle distance between two geographic coordinates in kilometers.
        """
        R = 6371.0  # Earth's mean radius in kilometers
        dlat = math.radians(n2.Lat - n1.Lat)
        dlng = math.radians(n2.Lng - n1.Lng)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(n1.Lat))
            * math.cos(math.radians(n2.Lat))
            * math.sin(dlng / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def _calc_cost(self, n1: Node, n2: Node, distance: float) -> float:
        """
        Calculate baseline generalized cost considering distance, congestion, and toll multipliers.
        """
        time_factor = 1.0
        toll_cost = 0.0
        cost = distance * time_factor + toll_cost
        return max(cost, 0.0001)

    # -------------------------------------------------------------
    # Query APIs
    # -------------------------------------------------------------
    def get_distance(self, n1: Node, n2: Node) -> float:
        """Retrieve physical distance (used for spatial neighborhood search)."""
        if n1.id == n2.id:
            return 0.0
        # Explicit lookup: dict.get(key, default) would eagerly evaluate the Haversine fallback on every call
        cached = self._distance_cache.get((n1.id, n2.id))
        if cached is not None:
            return cached
        return self._calc_haversine_distance(n1, n2)

    def get_cost(self, n1: Node, n2: Node) -> float:
        """Retrieve generalized cost (used for heuristic evaluation eta = 1/cost)."""
        if n1.id == n2.id:
            return 0.0
        cached = self._cost_cache.get((n1.id, n2.id))
        if cached is not None:
            return cached
        return self._calc_cost(n1, n2, self.get_distance(n1, n2))

    def get_random_node(self) -> Node:
        """Randomly select a node to serve as an ant's initial origin."""
        return random.choice(self.nodes)

    # -------------------------------------------------------------
    # Pheromone Management APIs
    # -------------------------------------------------------------
    def get_pheromone(self, n1: Node, n2: Node) -> float:
        """Retrieve current pheromone intensity on directed edge (n1, n2)."""
        return self.pheromone_matrix.get((n1.id, n2.id), self.init_pheromone)

    def evaporate(self, rho: float):
        """
        Global pheromone evaporation: tau = (1 - rho) * tau
        :param rho: Evaporation coefficient in range [0, 1]
        """
        for edge in self.pheromone_matrix:
            self.pheromone_matrix[edge] *= (1.0 - rho)

    def release_pheromone(self, path: List[Node], total_cost: float, q: float = 100.0):
        """
        Deposit pheromone along completed circuit (Ant-Cycle model):
        Lower total_cost results in higher pheromone increment (delta = Q / total_cost).
        """
        if total_cost <= 0 or len(path) < 2:
            return

        delta_tau = q / total_cost
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            key = (u.id, v.id)
            self.pheromone_matrix[key] = self.pheromone_matrix.get(key, 0.0) + delta_tau
