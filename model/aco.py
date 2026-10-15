import random
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np

from model.aco_core import ACOParams, ACOProblem, solve_aco
from model.enviroment import Environment
from model.node import Node


class ACO:
    """
    Ant Colony Optimization (ACO) solver for smart logistics route planning.
    - Uses geodesic distance for spatial pruning in neighborhood search.
    - Uses generalized business cost for heuristic evaluation and route selection.
    Node/UUID objects are translated once into integer-indexed numpy matrices; the hot loop
    runs in model.aco_core on plain arrays (see build_problem / to_node_path for process pools).
    """
    def __init__(
        self,
        env: Environment,
        depot: Optional[Node] = None, # Optional central logistics depot (fixed start/return origin)
        num_ants: int = 30,          # Population size (number of ants)
        max_iter: int = 100,         # Maximum iteration epochs
        alpha: float = 1.0,          # Pheromone importance factor
        beta: float = 2.0,           # Heuristic cost importance factor
        rho: float = 0.1,            # Global pheromone evaporation rate
        q: float = 100.0,            # Pheromone deposit intensity constant
        initial_radius: float = 30.0,# Initial neighborhood search radius (km)
        radius_step: float = 10.0,   # Adaptive radius expansion step (km)
        max_radius: float = 200.0,   # Maximum allowable search radius (km)
        initial_path: Optional[List[Node]] = None,  # Warm-start closed-loop path (e.g. previous route + insertions)
    ):
        self.env = env
        self.depot = depot
        self.params = ACOParams(
            num_ants=num_ants,
            max_iter=max_iter,
            alpha=alpha,
            beta=beta,
            rho=rho,
            q=q,
            initial_radius=initial_radius,
            radius_step=radius_step,
            max_radius=max_radius,
        )
        self.initial_path = initial_path
        self.index_of: Dict[UUID, int] = {n.id: i for i, n in enumerate(env.nodes)}

        # Global best solution trackers
        self.best_path: List[Node] = []
        self.best_cost: float = float("inf")

    def _eligible_start_indices(self) -> List[int]:
        """Fixed depot if specified, otherwise nodes without prerequisites (or any node as last resort)."""
        if self.depot:
            return [self.index_of[self.depot.id]]
        starts = [i for i, node in enumerate(self.env.nodes) if not node.prerequisites]
        return starts or list(range(len(self.env.nodes)))

    def _warm_start_indices(self, eligible_starts: List[int]) -> Optional[np.ndarray]:
        """Validate the warm-start path: closed loop, eligible origin, covering exactly the node set."""
        path = self.initial_path
        if not path or len(path) < 2 or path[-1].id != path[0].id:
            return None
        if {n.id for n in path} != set(self.index_of):
            return None
        idx = np.array([self.index_of[n.id] for n in path], dtype=np.int64)
        if idx[0] not in eligible_starts:
            return None
        return idx

    def build_problem(self, seed: Optional[int] = None) -> ACOProblem:
        """Translate the Node/UUID environment into a picklable, integer-indexed ACOProblem."""
        nodes = self.env.nodes
        n = len(nodes)
        dist = np.array([[self.env.get_distance(u, v) for v in nodes] for u in nodes], dtype=float)
        cost = np.array([[self.env.get_cost(u, v) for v in nodes] for u in nodes], dtype=float)
        pheromone = np.array([[self.env.get_pheromone(u, v) for v in nodes] for u in nodes], dtype=float)

        # Precedence: a prerequisite outside this environment can never be visited, so it stays unmet forever
        prereq_count = np.zeros(n, dtype=np.int64)
        dependents = np.zeros((n, n), dtype=np.int64)
        for j, node in enumerate(nodes):
            for pre_id in node.prerequisites:
                prereq_count[j] += 1
                if pre_id in self.index_of:
                    dependents[self.index_of[pre_id], j] += 1

        eligible = self._eligible_start_indices()
        return ACOProblem(
            dist=dist,
            cost=cost,
            pheromone=pheromone,
            prereq_count=prereq_count,
            dependents=dependents,
            eligible_starts=np.array(eligible, dtype=np.int64),
            params=self.params,
            initial_path=self._warm_start_indices(eligible),
            seed=seed if seed is not None else random.getrandbits(63),
        )

    def to_node_path(self, path_indices: List[int]) -> List[Node]:
        return [self.env.nodes[i] for i in path_indices]

    def run(self) -> Tuple[List[Node], float]:
        """
        Execute main ACO optimization loop.
        :return: (best_route_nodes, best_total_cost)
        """
        result = solve_aco(self.build_problem())

        # Write back learned trails so the Environment remains the owner of the pheromone network
        nodes = self.env.nodes
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i != j:
                    self.env.pheromone_matrix[(u.id, v.id)] = float(result.pheromone[i, j])

        self.best_path = self.to_node_path(result.best_path)
        self.best_cost = result.best_cost
        return self.best_path, self.best_cost
