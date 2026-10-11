import random
from typing import List, Optional, Tuple
from model.ant import Ant
from model.enviroment import Environment
from model.node import Node


class ACO:
    """
    Ant Colony Optimization (ACO) solver for smart logistics route planning.
    - Uses geodesic distance for spatial pruning in neighborhood search.
    - Uses generalized business cost for heuristic evaluation and route selection.
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
    ):
        self.env = env
        self.depot = depot
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.initial_radius = initial_radius
        self.radius_step = radius_step
        self.max_radius = max_radius

        # Global best solution trackers
        self.best_path: List[Node] = []
        self.best_cost: float = float("inf")

    def _select_next_node(self, current: Node, candidates: List[Node]) -> Node:
        """
        Select next destination node using roulette wheel selection
        based on pheromone intensity (tau) and heuristic cost (eta = 1/cost).
        """
        if len(candidates) == 1:
            return candidates[0]

        weights: List[float] = []
        for candidate in candidates:
            # 1. Pheromone trail component
            tau = self.env.get_pheromone(current, candidate) ** self.alpha

            # 2. Heuristic component (lower cost results in higher attractiveness)
            cost = self.env.get_cost(current, candidate)
            eta = (1.0 / cost) ** self.beta

            weights.append(tau * eta)

        total_weight = sum(weights)
        if total_weight <= 0:
            return random.choice(candidates)

        # Roulette wheel sampling
        pick = random.uniform(0, total_weight)
        current_sum = 0.0
        for i, weight in enumerate(weights):
            current_sum += weight
            if current_sum >= pick:
                return candidates[i]

        return candidates[-1]

    def run(self) -> Tuple[List[Node], float]:
        """
        Execute main ACO optimization loop.
        :return: (best_route_nodes, best_total_cost)
        """
        total_nodes_count = len(self.env.nodes)

        # Determine eligible starting nodes: fixed depot if specified, or nodes without prerequisites
        if self.depot:
            eligible_starts = [self.depot]
        else:
            eligible_starts = [node for node in self.env.nodes if not node.prerequisites]
            if not eligible_starts:
                eligible_starts = self.env.nodes

        for iteration in range(self.max_iter):
            # 1. Spawn num_ants per generation at eligible origin nodes
            ants = [
                Ant(start_node=random.choice(eligible_starts))
                for _ in range(self.num_ants)
            ]

            completed_ants: List[Ant] = []

            # 2. Path construction for each ant
            for ant in ants:
                while True:
                    available = ant.get_available_nodes(
                        all_nodes=self.env.nodes,
                        get_distance_func=self.env.get_distance,
                        initial_radius=self.initial_radius,
                        radius_step=self.radius_step,
                        max_radius=self.max_radius,
                    )

                    if not available:
                        break  # No more candidate nodes available

                    # Select next node via roulette wheel based on generalized cost
                    next_node = self._select_next_node(ant.current_node, available)
                    step_cost = self.env.get_cost(ant.current_node, next_node)
                    ant.move_to(next_node, step_cost)

                # Check if all customer nodes were successfully serviced
                if len(ant.visited_ids) == total_nodes_count:
                    # Circuit closing: return from final customer back to origin depot
                    return_cost = self.env.get_cost(ant.current_node, ant.start_node)
                    ant.move_to(ant.start_node, return_cost)
                    completed_ants.append(ant)

            # 3. Update global best solution with current generation achievements
            for ant in completed_ants:
                if ant.total_cost < self.best_cost:
                    self.best_cost = ant.total_cost
                    self.best_path = list(ant.path)

            # 4. Global pheromone evaporation
            self.env.evaporate(self.rho)

            # 5. Pheromone reinforcement from successful ants
            for ant in completed_ants:
                self.env.release_pheromone(ant.path, ant.total_cost, self.q)

            # Elitist strategy: provide additional reinforcement to global best route
            if self.best_path:
                self.env.release_pheromone(self.best_path, self.best_cost, self.q * 1.5)

        return self.best_path, self.best_cost
