import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Dict, Optional, Set, Tuple, Any
from uuid import UUID
import h3

from model.driver import Driver
from model.task import DeliveryTask, TaskStatus
from model.enviroment import Environment
from model.aco import ACO
from model.aco_core import ACOProblem, solve_aco_path
from services.spatial_index import SpatialIndexer
from services.route_insertion import RouteInsertion


class BatchSummary:
    """Detailed summary statistics for a completed 30-second dispatch batch."""
    def __init__(self, batch_id: int, timestamp_sec: int):
        self.batch_id = batch_id
        self.timestamp_sec = timestamp_sec
        self.broadcast_count: int = 0
        self.claimed_count: int = 0
        self.unassigned_count: int = 0
        self.aco_latencies: Dict[str, float] = {}
        self.batch_duration_ms: float = 0.0


class BatchDispatcher:
    """
    Industrial 30-second rolling batch dispatch engine with multi-threading acceleration.
    Integrates:
    - Scenario 1: Fleet-level concurrent multi-threaded ACO route optimization.
    - Scenario 3: Multi-threaded concurrent order claiming race condition control.
    - Strict non-re-allocation registry and rolling timeline stepping.
    """
    def __init__(
        self,
        drivers: List[Driver],
        spatial_indexer: Optional[SpatialIndexer] = None,
        batch_interval_seconds: int = 30,
        max_h3_steps: int = 5,
        enable_concurrent_claims: bool = True,
        enable_concurrent_routing: bool = True,
        max_workers: int = 4,
        routing_client: Optional[Any] = None,
        aco_max_iter: int = 25,
        reroute_max_iter: int = 10,
        routing_backend: str = "thread"
    ):
        """
        :param routing_backend: "thread" solves ACO in a thread pool (GIL-bound, no startup cost);
            "process" solves ACO in a persistent process pool (true multi-core; call shutdown() or use
            the dispatcher as a context manager to stop the workers).
        """
        if routing_backend not in ("thread", "process"):
            raise ValueError(f"Unknown routing_backend '{routing_backend}', expected 'thread' or 'process'")
        self.drivers = drivers
        self.indexer = spatial_indexer or SpatialIndexer(resolution=8)
        self.batch_interval_seconds = batch_interval_seconds
        self.max_h3_steps = max_h3_steps
        self.enable_concurrent_claims = enable_concurrent_claims
        self.enable_concurrent_routing = enable_concurrent_routing
        self.max_workers = max_workers
        self.routing_client = routing_client
        self.aco_max_iter = aco_max_iter          # Iterations for a courier's first route (insertion-seeded)
        self.reroute_max_iter = reroute_max_iter  # Iterations for in-transit re-routing (warm-started from previous route)
        self.routing_backend = routing_backend
        self._process_pool: Optional[ProcessPoolExecutor] = None

        # Internal timeline clock and batch progression counter
        self.current_batch_id: int = 0
        self.current_time_sec: int = 0

        # Global pool of orders pending allocation
        # Strictly contains only tasks with status PENDING or OFFERED (never LOCKED)
        self.unallocated_order_pool: List[DeliveryTask] = []

        # Historical registry of claimed order IDs to guarantee zero re-allocation
        self.claimed_order_registry: Set[UUID] = set()

    def is_task_allocated(self, task: DeliveryTask) -> bool:
        """
        Check whether an order has already been claimed or locked.
        Guarantees already allocated tasks cannot be re-assigned.
        """
        return (
            task.id in self.claimed_order_registry or
            task.status in (TaskStatus.LOCKED, TaskStatus.COMPLETED)
        )

    def add_orders(self, orders: List[DeliveryTask]):
        """
        Ingest newly arrived orders from streaming upstream into the unallocated pool.
        Strictly excludes any order that has already been claimed, locked, or completed.
        """
        for order in orders:
            # Ironclad guard: ignore orders that have already been allocated
            if self.is_task_allocated(order):
                continue

            if order.status in (TaskStatus.PENDING, TaskStatus.OFFERED) and order not in self.unallocated_order_pool:
                self.unallocated_order_pool.append(order)
                self.indexer.add_node(order.pickup)

    def _courier_claim_worker(self, driver: Driver) -> List[DeliveryTask]:
        """Worker function for concurrent multi-threaded order claiming (Scenario 3)."""
        claimed_tasks = []
        if not driver.offered_tasks:
            return claimed_tasks

        candidate_offers = list(driver.offered_tasks)
        for order in candidate_offers:
            # Atomic mutex check via order._lock
            if driver.accept_task(order):
                claimed_tasks.append(order)
        return claimed_tasks

    def _prepare_driver_route(self, driver: Driver) -> Tuple[Driver, ACO, ACOProblem, float]:
        """
        Build one courier's routing problem (Scenario 1, I/O phase: OSRM matrix + insertion seed).
        In-transit couriers are warm-started: the previous itinerary keeps its order, newly stacked
        tasks are placed by cheapest insertion, and ACO refines that seed with fewer iterations.
        Returns the picklable index-based ACOProblem so the solve can run in a worker process.
        """
        t_prep_start = time.perf_counter()
        routing_nodes = driver.get_pending_routing_nodes()
        local_env = Environment(nodes=routing_nodes, routing_client=self.routing_client, init_pheromone=1.0)

        is_reroute = bool(driver.assigned_route)
        seed_path = RouteInsertion(local_env.get_cost).build_seed_route(
            origin=driver.current_location,
            routing_nodes=routing_nodes,
            previous_route=driver.assigned_route,
            tasks=driver.active_tasks,
        )
        aco = ACO(
            env=local_env,
            depot=driver.current_location,
            num_ants=25,
            max_iter=self.reroute_max_iter if (is_reroute and seed_path) else self.aco_max_iter,
            initial_radius=10.0,
            radius_step=5.0,
            max_radius=50.0,
            initial_path=seed_path
        )
        problem = aco.build_problem()
        return driver, aco, problem, (time.perf_counter() - t_prep_start) * 1000

    def _get_process_pool(self) -> ProcessPoolExecutor:
        """Lazily start the persistent worker-process pool (reused across batches)."""
        if self._process_pool is None:
            self._process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        return self._process_pool

    def warm_up(self):
        """Spawn all routing worker processes ahead of the first batch (process startup is ~100s of ms)."""
        if self.routing_backend == "process":
            list(self._get_process_pool().map(abs, range(self.max_workers)))

    def shutdown(self):
        """Stop the routing worker processes, if any were started."""
        if self._process_pool is not None:
            self._process_pool.shutdown()
            self._process_pool = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()

    def dispatch_batch(self, batch_id: int, current_time_sec: int) -> BatchSummary:
        """
        Execute one 30-second rolling dispatch batch:
        1. Query only unallocated pending orders (already claimed orders are strictly excluded).
        2. Broadcast offers to candidate proximate drivers via H3 grid (in-transit drivers eligible).
        3. Couriers independently accept and lock orders via multi-threaded atomic mutex claiming (Scenario 3).
        4. Claimed orders are registered and permanently purged from the unallocated pool.
        5. Multi-threaded ACO route optimization runs once per courier whose task set changed (Scenario 1).
        """
        t_batch_start = time.perf_counter()
        summary = self.claim_batch(batch_id=batch_id, current_time_sec=current_time_sec)
        self.route_pending_drivers(summary=summary)
        summary.batch_duration_ms = (time.perf_counter() - t_batch_start) * 1000
        return summary

    def claim_batch(self, batch_id: int, current_time_sec: int) -> BatchSummary:
        """
        Claim phase only (steps 1-4 of dispatch_batch): broadcast, atomic claiming and registry purge.
        Couriers that claim orders are flagged with needs_reroute; routing is deferred so that
        multi-zone callers can claim across all zones first and solve each courier exactly once.
        """
        summary = BatchSummary(batch_id=batch_id, timestamp_sec=current_time_sec)
        t_batch_start = time.perf_counter()

        # Strict filtration: purge any orders that are already LOCKED, COMPLETED, or CANCELLED
        active_candidate_orders = [
            order for order in self.unallocated_order_pool
            if order.status in (TaskStatus.PENDING, TaskStatus.OFFERED)
            and not self.is_task_allocated(order)
        ]

        if not active_candidate_orders:
            summary.batch_duration_ms = (time.perf_counter() - t_batch_start) * 1000
            summary.unassigned_count = len(self.unallocated_order_pool)
            return summary

        # ---------------------------------------------------------------------
        # Step 1: Spatial Broadcast Offering via Uber H3 Grid
        # ---------------------------------------------------------------------
        for order in active_candidate_orders:
            order_cell = self.indexer.get_node_cell(order.pickup)
            offered_to_any = False

            for driver in self.drivers:
                # In-transit couriers (BUSY) with spare payload capacity remain eligible
                if not driver.is_available:
                    continue

                driver_cell = self.indexer.get_node_cell(driver.current_location)
                steps = h3.grid_distance(driver_cell, order_cell)

                if steps <= self.max_h3_steps:
                    if driver.receive_offer(order):
                        offered_to_any = True

            if offered_to_any:
                summary.broadcast_count += 1

        # ---------------------------------------------------------------------
        # Step 2: Multi-Threaded Courier Decision & Atomic Claim Locking (Scenario 3)
        # ---------------------------------------------------------------------
        claimed_in_this_batch: List[DeliveryTask] = []
        candidate_drivers = [d for d in self.drivers if d.offered_tasks]

        if self.enable_concurrent_claims and len(candidate_drivers) > 1:
            workers = min(len(candidate_drivers), self.max_workers)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                claim_results = executor.map(self._courier_claim_worker, candidate_drivers)
            for courier_claimed in claim_results:
                for order in courier_claimed:
                    claimed_in_this_batch.append(order)
                    summary.claimed_count += 1
        else:
            for driver in candidate_drivers:
                courier_claimed = self._courier_claim_worker(driver)
                for order in courier_claimed:
                    claimed_in_this_batch.append(order)
                    summary.claimed_count += 1

        # ---------------------------------------------------------------------
        # Step 3: Register and Permanently Expunge Claimed Orders (Zero Re-allocation)
        # ---------------------------------------------------------------------
        for claimed_order in claimed_in_this_batch:
            self.claimed_order_registry.add(claimed_order.id)
            if claimed_order in self.unallocated_order_pool:
                self.unallocated_order_pool.remove(claimed_order)

            # Purge newly claimed order from any competing drivers' pending offer queues
            for driver in self.drivers:
                driver.reject_task(claimed_order)

        summary.unassigned_count = len(self.unallocated_order_pool)
        summary.batch_duration_ms = (time.perf_counter() - t_batch_start) * 1000
        return summary

    def route_pending_drivers(
        self,
        summary: Optional[BatchSummary] = None,
        drivers: Optional[List[Driver]] = None
    ) -> Dict[str, float]:
        """
        Step 5: Multi-Threaded ACO Route Optimization (Scenario 1).
        Solves each courier whose active task set changed since its last route (needs_reroute)
        exactly once; couriers with unchanged tasks keep their current itinerary.
        Returns courier name -> ACO latency (ms), also recorded into summary when given.
        """
        candidates = drivers if drivers is not None else self.drivers
        # Dedupe by courier id: the same courier may be shared by several zones
        pending: Dict[UUID, Driver] = {}
        for d in candidates:
            if d.needs_reroute and d.active_tasks:
                pending.setdefault(d.id, d)
        pending_drivers = list(pending.values())
        concurrent = self.enable_concurrent_routing and len(pending_drivers) > 1

        # Phase A (I/O bound): OSRM matrices + insertion seeds; threads overlap the network waits
        if concurrent:
            with ThreadPoolExecutor(max_workers=min(len(pending_drivers), self.max_workers)) as executor:
                prepared = list(executor.map(self._prepare_driver_route, pending_drivers))
        else:
            prepared = [self._prepare_driver_route(d) for d in pending_drivers]

        # Phase B (CPU bound): ACO solves; worker processes bypass the GIL, threads do not
        problems = [problem for _, _, problem, _ in prepared]
        if concurrent and self.routing_backend == "process":
            solved = list(self._get_process_pool().map(solve_aco_path, problems))
        elif concurrent:
            with ThreadPoolExecutor(max_workers=min(len(problems), self.max_workers)) as executor:
                solved = list(executor.map(solve_aco_path, problems))
        else:
            solved = [solve_aco_path(problem) for problem in problems]

        latencies: Dict[str, float] = {}
        for (driver, aco, _, prep_ms), (path_indices, best_cost, solve_ms) in zip(prepared, solved):
            driver.assign_route(aco.to_node_path(path_indices), best_cost)
            latencies[driver.name] = prep_ms + solve_ms

        if summary is not None:
            summary.aco_latencies.update(latencies)
        return latencies

    def step_batch(self, incoming_orders: Optional[List[DeliveryTask]] = None) -> BatchSummary:
        """
        Advance the timeline by one batch window (default 30 seconds):
        1. Ingest newly arrived orders into the pending pool.
        2. Increment batch sequence number and timeline timestamp.
        3. Trigger dispatch_batch and return summary.
        """
        if incoming_orders:
            self.add_orders(incoming_orders)

        self.current_batch_id += 1
        self.current_time_sec = (self.current_batch_id - 1) * self.batch_interval_seconds

        return self.dispatch_batch(
            batch_id=self.current_batch_id,
            current_time_sec=self.current_time_sec
        )
