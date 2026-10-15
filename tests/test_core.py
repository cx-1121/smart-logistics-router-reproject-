import sys
from pathlib import Path
import unittest
import h3

# Append project workspace root to sys.path to allow execution directly or via test runners
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.node import Node
from model.ant import Ant
from model.driver import Driver, DriverStatus
from model.task import DeliveryTask, TaskStatus
from model.enviroment import Environment
from model.aco import ACO
from services.spatial_index import SpatialIndexer
from services.batch_dispatcher import BatchDispatcher
from services.route_insertion import RouteInsertion


class TestDriverDomainModel(unittest.TestCase):
    """Verify Driver domain entity state transitions and payload capacity enforcement."""

    def setUp(self):
        self.depot = Node(name="Depot", Lng=121.4737, Lat=31.2304)
        self.driver = Driver(name="TestDriver", current_location=self.depot, capacity=50.0)

    def test_initial_state(self):
        self.assertEqual(self.driver.status, DriverStatus.IDLE)
        self.assertTrue(self.driver.is_available)
        self.assertEqual(self.driver.current_load, 0.0)
        self.assertEqual(self.driver.remaining_capacity, 50.0)
        self.assertIsNone(self.driver.next_stop)

    def test_load_and_unload_cargo(self):
        self.driver.load_cargo(20.0)
        self.assertEqual(self.driver.current_load, 20.0)
        self.assertEqual(self.driver.remaining_capacity, 30.0)

        self.driver.unload_cargo(10.0)
        self.assertEqual(self.driver.current_load, 10.0)
        self.assertEqual(self.driver.remaining_capacity, 40.0)

    def test_overload_prevention(self):
        """Ensure overloading throws a ValueError exception."""
        self.driver.load_cargo(40.0)
        with self.assertRaises(ValueError):
            self.driver.load_cargo(15.0)  # 40.0 + 15.0 exceeds capacity 50.0

    def test_route_assignment_and_lifecycle(self):
        """Verify itinerary assignment updates status to BUSY and clears back to IDLE."""
        stop1 = Node(name="Stop 1", Lng=121.4800, Lat=31.2350)
        stop2 = Node(name="Stop 2", Lng=121.4900, Lat=31.2400)

        self.driver.assign_route([stop1, stop2], route_cost=18.5)
        self.assertEqual(self.driver.status, DriverStatus.BUSY)
        # In-transit couriers with spare payload capacity remain available for new tasks
        self.assertTrue(self.driver.is_available)
        self.assertEqual(self.driver.next_stop.name, "Stop 1")
        self.assertEqual(self.driver.total_route_cost, 18.5)

        # Saturated vehicle payload makes driver unavailable
        self.driver.load_cargo(50.0)
        self.assertFalse(self.driver.is_available)
        self.driver.unload_cargo(50.0)

        self.driver.clear_route()
        self.assertEqual(self.driver.status, DriverStatus.IDLE)
        self.assertTrue(self.driver.is_available)
        self.assertIsNone(self.driver.next_stop)


class TestAntBehaviorAndConstraints(unittest.TestCase):
    """Verify Ant precedence constraints (PDP) and neighborhood search safety."""

    def setUp(self):
        self.origin = Node(name="Origin", Lng=121.4737, Lat=31.2304)
        self.pickup = Node(name="Pickup Point", Lng=121.4800, Lat=31.2350)
        self.delivery = Node(
            name="Delivery Point",
            Lng=121.4900,
            Lat=31.2400,
            prerequisites=[self.pickup.id]  # Delivery depends on prior pickup visit
        )
        self.ant = Ant(start_node=self.origin)

    def dummy_dist(self, n1: Node, n2: Node) -> float:
        return 1.0

    def test_precedence_constraint_filtering(self):
        """Delivery node must NOT be selectable prior to completing the pickup visit."""
        all_nodes = [self.origin, self.pickup, self.delivery]

        # Initially, only origin has been visited; pickup is pending
        available = self.ant.get_available_nodes(
            all_nodes=all_nodes,
            get_distance_func=self.dummy_dist,
            initial_radius=10.0,
            radius_step=5.0,
            max_radius=20.0
        )

        candidate_ids = {node.id for node in available}
        self.assertIn(self.pickup.id, candidate_ids, "Pickup point should be available")
        self.assertNotIn(self.delivery.id, candidate_ids, "Delivery point must be filtered out")

        # Move ant to the pickup stop
        self.ant.move_to(self.pickup, cost=1.0)

        # Once pickup is visited, delivery point should unlock
        available_after = self.ant.get_available_nodes(
            all_nodes=all_nodes,
            get_distance_func=self.dummy_dist,
            initial_radius=10.0,
            radius_step=5.0,
            max_radius=20.0
        )
        candidate_ids_after = {node.id for node in available_after}
        self.assertIn(self.delivery.id, candidate_ids_after, "Delivery point must become available")

    def test_deadlock_free_when_no_candidates(self):
        """Safely return empty list when all candidate nodes are exhausted without infinite loops."""
        all_nodes = [self.origin]  # Origin already visited
        available = self.ant.get_available_nodes(
            all_nodes=all_nodes,
            get_distance_func=self.dummy_dist,
            initial_radius=10.0,
            radius_step=5.0,
            max_radius=20.0
        )
        self.assertEqual(available, [], "Must safely return empty list when candidates exhausted")


class TestSpatialIndexerH3(unittest.TestCase):
    """Verify Uber H3 spatial indexing and concentric ring pruning."""

    def setUp(self):
        self.indexer = SpatialIndexer(resolution=8)
        self.center_node = Node(name="Center (KLCC)", Lng=101.7118, Lat=3.1578)
        self.near_node = Node(name="Near (500m)", Lng=101.7130, Lat=3.1590)
        self.far_node = Node(name="Far (Rawang 25km)", Lng=101.5750, Lat=3.3210)

    def test_spatial_pruning(self):
        self.indexer.build_index([self.near_node, self.far_node])

        # Query ring k=1: nearby node should be captured, distant node must be pruned
        nearby = self.indexer.find_nearby_nodes(self.center_node, max_k_ring=1)
        nearby_names = [n.name for n in nearby]

        self.assertIn("Near (500m)", nearby_names)
        self.assertNotIn("Far (Rawang 25km)", nearby_names)


class TestACOEndToEnd(unittest.TestCase):
    """Verify ACO solver core business invariants."""

    def test_aco_solves_and_respects_precedence(self):
        depot = Node(name="Depot", Lng=121.4737, Lat=31.2304)
        pickup = Node(name="Pickup", Lng=121.4900, Lat=31.2400)
        delivery = Node(name="Delivery", Lng=121.5100, Lat=31.2500, prerequisites=[pickup.id])
        client = Node(name="Client", Lng=121.4600, Lat=31.2200)

        nodes = [depot, pickup, delivery, client]
        env = Environment(nodes=nodes, routing_client=None, init_pheromone=1.0)

        aco = ACO(
            env=env,
            depot=depot,
            num_ants=15,
            max_iter=15,
            initial_radius=20.0,
            radius_step=10.0,
            max_radius=50.0
        )

        best_path, best_cost = aco.run()

        # 1. Feasible route must be generated
        self.assertTrue(len(best_path) > 0, "Valid route must be produced")
        self.assertLess(best_cost, float("inf"), "Solution cost must be finite")

        # 2. Closed-circuit verification: itinerary must begin and end at depot
        self.assertEqual(best_path[0].name, "Depot")
        self.assertEqual(best_path[-1].name, "Depot")

        # 3. Precedence invariant: pickup visit index must precede delivery visit index
        path_names = [n.name for n in best_path]
        idx_p = path_names.index("Pickup")
        idx_d = path_names.index("Delivery")
        self.assertLess(idx_p, idx_d, f"Pickup index ({idx_p}) must precede delivery index ({idx_d})")


class TestDeliveryTaskDomainModel(unittest.TestCase):
    """Verify DeliveryTask atomic binding, automatic precedence injection, and midpoint metric."""

    def test_automatic_precedence_injection(self):
        p = Node(name="Pickup", Lng=121.4, Lat=31.2)
        d = Node(name="Delivery", Lng=121.5, Lat=31.3)
        # Initially delivery prerequisites is empty
        self.assertNotIn(p.id, d.prerequisites)

        # Assemble DeliveryTask
        task = DeliveryTask(task_name="Order 1", pickup=p, delivery=d, weight=5.0)

        # Verify automatic hook injected pickup.id into delivery prerequisites
        self.assertIn(p.id, task.delivery.prerequisites, "Delivery node must automatically depend on pickup ID")
        self.assertEqual(task.weight, 5.0)

    def test_midpoint_calculation(self):
        p = Node(name="P", Lng=120.0, Lat=30.0)
        d = Node(name="D", Lng=122.0, Lat=32.0)
        task = DeliveryTask(task_name="Order Mid", pickup=p, delivery=d)

        mid_lng, mid_lat = task.midpoint
        self.assertAlmostEqual(mid_lng, 121.0)
        self.assertAlmostEqual(mid_lat, 31.0)


class TestTaskBroadcastAndAtomicLock(unittest.TestCase):
    """Verify task multi-driver broadcasting and atomic locking prevents duplicate claiming."""

    def setUp(self):
        self.depot1 = Node(name="Depot 1", Lng=121.47, Lat=31.23)
        self.depot2 = Node(name="Depot 2", Lng=121.48, Lat=31.24)
        self.driver_a = Driver(name="Driver A", current_location=self.depot1, capacity=30.0)
        self.driver_b = Driver(name="Driver B", current_location=self.depot2, capacity=30.0)

        p = Node(name="Pickup", Lng=121.475, Lat=31.235)
        d = Node(name="Delivery", Lng=121.485, Lat=31.245)
        self.task = DeliveryTask(task_name="Shared Order", pickup=p, delivery=d, weight=10.0)

    def test_broadcast_to_multiple_drivers(self):
        """A single pending task can be simultaneously offered to multiple eligible drivers."""
        res_a = self.driver_a.receive_offer(self.task)
        res_b = self.driver_b.receive_offer(self.task)

        self.assertTrue(res_a)
        self.assertTrue(res_b)
        self.assertIn(self.task, self.driver_a.offered_tasks)
        self.assertIn(self.task, self.driver_b.offered_tasks)
        self.assertEqual(len(self.task.candidate_driver_ids), 2)

    def test_atomic_claim_locks_task_and_blocks_competitor(self):
        """When Driver A claims the task, Driver B's attempt to claim the same task must fail."""
        self.driver_a.receive_offer(self.task)
        self.driver_b.receive_offer(self.task)

        # Driver A accepts first
        accepted_a = self.driver_a.accept_task(self.task)
        self.assertTrue(accepted_a, "First driver should successfully claim and lock the task")
        self.assertEqual(self.task.assigned_driver_id, self.driver_a.id)
        self.assertEqual(self.driver_a.current_load, 10.0)
        self.assertIn(self.task, self.driver_a.active_tasks)

        # Driver B attempts to accept the already-locked task
        accepted_b = self.driver_b.accept_task(self.task)
        self.assertFalse(accepted_b, "Second driver MUST fail to claim an already locked task")
        self.assertEqual(self.driver_b.current_load, 0.0, "Driver B payload must remain unchanged")
        self.assertNotIn(self.task, self.driver_b.active_tasks)

    def test_rejection_leaves_task_open_for_others(self):
        """If Driver A rejects the offer, Driver B can still claim it subsequently."""
        self.driver_a.receive_offer(self.task)
        self.driver_b.receive_offer(self.task)

        # Driver A declines
        self.driver_a.reject_task(self.task)
        self.assertNotIn(self.task, self.driver_a.offered_tasks)

        # Driver B accepts
        accepted_b = self.driver_b.accept_task(self.task)
        self.assertTrue(accepted_b)
        self.assertEqual(self.task.assigned_driver_id, self.driver_b.id)

    def test_in_transit_busy_driver_accepts_new_task(self):
        """In-transit driver (BUSY) with spare capacity must be allowed to accept new tasks."""
        # 1. Driver A accepts initial task and starts transit (becomes BUSY)
        self.driver_a.receive_offer(self.task)
        self.driver_a.accept_task(self.task)
        self.driver_a.assign_route([self.task.pickup, self.task.delivery], route_cost=10.0)
        self.assertEqual(self.driver_a.status, DriverStatus.BUSY)
        self.assertTrue(self.driver_a.is_available, "Busy driver with spare capacity must remain available")

        # 2. A new Task arrives while Driver A is in-transit
        p2 = Node(name="P2", Lng=121.48, Lat=31.23)
        d2 = Node(name="D2", Lng=121.49, Lat=31.24)
        new_task = DeliveryTask(task_name="New In-Transit Order", pickup=p2, delivery=d2, weight=5.0)

        # 3. Driver A receives offer and successfully accepts while still BUSY
        offered = self.driver_a.receive_offer(new_task)
        self.assertTrue(offered)
        claimed = self.driver_a.accept_task(new_task)
        self.assertTrue(claimed, "Driver in transit must successfully claim additional concurrent task")

        # 4. Verify updated load and routing nodes collection
        self.assertEqual(self.driver_a.current_load, 15.0)  # 10.0 + 5.0
        self.assertEqual(len(self.driver_a.active_tasks), 2)

        pending_nodes = self.driver_a.get_pending_routing_nodes()
        self.assertIn(self.task.pickup, pending_nodes)
        self.assertIn(self.task.delivery, pending_nodes)
        self.assertIn(p2, pending_nodes)
        self.assertIn(d2, pending_nodes)


class TestBatchDispatcher(unittest.TestCase):
    """Verify 30-second rolling batch dispatcher, strict non-re-allocation, and ACO gating."""

    def setUp(self):
        depot = Node(name="Depot", Lng=121.47, Lat=31.23)
        self.driver = Driver(name="Driver 1", current_location=depot, capacity=30.0)
        self.dispatcher = BatchDispatcher(drivers=[self.driver], batch_interval_seconds=30)

        self.task1 = DeliveryTask(
            task_name="Order 1",
            pickup=Node(name="P1", Lng=121.472, Lat=31.232),
            delivery=Node(name="D1", Lng=121.478, Lat=31.238),
            weight=5.0
        )
        self.task2 = DeliveryTask(
            task_name="Order 2",
            pickup=Node(name="P2", Lng=121.473, Lat=31.233),
            delivery=Node(name="D2", Lng=121.479, Lat=31.239),
            weight=5.0
        )

    def test_claimed_task_never_redispatched_in_subsequent_batch(self):
        """Once claimed in Batch 1, the order must be purged and never re-offered in Batch 2."""
        # Window 1: Add Order 1 and dispatch
        self.dispatcher.add_orders([self.task1])
        summary_1 = self.dispatcher.dispatch_batch(batch_id=1, current_time_sec=0)

        self.assertEqual(summary_1.claimed_count, 1)
        self.assertEqual(self.task1.status, TaskStatus.LOCKED)
        self.assertNotIn(self.task1, self.dispatcher.unallocated_order_pool, "Claimed order must be removed from pool")

        # Window 2 (T = 30s): New Order 2 arrives
        self.dispatcher.add_orders([self.task2])
        summary_2 = self.dispatcher.dispatch_batch(batch_id=2, current_time_sec=30)

        # Order 1 was already locked, only Order 2 should be processed
        self.assertEqual(summary_2.claimed_count, 1)
        self.assertEqual(len(self.driver.active_tasks), 2)
        # Verify Order 1 was NOT re-offered
        self.assertNotIn(self.task1, self.dispatcher.unallocated_order_pool)

    def test_re_adding_claimed_task_is_strictly_blocked(self):
        """Re-submitting an already-claimed task to add_orders must be completely ignored."""
        self.dispatcher.add_orders([self.task1])
        summary = self.dispatcher.dispatch_batch(batch_id=1, current_time_sec=0)
        self.assertEqual(summary.claimed_count, 1)
        self.assertTrue(self.dispatcher.is_task_allocated(self.task1))

        # Attempt to maliciously or accidentally re-add the claimed task
        self.dispatcher.add_orders([self.task1])
        self.assertNotIn(self.task1, self.dispatcher.unallocated_order_pool, "Claimed order must never re-enter pool")

    def test_step_batch_advances_30s_window(self):
        """Verify step_batch automatically increments batch ID and advances clock by 30 seconds."""
        # Batch 1 (T = 0s)
        s1 = self.dispatcher.step_batch(incoming_orders=[self.task1])
        self.assertEqual(s1.batch_id, 1)
        self.assertEqual(s1.timestamp_sec, 0)
        self.assertEqual(s1.claimed_count, 1)

        # Batch 2 (T = 30s)
        s2 = self.dispatcher.step_batch(incoming_orders=[self.task2])
        self.assertEqual(s2.batch_id, 2)
        self.assertEqual(s2.timestamp_sec, 30)
        self.assertEqual(s2.claimed_count, 1)

    def test_claimed_task_cleared_from_competing_drivers_offers(self):
        """When an order is claimed, competing drivers must have that order purged from offered_tasks."""
        driver_a = Driver(name="Driver A", current_location=Node(name="D1", Lng=121.47, Lat=31.23), capacity=20.0)
        driver_b = Driver(name="Driver B", current_location=Node(name="D2", Lng=121.47, Lat=31.23), capacity=20.0)
        multi_dispatcher = BatchDispatcher(drivers=[driver_a, driver_b], batch_interval_seconds=30)

        # Add single task
        multi_dispatcher.add_orders([self.task1])
        multi_dispatcher.dispatch_batch(batch_id=1, current_time_sec=0)

        # One driver claims it, both offered_tasks should now NOT contain self.task1
        self.assertNotIn(self.task1, driver_a.offered_tasks)
        self.assertNotIn(self.task1, driver_b.offered_tasks)
        self.assertTrue(multi_dispatcher.is_task_allocated(self.task1))

    def test_aco_only_runs_for_drivers_with_claimed_tasks(self):
        """Drivers with 0 claimed tasks must never trigger ACO solver."""
        # Unreachable order (far away)
        far_task = DeliveryTask(
            task_name="Far Order",
            pickup=Node(name="Far P", Lng=125.0, Lat=35.0),
            delivery=Node(name="Far D", Lng=125.1, Lat=35.1),
            weight=5.0
        )
        self.dispatcher.add_orders([far_task])
        summary = self.dispatcher.dispatch_batch(batch_id=1, current_time_sec=0)

        # Far order cannot be reached, so 0 tasks claimed
        self.assertEqual(summary.claimed_count, 0)
        self.assertEqual(len(summary.aco_latencies), 0, "No ACO compute should be triggered for 0 tasks")


class TestMultiThreadedScenarios(unittest.TestCase):
    """
    Verify multi-threading integration across Scenarios 1, 2, and 3:
    - Scenario 1: Multi-threaded fleet-level ACO itinerary optimization.
    - Scenario 2: Multi-threaded OSRM matrix querying.
    - Scenario 3: High-concurrency multi-threaded atomic order claiming race condition.
    """

    def test_scenario_1_concurrent_aco_routing(self):
        """Verify concurrent multi-threaded route optimization for multiple active couriers."""
        d1 = Driver(name="Driver 1", current_location=Node(name="Depot 1", Lng=121.47, Lat=31.23), capacity=6.0)
        d2 = Driver(name="Driver 2", current_location=Node(name="Depot 2", Lng=121.50, Lat=31.24), capacity=6.0)

        t1 = DeliveryTask(
            task_name="Task 1",
            pickup=Node(name="P1", Lng=121.472, Lat=31.232),
            delivery=Node(name="D1", Lng=121.478, Lat=31.238),
            weight=5.0
        )
        t2 = DeliveryTask(
            task_name="Task 2",
            pickup=Node(name="P2", Lng=121.502, Lat=31.242),
            delivery=Node(name="D2", Lng=121.508, Lat=31.248),
            weight=5.0
        )

        dispatcher = BatchDispatcher(
            drivers=[d1, d2],
            enable_concurrent_routing=True,
            max_workers=2
        )
        dispatcher.add_orders([t1, t2])
        summary = dispatcher.dispatch_batch(batch_id=1, current_time_sec=0)

        self.assertEqual(summary.claimed_count, 2)
        self.assertIn("Driver 1", summary.aco_latencies)
        self.assertIn("Driver 2", summary.aco_latencies)
        self.assertTrue(len(d1.assigned_route) > 0)
        self.assertTrue(len(d2.assigned_route) > 0)

    def test_scenario_2_concurrent_osrm_client(self):
        """Verify OSRMClient handles multi-threaded concurrent requests gracefully."""
        from services.osrm_client import OSRMClient
        client = OSRMClient(timeout=0.5)

        group_1 = [Node(name="N1", Lng=121.47, Lat=31.23), Node(name="N2", Lng=121.48, Lat=31.24)]
        group_2 = [Node(name="N3", Lng=121.50, Lat=31.25), Node(name="N4", Lng=121.51, Lat=31.26)]

        results = client.get_matrices_concurrent([group_1, group_2], max_workers=2)
        self.assertEqual(len(results), 2, "Should return 2 result entries for 2 node groups")

    def test_scenario_3_multi_threaded_atomic_claim_race(self):
        """
        Scenario 3: Stress test 10 concurrent threads simultaneously trying to claim 1 task.
        Guarantees exact-once semantics (1 claimed, 9 failed) with zero race condition glitches.
        """
        import threading
        from uuid import uuid4

        task = DeliveryTask(
            task_name="Hotspot Single Order",
            pickup=Node(name="P", Lng=121.47, Lat=31.23),
            delivery=Node(name="D", Lng=121.48, Lat=31.24),
            weight=3.0
        )

        num_threads = 10
        barrier = threading.Barrier(num_threads)
        results = []
        threads = []

        def racer():
            driver_uuid = uuid4()
            # Wait until all 10 threads are ready, then fire at the exact same instant
            barrier.wait()
            success = task.try_claim(driver_uuid)
            results.append(success)

        for _ in range(num_threads):
            t = threading.Thread(target=racer)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        success_count = sum(1 for r in results if r is True)
        fail_count = sum(1 for r in results if r is False)

        self.assertEqual(success_count, 1, "Exactly one thread must successfully claim the task")
        self.assertEqual(fail_count, 9, "All 9 competing threads must fail to claim")
        self.assertEqual(task.status, TaskStatus.LOCKED)


class TestAdaptiveH3Partitioner(unittest.TestCase):
    """Verify cross-border density-aware adaptive spatial partitioning (Scenario: Malaysia Klang Valley & Dense Hubs)."""

    def setUp(self):
        from services.adaptive_partitioner import AdaptiveH3Partitioner
        self.partitioner = AdaptiveH3Partitioner(
            base_resolution=5,
            max_resolution=8,
            max_orders_per_zone=15,  # Low threshold for testing split trigger
            enable_nearby_driver_fallback=True
        )

    def test_malaysia_klang_valley_cross_border_partitioning(self):
        """Verify seamless operation on Malaysia coordinates (Kuala Lumpur, PJ, Subang)."""
        # KLCC (Downtown KL)
        t_klcc = DeliveryTask(
            task_name="Order-KLCC-Pavilion",
            pickup=Node(name="KLCC Churn", Lng=101.7118, Lat=3.1578),
            delivery=Node(name="Pavilion KL", Lng=101.7132, Lat=3.1490)
        )
        # Petaling Jaya (PJ SS2)
        t_pj = DeliveryTask(
            task_name="Order-PJ-SS2-Durian",
            pickup=Node(name="PJ SS2 Food", Lng=101.6212, Lat=3.1189),
            delivery=Node(name="Damansara Uptown", Lng=101.6235, Lat=3.1362)
        )

        driver_kl = Driver(
            name="Rider-Ahmad (KLCC)",
            current_location=Node(name="Depot-KLCC", Lng=101.7120, Lat=3.1580),
            capacity=30.0
        )

        zones = self.partitioner.build_partition_zones(orders=[t_klcc, t_pj], drivers=[driver_kl])

        self.assertGreaterEqual(len(zones), 1)
        all_partition_orders = sum(z.order_count for z in zones)
        self.assertEqual(all_partition_orders, 2, "All Malaysian orders must be preserved in partition zones")

        # Verify Ahmad is matched as primary driver in his respective zone
        matched = any(driver_kl in z.primary_drivers or driver_kl in z.nearby_drivers for z in zones)
        self.assertTrue(matched, "Malaysian courier must be spatially matched to their local zone")

    def test_density_based_recursive_splitting(self):
        """When an area has order density exceeding threshold, it must split into higher resolutions."""
        # Generate 30 orders clustered in a small dense hotspot in downtown KLCC (exceeding max_orders_per_zone=15)
        clustered_orders = []
        for i in range(30):
            p = Node(name=f"P-{i}", Lng=101.7118 + (i % 5) * 0.001, Lat=3.1578 + (i // 5) * 0.001)
            d = Node(name=f"D-{i}", Lng=101.7150, Lat=3.1600)
            clustered_orders.append(DeliveryTask(task_name=f"Dense-Order-{i}", pickup=p, delivery=d))

        partitions = self.partitioner.partition_orders(clustered_orders)

        # Base resolution is 5. Since 30 > 15, the partitioner must have split down to Res 6 or higher
        resolutions = [h3.get_resolution(cell) for cell in partitions.keys()]
        self.assertTrue(any(r > 5 for r in resolutions), "Dense hotspot must recursively split past base resolution 5")

    def test_sparse_orders_remain_at_macro_resolution(self):
        """Sparse rural/suburban orders below threshold must remain merged at base resolution 5."""
        sparse_orders = [
            DeliveryTask(
                task_name="Sparse-1",
                pickup=Node(name="P-Sparse", Lng=101.500, Lat=3.000),
                delivery=Node(name="D-Sparse", Lng=101.505, Lat=3.005)
            )
        ]

        partitions = self.partitioner.partition_orders(sparse_orders)
        self.assertEqual(len(partitions), 1)
        single_cell = list(partitions.keys())[0]
        self.assertEqual(h3.get_resolution(single_cell), 5, "Sparse task must remain at base resolution 5")


class TestEnvironmentCache(unittest.TestCase):
    """Verify cached distance/cost lookups never recompute the Haversine fallback."""

    def test_cache_hit_skips_haversine(self):
        n1 = Node(name="A", Lng=101.70, Lat=3.15)
        n2 = Node(name="B", Lng=101.71, Lat=3.16)
        env = Environment(nodes=[n1, n2])
        expected_dist, expected_cost = env.get_distance(n1, n2), env.get_cost(n1, n2)

        calls = []
        env._calc_haversine_distance = lambda a, b: calls.append((a, b)) or 0.0
        self.assertEqual(env.get_distance(n1, n2), expected_dist)
        self.assertEqual(env.get_cost(n1, n2), expected_cost)
        self.assertEqual(calls, [], "Cache hit must not evaluate the Haversine fallback")


class TestRouteInsertion(unittest.TestCase):
    """Verify cheapest-insertion heuristic used for in-transit order stacking."""

    def setUp(self):
        self.origin = Node(name="Origin", Lng=101.60, Lat=3.10)
        self.tasks = [
            DeliveryTask(
                task_name=f"T{i}",
                pickup=Node(name=f"P{i}", Lng=101.60 + 0.01 * i, Lat=3.10 + 0.007 * (i % 3)),
                delivery=Node(name=f"D{i}", Lng=101.61 + 0.01 * i, Lat=3.12 - 0.006 * (i % 2)),
            )
            for i in range(4)
        ]
        nodes = [self.origin] + [n for t in self.tasks for n in (t.pickup, t.delivery)]
        self.env = Environment(nodes=nodes)
        self.ins = RouteInsertion(self.env.get_cost)

    def test_insert_pair_is_cheapest_feasible(self):
        stops = [self.tasks[0].pickup, self.tasks[0].delivery, self.tasks[1].pickup, self.tasks[1].delivery]
        p, d = self.tasks[2].pickup, self.tasks[2].delivery
        new_stops, delta = self.ins.insert_pair(self.origin, stops, p, d)

        self.assertLess(new_stops.index(p), new_stops.index(d), "Pickup must precede delivery")
        base = self.ins.route_cost(self.origin, stops)
        self.assertAlmostEqual(self.ins.route_cost(self.origin, new_stops) - base, delta, places=9)

        # Brute force over all feasible (i <= j) slot combinations
        best = min(
            self.ins.route_cost(self.origin, stops[:i] + [p] + stops[i:j] + [d] + stops[j:])
            for i in range(len(stops) + 1) for j in range(i, len(stops) + 1)
        )
        self.assertAlmostEqual(base + delta, best, places=9)

    def test_seed_keeps_previous_order_and_inserts_new_task(self):
        previous_route = [self.origin, self.tasks[1].pickup, self.tasks[0].pickup,
                          self.tasks[1].delivery, self.tasks[0].delivery, self.origin]
        tasks = self.tasks[:3]
        routing_nodes = [self.origin] + [n for t in tasks for n in (t.pickup, t.delivery)]

        seed = self.ins.build_seed_route(self.origin, routing_nodes, previous_route, tasks)

        self.assertEqual(seed[0], self.origin)
        self.assertEqual(seed[-1], self.origin)
        self.assertEqual(len(seed), len(routing_nodes) + 1)
        old_order = [n.id for n in seed if n.id in {x.id for x in previous_route[1:-1]}]
        self.assertEqual(old_order, [n.id for n in previous_route[1:-1]], "Previous itinerary order must be kept")
        required = {n.id for n in routing_nodes[1:]}
        self.assertTrue(RouteInsertion.respects_precedence(seed[1:-1], required))


class TestInTransitReRouting(unittest.TestCase):
    """Verify dynamic re-routing for in-transit order stacking."""

    def _task(self, i, lng, lat):
        return DeliveryTask(
            task_name=f"Order {i}",
            pickup=Node(name=f"P{i}", Lng=lng, Lat=lat),
            delivery=Node(name=f"D{i}", Lng=lng + 0.004, Lat=lat + 0.004),
            weight=2.0
        )

    def test_shared_courier_routed_once_across_zones(self):
        """A courier claiming in several zones must be solved exactly once."""
        courier = Driver(name="Shared", current_location=Node(name="Hub", Lng=101.70, Lat=3.15), capacity=50.0)
        zone_a = BatchDispatcher(drivers=[courier], enable_concurrent_routing=False)
        zone_b = BatchDispatcher(drivers=[courier], enable_concurrent_routing=False)
        zone_a.add_orders([self._task(1, 101.701, 3.151)])
        zone_b.add_orders([self._task(2, 101.702, 3.152)])

        zone_a.claim_batch(batch_id=1, current_time_sec=0)
        zone_b.claim_batch(batch_id=1, current_time_sec=0)
        self.assertTrue(courier.needs_reroute)
        self.assertEqual(courier.assigned_route, [], "Claim phase must not trigger routing")

        router = BatchDispatcher(drivers=[courier, courier], enable_concurrent_routing=False)
        latencies = router.route_pending_drivers()
        self.assertEqual(list(latencies), ["Shared"])
        self.assertFalse(courier.needs_reroute)
        self.assertEqual(len(courier.assigned_route), 6, "Hub + 2 pickups + 2 deliveries + return")
        self.assertEqual(router.route_pending_drivers(), {}, "Unchanged couriers must not be re-solved")

    def test_in_transit_reroute_after_pickup(self):
        """After a pickup, re-routing skips that pickup and still delivers it, plus the new order."""
        courier = Driver(name="Rider", current_location=Node(name="Hub", Lng=101.70, Lat=3.15), capacity=50.0)
        dispatcher = BatchDispatcher(drivers=[courier], enable_concurrent_routing=False)
        first = self._task(1, 101.701, 3.151)
        dispatcher.step_batch(incoming_orders=[first])

        # Travel until the first order's pickup has been completed
        while courier.current_location.id != first.pickup.id:
            courier.advance_to_next_stop()
        self.assertIn(first.id, courier.picked_up_task_ids)

        second = self._task(2, 101.703, 3.153)
        summary = dispatcher.step_batch(incoming_orders=[second])
        self.assertIn("Rider", summary.aco_latencies, "In-transit stacking must trigger re-routing")

        route_ids = [n.id for n in courier.assigned_route]
        self.assertEqual(route_ids[0], first.pickup.id, "Route starts at the courier's current location")
        self.assertEqual(route_ids.count(first.pickup.id), 2, "Completed pickup appears only as origin and return")
        self.assertIn(first.delivery.id, route_ids)
        self.assertLess(route_ids.index(second.pickup.id), route_ids.index(second.delivery.id))

    def test_aco_warm_start_never_worse_than_seed(self):
        origin = Node(name="Origin", Lng=101.60, Lat=3.10)
        tasks = [self._task(i, 101.60 + 0.01 * i, 3.10 + 0.005 * i) for i in range(5)]
        nodes = [origin] + [n for t in tasks for n in (t.pickup, t.delivery)]
        env = Environment(nodes=nodes)
        seed = RouteInsertion(env.get_cost).build_seed_route(origin, nodes, [], tasks)
        seed_cost = sum(env.get_cost(seed[i], seed[i + 1]) for i in range(len(seed) - 1))

        best_path, best_cost = ACO(env=env, depot=origin, num_ants=5, max_iter=3, initial_path=seed).run()
        self.assertLessEqual(best_cost, seed_cost + 1e-9)
        self.assertEqual(len(best_path), len(nodes) + 1)


class TestProcessPoolRouting(unittest.TestCase):
    """Verify ACO solving in worker processes (GIL-free multi-core routing)."""

    def _courier_with_tasks(self, i):
        lng, lat = 101.60 + 0.05 * i, 3.10
        courier = Driver(name=f"Courier {i}", current_location=Node(name=f"Hub {i}", Lng=lng, Lat=lat), capacity=50.0)
        tasks = [
            DeliveryTask(
                task_name=f"C{i}-T{k}",
                pickup=Node(name=f"C{i}-P{k}", Lng=lng + 0.002 * k, Lat=lat + 0.003),
                delivery=Node(name=f"C{i}-D{k}", Lng=lng - 0.002 * k, Lat=lat - 0.003),
                weight=2.0
            )
            for k in range(4)
        ]
        for t in tasks:
            courier.receive_offer(t)
            courier.accept_task(t)
        return courier, tasks

    def test_problem_is_picklable(self):
        import pickle
        courier, _ = self._courier_with_tasks(0)
        env = Environment(nodes=courier.get_pending_routing_nodes())
        problem = ACO(env=env, depot=courier.current_location, num_ants=5, max_iter=2).build_problem()
        restored = pickle.loads(pickle.dumps(problem))
        self.assertEqual(restored.dist.shape, (9, 9))

    def test_process_backend_routes_all_couriers(self):
        fleet = [self._courier_with_tasks(i) for i in range(3)]
        with BatchDispatcher(drivers=[c for c, _ in fleet], max_workers=2, routing_backend="process") as router:
            latencies = router.route_pending_drivers()

        self.assertEqual(len(latencies), 3)
        for courier, tasks in fleet:
            route_ids = [n.id for n in courier.assigned_route]
            self.assertEqual(route_ids[0], courier.current_location.id)
            self.assertEqual(route_ids[-1], courier.current_location.id)
            for t in tasks:
                self.assertLess(route_ids.index(t.pickup.id), route_ids.index(t.delivery.id))
            self.assertFalse(courier.needs_reroute)

    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError):
            BatchDispatcher(drivers=[], routing_backend="gpu")


if __name__ == "__main__":
    unittest.main()




