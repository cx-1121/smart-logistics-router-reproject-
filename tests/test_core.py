import sys
from pathlib import Path
import unittest

# 将项目根目录添加至模块搜索路径，确保直接运行该脚本时能正确导入 model 与 services
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.node import Node
from model.ant import Ant
from model.driver import Driver, DriverStatus
from model.task import DeliveryTask
from model.enviroment import Environment
from model.aco import ACO
from services.spatial_index import SpatialIndexer


class TestDriverDomainModel(unittest.TestCase):
    """验证 Driver 业务实体的状态流转与载重约束防御"""

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
        """测试超载时是否准时抛出异常"""
        self.driver.load_cargo(40.0)
        with self.assertRaises(ValueError):
            self.driver.load_cargo(15.0)  # 40 + 15 > 50.0

    def test_route_assignment_and_lifecycle(self):
        """测试路径下发与状态变迁"""
        stop1 = Node(name="Stop 1", Lng=121.4800, Lat=31.2350)
        stop2 = Node(name="Stop 2", Lng=121.4900, Lat=31.2400)

        self.driver.assign_route([stop1, stop2], route_cost=18.5)
        self.assertEqual(self.driver.status, DriverStatus.BUSY)
        self.assertFalse(self.driver.is_available)
        self.assertEqual(self.driver.next_stop.name, "Stop 1")
        self.assertEqual(self.driver.total_route_cost, 18.5)

        self.driver.clear_route()
        self.assertEqual(self.driver.status, DriverStatus.IDLE)
        self.assertTrue(self.driver.is_available)
        self.assertIsNone(self.driver.next_stop)


class TestAntBehaviorAndConstraints(unittest.TestCase):
    """验证 Ant 个体的先序依赖约束 (PDP) 与邻域搜索安全性"""

    def setUp(self):
        self.origin = Node(name="Origin", Lng=121.4737, Lat=31.2304)
        self.pickup = Node(name="Pickup Point", Lng=121.4800, Lat=31.2350)
        self.delivery = Node(
            name="Delivery Point",
            Lng=121.4900,
            Lat=31.2400,
            prerequisites=[self.pickup.id]  # 送货点依赖取货点
        )
        self.ant = Ant(start_node=self.origin)

    def dummy_dist(self, n1: Node, n2: Node) -> float:
        return 1.0

    def test_precedence_constraint_filtering(self):
        """在未取货前，送货点严禁出现在可选候选集中"""
        all_nodes = [self.origin, self.pickup, self.delivery]

        # 此时只访问过 origin，pickup 未被访问
        available = self.ant.get_available_nodes(
            all_nodes=all_nodes,
            get_distance_func=self.dummy_dist,
            initial_radius=10.0,
            radius_step=5.0,
            max_radius=20.0
        )

        candidate_ids = {node.id for node in available}
        self.assertIn(self.pickup.id, candidate_ids, "取货点应该可选")
        self.assertNotIn(self.delivery.id, candidate_ids, "送货点必须被先序规则拦截")

        # 蚂蚁移动到取货点
        self.ant.move_to(self.pickup, cost=1.0)

        # 此时 pickup 已被访问，delivery 应该被成功解禁
        available_after = self.ant.get_available_nodes(
            all_nodes=all_nodes,
            get_distance_func=self.dummy_dist,
            initial_radius=10.0,
            radius_step=5.0,
            max_radius=20.0
        )
        candidate_ids_after = {node.id for node in available_after}
        self.assertIn(self.delivery.id, candidate_ids_after, "取货完成后送货点必须解禁")

    def test_deadlock_free_when_no_candidates(self):
        """当所有节点已被访问完时，必须安全返回空列表，绝不死循环"""
        all_nodes = [self.origin]  # 只有起点，已被访问
        available = self.ant.get_available_nodes(
            all_nodes=all_nodes,
            get_distance_func=self.dummy_dist,
            initial_radius=10.0,
            radius_step=5.0,
            max_radius=20.0
        )
        self.assertEqual(available, [], "无候选点时必须安全返回空列表")


class TestSpatialIndexerH3(unittest.TestCase):
    """验证 Uber H3 空间网格索引与剪枝过滤"""

    def setUp(self):
        self.indexer = SpatialIndexer(resolution=8)
        self.center_node = Node(name="Center", Lng=121.4737, Lat=31.2304)
        self.near_node = Node(name="Near (500m)", Lng=121.4750, Lat=31.2315)
        self.far_node = Node(name="Far (Jiading 25km)", Lng=121.2500, Lat=31.3500)

    def test_spatial_pruning(self):
        self.indexer.build_index([self.near_node, self.far_node])

        # 在 k=1 圈层内搜索，应该只圈中近处节点，成功剪掉远距离节点
        nearby = self.indexer.find_nearby_nodes(self.center_node, max_k_ring=1)
        nearby_names = [n.name for n in nearby]

        self.assertIn("Near (500m)", nearby_names)
        self.assertNotIn("Far (Jiading 25km)", nearby_names)


class TestACOEndToEnd(unittest.TestCase):
    """验证 ACO 求解器的关键业务不变性（Invariants）"""

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

        # 1. 必须找到可行解
        self.assertTrue(len(best_path) > 0, "必须生成有效路径")
        self.assertLess(best_cost, float("inf"), "解的代价值必须为有限数")

        # 2. 闭环回路校验：起点和终点必须是 depot
        self.assertEqual(best_path[0].name, "Depot")
        self.assertEqual(best_path[-1].name, "Depot")

        # 3. 先序约束校验：Pickup 必须排在 Delivery 前面
        path_names = [n.name for n in best_path]
        idx_p = path_names.index("Pickup")
        idx_d = path_names.index("Delivery")
        self.assertLess(idx_p, idx_d, f"取货点索引({idx_p})必须小于送货点索引({idx_d})")


class TestDeliveryTaskDomainModel(unittest.TestCase):
    """验证 DeliveryTask 订单原子对的自动依赖注入与空间计算"""

    def test_automatic_precedence_injection(self):
        p = Node(name="Pickup", Lng=121.4, Lat=31.2)
        d = Node(name="Delivery", Lng=121.5, Lat=31.3)
        # 初始时 delivery 的 prerequisites 为空
        self.assertNotIn(p.id, d.prerequisites)

        # 组装为 DeliveryTask
        task = DeliveryTask(task_name="Order 1", pickup=p, delivery=d, weight=5.0)

        # 验证自动绑定钩子是否生效
        self.assertIn(p.id, task.delivery.prerequisites, "实例化时送货点必须自动注入取货点ID")
        self.assertEqual(task.weight, 5.0)

    def test_midpoint_calculation(self):
        p = Node(name="P", Lng=120.0, Lat=30.0)
        d = Node(name="D", Lng=122.0, Lat=32.0)
        task = DeliveryTask(task_name="Order Mid", pickup=p, delivery=d)

        mid_lng, mid_lat = task.midpoint
        self.assertAlmostEqual(mid_lng, 121.0)
        self.assertAlmostEqual(mid_lat, 31.0)


if __name__ == "__main__":
    unittest.main()

