import sys
import io
import time
from typing import List, Dict
from uuid import UUID
import h3

# Ensure UTF-8 console output encoding on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from model.node import Node
from model.driver import Driver
from model.enviroment import Environment
from model.aco import ACO
from services.spatial_index import SpatialIndexer


class DeliveryTask:
    """
    业务订单任务：包含取货点、送货点与货物重量。
    天然绑定 PDP 先序依赖约束（送货前必须先取货）。
    """
    def __init__(self, task_name: str, pickup: Node, delivery: Node, weight: float = 3.0):
        self.task_name = task_name
        self.pickup = pickup
        self.delivery = delivery
        self.weight = weight
        # 确保送货点的前置依赖严格包含取货点
        if self.pickup.id not in self.delivery.prerequisites:
            self.delivery.prerequisites.append(self.pickup.id)


def build_simulation_scenario():
    """
    构建现实仿真场景：
    - 3 位身处上海不同商圈/起点的骑手
    - 6 个分布在不同区域的即时取送订单（共 12 个物理站点）
    """
    # 1. 司机初始化（分布在三大核心商圈）
    driver_huangpu = Driver(
        name="骑手-小张(黄浦商圈)",
        current_location=Node(name="黄浦起点-人民广场", Lng=121.4740, Lat=31.2304),
        capacity=25.0  # 最大装载 25 kg
    )
    driver_pudong = Driver(
        name="骑手-小王(陆家嘴商圈)",
        current_location=Node(name="浦东起点-国金中心", Lng=121.5030, Lat=31.2380),
        capacity=30.0  # 最大装载 30 kg
    )
    driver_jingan = Driver(
        name="骑手-小李(静安寺商圈)",
        current_location=Node(name="静安起点-静安寺", Lng=121.4460, Lat=31.2230),
        capacity=20.0  # 最大装载 20 kg
    )

    drivers = [driver_huangpu, driver_pudong, driver_jingan]

    # 2. 模拟取送订单任务（Task 1~6）
    tasks = [
        # 黄浦区区域订单 (靠近骑手小张)
        DeliveryTask(
            task_name="黄浦订单T1 (南京东路奶茶 -> 外滩观景台)",
            pickup=Node(name="[取]南京东路奶茶店", Lng=121.4850, Lat=31.2380),
            delivery=Node(name="[送]外滩金融中心", Lng=121.4920, Lat=31.2350),
            weight=4.0
        ),
        DeliveryTask(
            task_name="黄浦订单T2 (新天地西餐 -> 淮海中路办公楼)",
            pickup=Node(name="[取]新天地餐厅", Lng=121.4760, Lat=31.2210),
            delivery=Node(name="[送]淮海中路K11", Lng=121.4720, Lat=31.2230),
            weight=6.0
        ),

        # 浦东陆家嘴区域订单 (靠近骑手小王)
        DeliveryTask(
            task_name="浦东订单T3 (正大广场日料 -> 东方明珠塔)",
            pickup=Node(name="[取]正大广场日料", Lng=121.5000, Lat=31.2410),
            delivery=Node(name="[送]东方明珠观光厅", Lng=121.4990, Lat=31.2390),
            weight=5.0
        ),
        DeliveryTask(
            task_name="浦东订单T4 (世纪汇烘焙 -> 环球金融中心)",
            pickup=Node(name="[取]世纪大道世纪汇", Lng=121.5280, Lat=31.2300),
            delivery=Node(name="[送]上海环球金融中心", Lng=121.5070, Lat=31.2370),
            weight=8.0
        ),

        # 静安寺区域订单 (靠近骑手小李)
        DeliveryTask(
            task_name="静安订单T5 (久光百货轻食 -> 愚园路客户)",
            pickup=Node(name="[取]久光百货轻食", Lng=121.4470, Lat=31.2240),
            delivery=Node(name="[送]愚园路老洋房", Lng=121.4400, Lat=31.2210),
            weight=3.5
        ),
        DeliveryTask(
            task_name="静安订单T6 (芮欧百货咖啡 -> 陕西北路写字楼)",
            pickup=Node(name="[取]芮欧百货精品咖啡", Lng=121.4500, Lat=31.2250),
            delivery=Node(name="[送]陕西北路恒隆广场", Lng=121.4550, Lat=31.2280),
            weight=4.5
        ),
    ]

    return drivers, tasks


def main():
    print("=" * 70)
    print("🚀 [START] 工业级两阶段多运力智能调度仿真系统 (Two-Stage Fleet Router)")
    print("=" * 70)

    # 1. 初始化场景数据
    drivers, tasks = build_simulation_scenario()
    print(f"\n[INIT] 成功载入运力与订单:")
    print(f"  • 在线骑手: {len(drivers)} 名 (分布于黄浦、浦东、静安不同起点)")
    print(f"  • 待派订单: {len(tasks)} 笔 (对应 {len(tasks) * 2} 个取送货站点，具备严格先序约束)")

    t_total_start = time.perf_counter()

    # =========================================================================
    # 第一阶段：Uber H3 空间网格建立与微商圈订单分配 (Cluster / Matcher)
    # =========================================================================
    print("\n" + "-" * 70)
    print("📍 [STAGE 1] 基于 Uber H3 空间蜂窝网格执行微商圈订单分配...")
    print("-" * 70)

    t_stage1_start = time.perf_counter()

    # 构建 H3 空间索引器 (Resolution 8，单蜂窝边长约 460 米)
    indexer = SpatialIndexer(resolution=8)
    pickup_to_task: Dict[UUID, DeliveryTask] = {}

    for task in tasks:
        indexer.add_node(task.pickup)  # 以取餐商家作为空间索引锚点
        pickup_to_task[task.pickup.id] = task

    t_index_built = time.perf_counter()

    # 分配容器：每个司机分配到的 DeliveryTask 列表
    driver_assigned_tasks: Dict[Driver, List[DeliveryTask]] = {d: [] for d in drivers}
    unassigned_tasks = list(tasks)

    # 依次为每个待派订单，基于 H3 网格步数就近匹配最优骑手
    for task in tasks:
        task_cell = indexer.get_node_cell(task.pickup)
        best_driver = None
        min_h3_steps = float("inf")

        for driver in drivers:
            if not driver.can_carry(task.weight):
                continue
            driver_cell = indexer.get_node_cell(driver.current_location)
            # 通过 H3 蜂窝拓扑步数快速评估空间邻近度 (无需耗时路网计算)
            steps = h3.grid_distance(driver_cell, task_cell)
            if steps < min_h3_steps and steps <= 5:  # 限制在同城微商圈 5 步之内 (约 3.5km)
                min_h3_steps = steps
                best_driver = driver

        if best_driver:
            best_driver.load_cargo(task.weight)
            driver_assigned_tasks[best_driver].append(task)
            unassigned_tasks.remove(task)
        else:
            print(f"  ⚠️ 订单 {task.task_name} 无匹配顺路骑手或运力不足")

    t_stage1_end = time.perf_counter()
    stage1_duration_ms = (t_stage1_end - t_stage1_start) * 1000

    print(f"[STAGE 1 完成] H3 空间建索引耗时: {(t_index_built - t_stage1_start) * 1000:.3f} ms")
    print(f"[STAGE 1 完成] 空间邻域分配总耗时: {stage1_duration_ms:.3f} ms")
    for driver in drivers:
        assigned = driver_assigned_tasks[driver]
        print(f"  -> {driver.name:<18} 分配订单数: {len(assigned)} | "
              f"在车载荷: {driver.current_load:.1f} / {driver.capacity:.1f} kg "
              f"({driver.current_load / driver.capacity * 100:.1f}%)")

    if unassigned_tasks:
        print(f"  ⚠️ 滞留池未分配订单: {len(unassigned_tasks)} 笔 (周围无空闲运力)")

    # =========================================================================
    # 第二阶段：逐车 ACO 启发式路径规划与先序约束求解 (Local ACO Router)
    # =========================================================================
    print("\n" + "-" * 70)
    print("🐜 [STAGE 2] 各骑手基于专属起点与任务池，独立执行 ACO 路径寻优...")
    print("-" * 70)

    t_stage2_start = time.perf_counter()
    aco_time_records = {}

    for driver, assigned in driver_assigned_tasks.items():
        if not assigned:
            print(f"  • {driver.name}: 无分配任务，保持 IDLE 待命。")
            continue

        # 整理该骑手必须访问的所有节点（起点 + 该司机的所有取货点与送货点）
        driver_nodes: List[Node] = [driver.current_location]
        for task in assigned:
            driver_nodes.append(task.pickup)
            driver_nodes.append(task.delivery)

        # 构造局部拓扑环境与 ACO 求解器
        local_env = Environment(nodes=driver_nodes, routing_client=None, init_pheromone=1.0)
        aco = ACO(
            env=local_env,
            depot=driver.current_location,  # 以骑手实时坐标为起点
            num_ants=25,
            max_iter=30,
            initial_radius=10.0,
            radius_step=5.0,
            max_radius=50.0
        )

        t_driver_start = time.perf_counter()
        best_path, best_cost = aco.run()
        t_driver_end = time.perf_counter()

        driver_time_ms = (t_driver_end - t_driver_start) * 1000
        aco_time_records[driver.name] = driver_time_ms

        # 下发最佳路线至司机业务实体
        driver.assign_route(best_path, best_cost)

        print(f"\n  🎯 【{driver.name}】规划完毕 (求解耗时: {driver_time_ms:.2f} ms):")
        print(f"     路线总成本 (Cost): {best_cost:.2f} | 停靠站点数: {len(best_path)}")
        for step, stop in enumerate(best_path, 1):
            print(f"       Step {step:02d} -> {stop.name}")

        # 自动审计 PDP 取送先序合规性
        path_names = [n.name for n in best_path]
        compliance_failed = False
        for task in assigned:
            idx_p = path_names.index(task.pickup.name)
            idx_d = path_names.index(task.delivery.name)
            if idx_p > idx_d:
                compliance_failed = True
                print(f"       ❌ [违规] 取送先序错误: {task.pickup.name} (#{idx_p+1}) 在 {task.delivery.name} (#{idx_d+1}) 之后！")

        if not compliance_failed:
            print(f"     ✅ [合规审计] 所有订单均严格遵守「先取后送」先序依赖约束 (100% PASSED)！")

    t_stage2_end = time.perf_counter()
    stage2_duration_ms = (t_stage2_end - t_stage2_start) * 1000
    total_duration_ms = (time.perf_counter() - t_total_start) * 1000

    # =========================================================================
    # 汇总性能大盘与耗时审计报告 (Performance Profiling Dashboard)
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 [PERFORMANCE AUDIT] 调度与规划全链路耗时审计大盘")
    print("=" * 70)
    print(f" 阶段一 (H3 空间网格建桶与邻域匹配耗时):   {stage1_duration_ms:>8.2f} ms")
    print(f" 阶段二 (全部车队 ACO 路径规划总耗时):    {stage2_duration_ms:>8.2f} ms")
    for d_name, d_ms in aco_time_records.items():
        print(f"    - 单车寻优耗时 [{d_name:<16}]: {d_ms:>8.2f} ms")
    print("-" * 70)
    print(f" 🌟 全系统端到端总响应时延 (End-to-End Latency): {total_duration_ms:>8.2f} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
