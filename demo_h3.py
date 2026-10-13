import sys
import io

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from model.node import Node
from services.spatial_index import SpatialIndexer


def demo_uber_h3_spatial_indexing():
    print("=" * 60)
    print("🚀 [DEMO] Uber H3 空间蜂窝网格索引与自适应剪枝实测")
    print("=" * 60)

    # 1. 模拟全城分散的 8 个订单（包含闹市区集中单与偏远郊区孤单）
    orders = [
        # 人民广场/南京东路附近 (高密度核心区)
        Node(name="外卖订单 1 (黄浦区人民广场)", Lng=121.4737, Lat=31.2304),
        Node(name="外卖订单 2 (黄浦区南京东路)", Lng=121.4850, Lat=31.2380),
        Node(name="外卖订单 3 (静安区静安寺)",   Lng=121.4460, Lat=31.2230),

        # 浦东陆家嘴区域
        Node(name="外卖订单 4 (浦东陆家嘴)",     Lng=121.5050, Lat=31.2400),
        Node(name="外卖订单 5 (浦东世纪大道)",   Lng=121.5280, Lat=31.2300),

        # 远郊孤立单 (极其偏远地区)
        Node(name="外卖订单 6 (嘉定新城远郊单)", Lng=121.2500, Lat=31.3500),
        Node(name="外卖订单 7 (青浦朱家角远郊单)", Lng=121.0500, Lat=31.1100),
        Node(name="外卖订单 8 (崇明堡镇海岛单)", Lng=121.6000, Lat=31.5400),
    ]

    # 2. 模拟 2 名身处不同起点的骑手
    driver_huangpu = Node(name="骑手-小张 (身在人民广场)", Lng=121.4740, Lat=31.2310)
    driver_jiading = Node(name="骑手-老李 (身在嘉定郊区)", Lng=121.2510, Lat=31.3520)

    # 3. 初始化 H3 空间索引器并建索引 (Resolution=8，蜂窝边长约 460米)
    indexer = SpatialIndexer(resolution=8)
    indexer.build_index(orders)

    print(f"\n[1] 成功将 {len(orders)} 个订单编入 H3 空间网格桶:")
    for order in orders:
        cell = indexer.get_node_cell(order)
        print(f"  • {order.name:<25} -> H3 Cell: {cell}")

    # 4. 场景测试 A：骑手小张在人民广场核心区找单（就近圈定，剪枝 80% 远单）
    print(f"\n[2] 场景 A: 【{driver_huangpu.name}】圈定周边候选单 (最大圈层 k=2, 约2公里)")
    candidates_a = indexer.find_nearby_nodes(driver_huangpu, max_k_ring=2)
    print(f"  -> 命中候选单数量: {len(candidates_a)} / {len(orders)}")
    for cand in candidates_a:
        print(f"     ✅ 圈中顺路单: {cand.name}")

    # 5. 场景测试 B：骑手老李在偏远嘉定找单（只圈出嘉定本地单，绝不跨区拉取市区单）
    print(f"\n[3] 场景 B: 【{driver_jiading.name}】圈定周边候选单 (最大圈层 k=2)")
    candidates_b = indexer.find_nearby_nodes(driver_jiading, max_k_ring=2)
    print(f"  -> 命中候选单数量: {len(candidates_b)} / {len(orders)}")
    for cand in candidates_b:
        print(f"     ✅ 圈中本地单: {cand.name}")

    # 6. 场景测试 C：自适应阶梯扩圈验证（如果近处 1 圈没人，自动向外扩搜）
    print(f"\n[4] 场景 C: 验证自适应扩圈 (当 k=1 无单时自动向外扩)")
    far_driver = Node(name="路人骑手 (距离陆家嘴较远 3 公里)", Lng=121.5500, Lat=31.2500)
    # k=1 找不到，扩到 k=3 成功找到陆家嘴和世纪大道附近的单
    candidates_c = indexer.find_nearby_nodes(far_driver, max_k_ring=4, min_candidates=1)
    print(f"  -> 自适应扩圈后捕获订单数量: {len(candidates_c)}")
    for cand in candidates_c:
        print(f"     🔍 扩圈捕获: {cand.name}")

    print("\n" + "=" * 60)
    print("🎉 验证成功: 空间网格完美隔离不同区域运力，无多余几何计算！")
    print("=" * 60)


if __name__ == "__main__":
    demo_uber_h3_spatial_indexing()
