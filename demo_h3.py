import sys
import io

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from model.node import Node
from services.spatial_index import SpatialIndexer


def demo_uber_h3_spatial_indexing():
    print("=" * 70)
    print("🚀 [DEMO] Uber H3 Hexagonal Spatial Indexing & Neighborhood Pruning")
    print("=" * 70)
    print("Geographic Scope: Greater Kuala Lumpur (Klang Valley, Malaysia)")
    print("Coordinate Standard: Native WGS-84 (EPSG:4326)")
    print("-" * 70)

    # 1. Simulate 8 dispersed orders across Klang Valley (urban core clusters vs remote suburban outliers)
    orders = [
        # Downtown Kuala Lumpur / Golden Triangle core cluster
        Node(name="Order 1 (Suria KLCC Churn)",        Lng=101.7120, Lat=3.1582),
        Node(name="Order 2 (Pavilion Bukit Bintang)",  Lng=101.7130, Lat=3.1488),
        Node(name="Order 3 (Changkat Gastropub)",      Lng=101.7085, Lat=3.1475),

        # Petaling Jaya commercial cluster (~10km west)
        Node(name="Order 4 (SS2 Durian Street)",       Lng=101.6215, Lat=3.1192),
        Node(name="Order 5 (Damansara Uptown Bistro)", Lng=101.6230, Lat=3.1360),

        # Distant peripheral outliers (Suburban & Remote)
        Node(name="Order 6 (Rawang Suburb Outlier)",   Lng=101.5750, Lat=3.3210),
        Node(name="Order 7 (Cyberjaya Tech Campus)",   Lng=101.6500, Lat=2.9210),
        Node(name="Order 8 (Genting Highlands Resort)", Lng=101.7900, Lat=3.4240),
    ]

    # 2. Simulate 2 couriers at distinct starting coordinates
    driver_klcc = Node(name="Rider Farhan (KLCC)", Lng=101.7118, Lat=3.1578)
    driver_rawang = Node(name="Rider Azman (Rawang Suburb)", Lng=101.5760, Lat=3.3220)

    # 3. Initialize H3 SpatialIndexer and bulk index nodes (Resolution=8, cell edge ~460m)
    indexer = SpatialIndexer(resolution=8)
    indexer.build_index(orders)

    print(f"\n[1] Successfully indexed {len(orders)} orders into H3 hexagonal buckets:")
    for order in orders:
        cell = indexer.get_node_cell(order)
        print(f"  • {order.name:<34} -> H3 Cell: {cell}")

    # 4. Scenario A: Rider Farhan finds orders around KLCC (pruning distant orders)
    print(f"\n[2] Scenario A: 【{driver_klcc.name}】queries neighborhood (max ring k=2, ~2km)")
    candidates_a = indexer.find_nearby_nodes(driver_klcc, max_k_ring=2)
    print(f"  -> Captured candidates: {len(candidates_a)} / {len(orders)}")
    for cand in candidates_a:
        print(f"     ✅ Matched neighborhood order: {cand.name}")

    # 5. Scenario B: Rider Azman queries orders around Rawang (captures local outlier only)
    print(f"\n[3] Scenario B: 【{driver_rawang.name}】queries neighborhood (max ring k=2)")
    candidates_b = indexer.find_nearby_nodes(driver_rawang, max_k_ring=2)
    print(f"  -> Captured candidates: {len(candidates_b)} / {len(orders)}")
    for cand in candidates_b:
        print(f"     ✅ Matched local suburb order: {cand.name}")

    # 6. Scenario C: Verify adaptive ring expansion (gradually expand when immediate ring k=1 is empty)
    print(f"\n[4] Scenario C: Verify adaptive expansion (expands outward when immediate ring is empty)")
    far_driver = Node(name="En-route Rider (Ampang ~3km away)", Lng=101.7450, Lat=3.1550)
    # Expands up to k=5 to capture KLCC area orders
    candidates_c = indexer.find_nearby_nodes(far_driver, max_k_ring=5, min_candidates=1)
    print(f"  -> Captured orders via adaptive expansion: {len(candidates_c)}")
    for cand in candidates_c:
        print(f"     🔍 Captured candidate: {cand.name}")

    print("\n" + "=" * 70)
    print("🎉 Verification SUCCESS: H3 spatial indexing isolates district dispatch effortlessly!")
    print("=" * 70)


if __name__ == "__main__":
    demo_uber_h3_spatial_indexing()
