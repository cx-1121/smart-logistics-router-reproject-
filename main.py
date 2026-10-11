import sys
import io

# Ensure UTF-8 console output encoding on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from model.node import Node
from model.enviroment import Environment
from model.aco import ACO
from services.osrm_client import OSRMClient


def main():
    print("[START] Launching Smart Logistics Router simulation...")

    # 1. Create simulated logistics nodes (using Shanghai coordinates as demonstration)
    depot = Node(name="Central Depot", Lng=121.4737, Lat=31.2304)
    pickup_a = Node(name="Pickup Point A", Lng=121.5000, Lat=31.2400)
    pickup_b = Node(name="Pickup Point B", Lng=121.4500, Lat=31.2100)

    # Precedence constraint: delivery_a must be visited after pickup_a
    delivery_a = Node(
        name="Delivery Point A",
        Lng=121.5200,
        Lat=31.2600,
        prerequisites=[pickup_a.id],  # Must visit Pickup A first
    )

    # Precedence constraint: delivery_b must be visited after pickup_b
    delivery_b = Node(
        name="Delivery Point B",
        Lng=121.4200,
        Lat=31.1900,
        prerequisites=[pickup_b.id],  # Must visit Pickup B first
    )

    client_c = Node(name="Regular Customer C", Lng=121.4800, Lat=31.2200)

    nodes = [depot, pickup_a, pickup_b, delivery_a, delivery_b, client_c]
    print(f"[INFO] Loaded {len(nodes)} logistics nodes with precedence constraints.")

    # 2. Build dedicated routing client and inject into environment
    osrm_client = OSRMClient(host="http://router.project-osrm.org", timeout=5.0)
    env = Environment(nodes=nodes, routing_client=osrm_client, init_pheromone=1.0)

    # 3. Initialize and execute Ant Colony Optimization (ACO)
    aco = ACO(
        env=env,
        depot=depot,  # ★ Fixed start & return depot for commercial logistics dispatch
        num_ants=20,
        max_iter=30,
        alpha=1.0,
        beta=2.5,
        rho=0.1,
        q=100.0,
        initial_radius=10.0,
        radius_step=5.0,
        max_radius=50.0,
    )

    print("[INFO] Running ACO algorithm to optimize route...")
    best_path, best_cost = aco.run()

    if not best_path:
        print("[ERROR] Failed to find a valid route satisfying all constraints.")
        return

    print("\n" + "=" * 55)
    print(f"[SUCCESS] Optimal route found! Total Cost: {best_cost:.2f}")
    print("=" * 55)

    print("Recommended sequence of stops:")
    for step, node in enumerate(best_path, 1):
        print(f"  Step {step:02d} -> {node.name} (Lng: {node.Lng:.4f}, Lat: {node.Lat:.4f})")

    # 4. Verify precedence constraint compliance
    visited_names = [n.name for n in best_path]
    idx_pa = visited_names.index(pickup_a.name)
    idx_da = visited_names.index(delivery_a.name)
    idx_pb = visited_names.index(pickup_b.name)
    idx_db = visited_names.index(delivery_b.name)

    print("\n[VERIFICATION] Checking precedence constraints:")
    print(f"  - Pickup A (Stop #{idx_pa + 1}) vs Delivery A (Stop #{idx_da + 1}) => "
          f"{'PASSED (pickup before delivery)' if idx_pa < idx_da else 'FAILED'}")
    print(f"  - Pickup B (Stop #{idx_pb + 1}) vs Delivery B (Stop #{idx_db + 1}) => "
          f"{'PASSED (pickup before delivery)' if idx_pb < idx_db else 'FAILED'}")


if __name__ == "__main__":
    main()
