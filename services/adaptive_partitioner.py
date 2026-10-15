import h3
from typing import List, Dict, Tuple, Optional, Set
from pydantic import BaseModel, Field

from model.node import Node
from model.driver import Driver
from model.task import DeliveryTask


class PartitionZone(BaseModel):
    """
    Domain entity representing an adaptively sized spatial dispatch zone.
    Maintains variable H3 resolution based on localized order density.
    """
    zone_id: str  # H3 Hexadecimal index address
    resolution: int
    orders: List[DeliveryTask] = Field(default_factory=list)
    primary_drivers: List[Driver] = Field(default_factory=list)
    nearby_drivers: List[Driver] = Field(default_factory=list)

    @property
    def order_count(self) -> int:
        return len(self.orders)

    @property
    def center_coord(self) -> Tuple[float, float]:
        """Geographic center (Lat, Lng) of this hexagonal zone."""
        lat, lng = h3.cell_to_latlng(self.zone_id)
        return (lat, lng)


class AdaptiveH3Partitioner:
    """
    Cross-border Density-Aware Adaptive Spatial Partitioning Engine.
    Dynamically splits dense commercial hotspots (e.g., KLCC, Pavilion, Golden Triangle)
    into high-resolution micro-grids (Res 7-8), while keeping sparse suburban/rural zones
    merged at macro-resolutions (Res 5-6) to maintain route feasibility and eliminate orphan tasks.
    """
    def __init__(
        self,
        base_resolution: int = 5,
        max_resolution: int = 8,
        max_orders_per_zone: int = 50,
        min_orders_per_zone: int = 10,
        enable_nearby_driver_fallback: bool = True
    ):
        """
        Initialize the adaptive partitioner.
        :param base_resolution: Starting macroscopic resolution (Res 5: ~8.5km edge).
        :param max_resolution: Finest allowable resolution for hyper-dense clusters (Res 8: ~460m edge).
        :param max_orders_per_zone: Capacity threshold. If orders exceed this, the zone splits into children.
        :param min_orders_per_zone: Sparsity threshold for grouping.
        :param enable_nearby_driver_fallback: Include 1-ring adjacent couriers as auxiliary capacity.
        """
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.max_orders_per_zone = max_orders_per_zone
        self.min_orders_per_zone = min_orders_per_zone
        self.enable_nearby_driver_fallback = enable_nearby_driver_fallback

    def partition_orders(self, orders: List[DeliveryTask]) -> Dict[str, List[DeliveryTask]]:
        """
        Partition arbitrary global orders into density-adaptive H3 cells.
        Recursively splits dense cells while retaining sparse cells at higher parent levels.
        Returns:
            Dictionary mapping H3 cell string (of heterogeneous resolutions) to order lists.
        """
        if not orders:
            return {}

        # 1. Initial coarse aggregation at base_resolution (e.g. Res 5)
        initial_buckets: Dict[str, List[DeliveryTask]] = {}
        for order in orders:
            base_cell = h3.latlng_to_cell(order.pickup.Lat, order.pickup.Lng, self.base_resolution)
            initial_buckets.setdefault(base_cell, []).append(order)

        final_partitions: Dict[str, List[DeliveryTask]] = {}

        # 2. Recursively split dense clusters down to max_resolution
        for cell, task_list in initial_buckets.items():
            self._split_or_finalize(cell, task_list, final_partitions)

        return final_partitions

    def _split_or_finalize(
        self,
        current_cell: str,
        task_list: List[DeliveryTask],
        output_map: Dict[str, List[DeliveryTask]]
    ):
        """Recursive worker for density-based hierarchical cell splitting."""
        current_res = h3.get_resolution(current_cell)

        # Base conditions:
        # - Orders within solver comfort threshold
        # - Or reached maximum allowable resolution
        if len(task_list) <= self.max_orders_per_zone or current_res >= self.max_resolution:
            output_map[current_cell] = task_list
            return

        # Split into next-level child cells (current_res + 1)
        next_res = current_res + 1
        child_buckets: Dict[str, List[DeliveryTask]] = {}

        for task in task_list:
            child_cell = h3.latlng_to_cell(task.pickup.Lat, task.pickup.Lng, next_res)
            child_buckets.setdefault(child_cell, []).append(task)

        # Recursively evaluate each child bucket
        for c_cell, c_tasks in child_buckets.items():
            self._split_or_finalize(c_cell, c_tasks, output_map)

    def build_partition_zones(
        self,
        orders: List[DeliveryTask],
        drivers: List[Driver]
    ) -> List[PartitionZone]:
        """
        Build fully resolved PartitionZone entities with adaptively allocated orders
        and spatially mapped primary/nearby couriers.
        """
        order_partitions = self.partition_orders(orders)
        zones: List[PartitionZone] = []

        for cell_id, zone_orders in order_partitions.items():
            res = h3.get_resolution(cell_id)

            zone = PartitionZone(
                zone_id=cell_id,
                resolution=res,
                orders=zone_orders
            )

            # Map couriers into this adaptive zone based on current GPS location
            for driver in drivers:
                d_cell = h3.latlng_to_cell(driver.current_location.Lat, driver.current_location.Lng, res)

                if d_cell == cell_id:
                    zone.primary_drivers.append(driver)
                elif self.enable_nearby_driver_fallback:
                    # Check if driver is within 1 ring of this adaptive zone
                    try:
                        step_dist = h3.grid_distance(d_cell, cell_id)
                        if step_dist == 1:
                            zone.nearby_drivers.append(driver)
                    except Exception:
                        pass

            zones.append(zone)

        return zones
