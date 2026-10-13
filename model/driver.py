from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from model.node import Node


class DriverStatus(str, Enum):
    IDLE = "IDLE"          # 待命中，可接单
    BUSY = "BUSY"          # 履约配送中
    OFFLINE = "OFFLINE"    # 离线/休息中


class Driver(BaseModel):
    """
    Logistics driver / courier domain entity.
    Maintains physical GPS location, capacity constraints, working status,
    and the assigned route itinerary planned by optimization engines.
    """
    id: UUID = Field(default_factory=uuid4)
    name: str
    current_location: Node  # Driver's real-time physical/GPS location (acts as routing origin)
    capacity: float = 100.0  # Maximum carry/weight capacity (e.g. kg or volume)
    current_load: float = 0.0  # Current onboard load
    status: DriverStatus = DriverStatus.IDLE
    assigned_route: List[Node] = Field(default_factory=list)  # Ordered stops assigned by solver
    total_route_cost: float = 0.0

    @property
    def is_available(self) -> bool:
        """Check whether the driver is currently online and eligible for dispatch."""
        return self.status == DriverStatus.IDLE

    @property
    def remaining_capacity(self) -> float:
        """Compute the remaining payload capacity on the vehicle."""
        return max(0.0, self.capacity - self.current_load)

    @property
    def next_stop(self) -> Optional[Node]:
        """Peek at the immediate next stop along the assigned route itinerary."""
        if self.assigned_route:
            return self.assigned_route[0]
        return None

    def can_carry(self, weight: float) -> bool:
        """Check if vehicle has sufficient spare capacity for the given weight."""
        return self.current_load + weight <= self.capacity

    def load_cargo(self, weight: float):
        """Load cargo into vehicle, updating onboard weight."""
        if not self.can_carry(weight):
            raise ValueError(
                f"Driver '{self.name}' capacity exceeded: {self.current_load} + {weight} > {self.capacity}"
            )
        self.current_load += weight

    def unload_cargo(self, weight: float):
        """Unload cargo upon delivery destination."""
        self.current_load = max(0.0, self.current_load - weight)

    def assign_route(self, route: List[Node], route_cost: float = 0.0):
        """
        Assign an optimized route itinerary computed by the routing engine (e.g., ACO).
        Automatically transitions driver status to BUSY when non-empty.
        """
        self.assigned_route = list(route)
        self.total_route_cost = route_cost
        if self.assigned_route:
            self.status = DriverStatus.BUSY

    def clear_route(self):
        """Clear completed itinerary and reset status to IDLE."""
        self.assigned_route.clear()
        self.total_route_cost = 0.0
        self.status = DriverStatus.IDLE

    def update_location(self, new_location: Node):
        """Update driver's current real-time GPS coordinate node."""
        self.current_location = new_location

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Driver):
            return self.id == other.id
        return False

