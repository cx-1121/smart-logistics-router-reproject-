from enum import Enum
import threading
from typing import List, Optional, Set
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, PrivateAttr
from model.node import Node
from model.task import DeliveryTask


class DriverStatus(str, Enum):
    IDLE = "IDLE"          # Available and stationary
    BUSY = "BUSY"          # In transit / delivering active orders
    OFFLINE = "OFFLINE"    # Offline / on break


class Driver(BaseModel):
    """
    Logistics driver / courier domain entity.
    Maintains physical GPS location, capacity constraints, working status,
    offered task pool for independent acceptance, active claimed tasks,
    and the assigned route itinerary planned by optimization engines.
    Supports in-transit order stacking and dynamic on-the-fly re-routing.
    """
    id: UUID = Field(default_factory=uuid4)
    name: str
    current_location: Node  # Driver's real-time physical/GPS location (acts as routing origin)
    capacity: float = 100.0  # Maximum carry/weight capacity (e.g. kg or volume)
    current_load: float = 0.0  # Current onboard load
    status: DriverStatus = DriverStatus.IDLE

    # Task offering & decision pools
    offered_tasks: List[DeliveryTask] = Field(default_factory=list)  # Broadcast offers pending driver decision
    active_tasks: List[DeliveryTask] = Field(default_factory=list)   # Tasks successfully claimed and locked

    # Final execution route itinerary
    assigned_route: List[Node] = Field(default_factory=list)  # Ordered stops assigned by solver
    total_route_cost: float = 0.0

    # In-transit re-routing bookkeeping
    needs_reroute: bool = False  # Set when the active task set changes; cleared once a new route is assigned
    picked_up_task_ids: Set[UUID] = Field(default_factory=set)  # Tasks whose pickup is done but delivery pending

    # Guards offer/claim bookkeeping when the same courier is shared by concurrently dispatched zones
    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)

    @property
    def is_available(self) -> bool:
        """
        Check whether the driver is eligible to receive and accept new tasks.
        In-transit couriers (BUSY) with spare payload capacity ARE permitted to accept new tasks.
        Only OFFLINE couriers or couriers with zero remaining capacity are ineligible.
        """
        return self.status != DriverStatus.OFFLINE and self.remaining_capacity > 0

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

    def receive_offer(self, task: DeliveryTask) -> bool:
        """
        Receive a broadcast order offer from dispatch matcher.
        In-transit drivers are eligible as long as vehicle payload permits.
        """
        with self._lock:
            if task in self.offered_tasks or task in self.active_tasks:
                return False
            if not self.can_carry(task.weight):
                return False

            offered = task.broadcast_offer(self.id)
            if not offered:
                return False

            self.offered_tasks.append(task)
            return True

    def accept_task(self, task: DeliveryTask) -> bool:
        """
        Driver decides to ACCEPT an offered task (even while in-transit / BUSY).
        Performs atomic lock check:
        - If claimed successfully, locks the task, updates vehicle cargo load, and moves to active_tasks.
        - If task was already claimed by another driver, claim fails and task is removed from offers.
        """
        # Capacity check and cargo load must be atomic per courier, otherwise two zones can overload one vehicle
        with self._lock:
            # Validate vehicle capacity limit
            if not self.can_carry(task.weight):
                return False

            # Attempt atomic claim on the task
            claimed = task.try_claim(self.id)
            if claimed:
                self.load_cargo(task.weight)
                self.active_tasks.append(task)
                self.needs_reroute = True
                if task in self.offered_tasks:
                    self.offered_tasks.remove(task)
                return True
            else:
                # Task was already claimed and locked by another competing driver
                if task in self.offered_tasks:
                    self.offered_tasks.remove(task)
                return False

    def reject_task(self, task: DeliveryTask):
        """Driver declines an offered task. Removes it from driver's candidate pool."""
        with self._lock:
            if task in self.offered_tasks:
                self.offered_tasks.remove(task)

    def advance_to_next_stop(self) -> Optional[Node]:
        """
        Simulate the courier progressing to the next scheduled stop along their route.
        Updates current_location to this stop, pops it from assigned_route,
        and automatically unloads delivered parcel weight.
        """
        if not self.assigned_route:
            return None

        # Pop current immediate stop
        completed_stop = self.assigned_route.pop(0)
        self.current_location = completed_stop

        for task in list(self.active_tasks):
            # Pickup done: re-routing must not send the courier back to this stop
            if task.pickup.id == completed_stop.id:
                self.picked_up_task_ids.add(task.id)
            # Delivery done: task is fulfilled and its cargo leaves the vehicle
            if task.delivery.id == completed_stop.id:
                self.unload_cargo(task.weight)
                self.active_tasks.remove(task)
                self.picked_up_task_ids.discard(task.id)

        # Reset to IDLE if all stops and tasks are exhausted
        if not self.assigned_route and not self.active_tasks:
            self.status = DriverStatus.IDLE

        return completed_stop

    def get_pending_routing_nodes(self) -> List[Node]:
        """
        Collect all distinct nodes currently requiring visits for ACO solver:
        Driver's current location (origin) + all pending pickups and deliveries of active tasks.
        For tasks already picked up, only the delivery remains, with its satisfied pickup
        prerequisite stripped (same node id, so delivery completion is still matched).
        """
        nodes = [self.current_location]
        visited_node_ids = {self.current_location.id}
        for task in self.active_tasks:
            picked_up = task.id in self.picked_up_task_ids
            if not picked_up and task.pickup.id not in visited_node_ids:
                nodes.append(task.pickup)
                visited_node_ids.add(task.pickup.id)
            if task.delivery.id not in visited_node_ids:
                delivery = task.delivery
                if picked_up:
                    delivery = delivery.model_copy(update={
                        "prerequisites": [p for p in delivery.prerequisites if p != task.pickup.id]
                    })
                nodes.append(delivery)
                visited_node_ids.add(task.delivery.id)
        return nodes

    def assign_route(self, route: List[Node], route_cost: float = 0.0):
        """
        Assign an optimized route itinerary computed by the routing engine (e.g., ACO).
        Automatically transitions driver status to BUSY when non-empty.
        """
        self.assigned_route = list(route)
        self.total_route_cost = route_cost
        self.needs_reroute = False
        if self.assigned_route:
            self.status = DriverStatus.BUSY

    def clear_route(self):
        """Clear completed itinerary and reset status to IDLE."""
        self.assigned_route.clear()
        self.active_tasks.clear()
        self.offered_tasks.clear()
        self.current_load = 0.0
        self.total_route_cost = 0.0
        self.needs_reroute = False
        self.picked_up_task_ids.clear()
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
