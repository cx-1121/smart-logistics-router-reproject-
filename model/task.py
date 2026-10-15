from enum import Enum
import threading
from typing import Optional, Set, Tuple
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from model.node import Node


class TaskStatus(str, Enum):
    PENDING = "PENDING"        # Initial state: waiting for dispatch / broadcast
    OFFERED = "OFFERED"        # Broadcast to candidate drivers for consideration
    LOCKED = "LOCKED"          # Claimed and locked by a driver (preventing duplicate acceptance)
    COMPLETED = "COMPLETED"    # Delivery completed
    CANCELLED = "CANCELLED"    # Cancelled


class DeliveryTask(BaseModel):
    """
    Logistics pickup & delivery task domain entity (PDP Atomic Unit).
    Binds a pickup node and a delivery node together atomically,
    maintaining precedence constraints, payload weight, offering pool,
    and thread-safe atomic lock mechanism.
    """
    id: UUID = Field(default_factory=uuid4)
    task_name: str
    pickup: Node
    delivery: Node
    weight: float = 3.0

    # Task status and assignment tracking
    status: TaskStatus = TaskStatus.PENDING
    assigned_driver_id: Optional[UUID] = None
    candidate_driver_ids: Set[UUID] = Field(default_factory=set)

    # Thread-safe mutex lock for concurrency control
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @model_validator(mode="after")
    def bind_precedence_constraint(self):
        """
        Enforce Pickup-and-Delivery precedence constraint:
        The delivery destination MUST depend on visiting the pickup node first.
        """
        if self.pickup.id not in self.delivery.prerequisites:
            self.delivery.prerequisites.append(self.pickup.id)
        return self

    @property
    def midpoint(self) -> Tuple[float, float]:
        """
        Geographic center coordinate (Lng, Lat) of the task pair.
        Useful for spatial indexing and clustering.
        """
        return (
            (self.pickup.Lng + self.delivery.Lng) / 2.0,
            (self.pickup.Lat + self.delivery.Lat) / 2.0,
        )

    def broadcast_offer(self, driver_id: UUID) -> bool:
        """
        Offer this task to a candidate driver for consideration.
        Can only offer if task is PENDING or already OFFERED to others.
        """
        with self._lock:
            if self.status in (TaskStatus.PENDING, TaskStatus.OFFERED):
                self.candidate_driver_ids.add(driver_id)
                self.status = TaskStatus.OFFERED
                return True
            return False

    def try_claim(self, driver_id: UUID) -> bool:
        """
        Atomically attempt to accept and lock the task for a driver.
        Returns:
            True if successfully claimed and locked.
            False if already claimed/locked by another driver.
        """
        with self._lock:
            if self.status in (TaskStatus.PENDING, TaskStatus.OFFERED):
                self.status = TaskStatus.LOCKED
                self.assigned_driver_id = driver_id
                return True
            return False

    def release_lock(self):
        """Release driver lock and reset task back to PENDING (e.g. if driver cancels)."""
        with self._lock:
            self.status = TaskStatus.PENDING
            self.assigned_driver_id = None
            self.candidate_driver_ids.clear()

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, DeliveryTask):
            return self.id == other.id
        return False
