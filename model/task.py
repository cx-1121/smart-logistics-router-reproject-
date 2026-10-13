from typing import Tuple
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, model_validator
from model.node import Node


class DeliveryTask(BaseModel):
    """
    Logistics pickup & delivery task domain entity (PDP Atomic Unit).
    Binds a pickup node and a delivery node together atomically,
    maintaining precedence constraints, payload weight, and spatial metrics.
    """
    id: UUID = Field(default_factory=uuid4)
    task_name: str
    pickup: Node
    delivery: Node
    weight: float = 3.0

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

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, DeliveryTask):
            return self.id == other.id
        return False
