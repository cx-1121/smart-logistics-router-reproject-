from typing import List
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


class Node(BaseModel):
    """
    Logistics node representation (e.g., depot, pickup point, delivery point).
    """
    id: UUID = Field(default_factory=uuid4)
    name: str
    Lng: float  # Longitude coordinate
    Lat: float  # Latitude coordinate
    prerequisites: List[UUID] = Field(default_factory=list)  # Precedence dependencies: must visit these nodes first