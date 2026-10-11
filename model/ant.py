from typing import List, Set, Callable
from uuid import UUID
from pydantic import BaseModel, Field, model_validator
from model.node import Node


class Ant(BaseModel):
    start_node: Node
    path: List[Node] = Field(default_factory=list)
    total_cost: float = 0.0
    visited_nodes: List[Node] = Field(default_factory=list)
    visited_ids: Set[UUID] = Field(default_factory=set) # O(1) check is visited

    @model_validator(mode="after")
    def init_ant(self):
        if not self.path:
            self.path = [self.start_node]
        if not self.visited_nodes:
            self.visited_nodes = [self.start_node]
        if not self.visited_ids:
            self.visited_ids = {self.start_node.id}
        return self

    @property
    def current_node(self) -> Node:
        return self.path[-1]

    def get_available_nodes(
        self,
        all_nodes: List[Node],
        get_distance_func: Callable[[Node, Node], float],
        initial_radius: float,
        radius_step: float,
        max_radius: float
    ) -> List[Node]:
        valid_candidates = []
        for node in all_nodes:
            if node.id in self.visited_ids:
                continue
            if node.prerequisites:
                if not all(pre_id in self.visited_ids for pre_id in node.prerequisites):
                    continue

            valid_candidates.append(node)

        if not valid_candidates:
            return []

        current_radius = initial_radius
        available = []

        while not available and current_radius <= max_radius:
            for node in valid_candidates:
                dist = get_distance_func(self.current_node, node)
                if dist <= current_radius:
                    available.append(node)

            if not available:
                current_radius += radius_step

        return available if available else valid_candidates


    def move_to(self, next_node: Node, cost: float):
        self.path.append(next_node)
        self.visited_nodes.append(next_node)
        self.visited_ids.add(next_node.id)
        self.total_cost += cost
