from typing import List, Dict, Optional
import h3
from model.node import Node


class SpatialIndexer:
    def __init__(self, resolution: int = 8):
        """
        :param resolution: H3 grid resolution (default=8: edge length ~460m, area ~0.7 km^2)
        """
        self.resolution = resolution
        # Hash bucket: { h3_cell_id: [Node, Node, ...] }
        self._grid_buckets: Dict[str, List[Node]] = {}

    def clear(self):
        """Reset internal spatial index buckets."""
        self._grid_buckets.clear()

    def add_node(self, node: Node) -> str:
        """
        Index a single logistics node into its corresponding H3 hexagonal cell.
        :return: Computed H3 cell string identifier
        """
        cell_id = h3.latlng_to_cell(node.Lat, node.Lng, self.resolution)
        if cell_id not in self._grid_buckets:
            self._grid_buckets[cell_id] = []
        self._grid_buckets[cell_id].append(node)
        return cell_id

    def build_index(self, nodes: List[Node]):
        """Bulk index a collection of logistics nodes."""
        self.clear()
        for node in nodes:
            self.add_node(node)

    def get_node_cell(self, node: Node) -> str:
        """Compute the H3 cell ID for a given node."""
        return h3.latlng_to_cell(node.Lat, node.Lng, self.resolution)

    def find_nearby_nodes(
        self,
        origin_node: Node,
        max_k_ring: int = 2,
        min_candidates: int = 1
    ) -> List[Node]:
        """
        Find candidate nodes within concentric hexagonal rings around the origin node.
        Supports adaptive radius expansion: starts at ring 1, and gradually expands up to max_k_ring
        if candidates are insufficient.

        :param origin_node: The reference center (e.g. driver's current position)
        :param max_k_ring: Maximum allowable concentric ring expansion depth
        :param min_candidates: Minimum number of candidates to trigger expansion stop
        :return: List of candidate logistics nodes within spatial neighborhood
        """
        origin_cell = self.get_node_cell(origin_node)
        candidates: List[Node] = []
        visited_cells = set()

        # Adaptive expansion loop: ring k = 0, 1, 2, ... up to max_k_ring
        for k in range(max_k_ring + 1):
            ring_cells = h3.grid_disk(origin_cell, k)
            # Find cells newly introduced in this expansion layer
            new_cells = set(ring_cells) - visited_cells

            for cell in new_cells:
                if cell in self._grid_buckets:
                    candidates.extend(self._grid_buckets[cell])

            visited_cells.update(new_cells)

            # Early stopping: if we gathered sufficient candidates, avoid scanning further rings
            if len(candidates) >= min_candidates:
                break

        return candidates
