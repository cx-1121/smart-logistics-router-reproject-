import json
import logging
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple
from uuid import UUID
from model.node import Node

logger = logging.getLogger(__name__)


class OSRMClient:
    """
    Dedicated client for communicating with OSRM (Open Source Routing Machine) Table Service.
    Handles network requests, error boundaries, and response transformation into distance/cost maps.
    """
    def __init__(
        self,
        host: str = "http://router.project-osrm.org",
        timeout: float = 5.0,
    ):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def get_matrix(
        self,
        nodes: List[Node],
    ) -> Optional[Tuple[Dict[Tuple[UUID, UUID], float], Dict[Tuple[UUID, UUID], float]]]:
        """
        Query OSRM Table API to retrieve distance (km) and generalized cost matrices for all node pairs.
        Returns:
            Tuple of (distance_map, cost_map) on success, or None on failure/timeout.
        """
        if len(nodes) < 2:
            return {}, {}

        try:
            # Format coordinates as "{lng},{lat}" joined by semicolons
            coords = ";".join([f"{node.Lng:.6f},{node.Lat:.6f}" for node in nodes])
            url = f"{self.host}/table/v1/driving/{coords}?annotations=distance,duration"

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "SmartLogisticsRouter/1.0"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    logger.warning(f"OSRM request returned non-200 status code: {resp.status}")
                    return None
                payload = json.loads(resp.read().decode("utf-8"))

            if payload.get("code") != "Ok":
                logger.warning(f"OSRM payload error code: {payload.get('code')}")
                return None

            distances_matrix = payload.get("distances", [])
            durations_matrix = payload.get("durations", [])

            dist_map: Dict[Tuple[UUID, UUID], float] = {}
            cost_map: Dict[Tuple[UUID, UUID], float] = {}
            n = len(nodes)

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    u = nodes[i]
                    v = nodes[j]

                    # OSRM distance is in meters, convert to kilometers
                    dist_meters = distances_matrix[i][j] if distances_matrix else 0.0
                    duration_sec = durations_matrix[i][j] if durations_matrix else 0.0

                    dist_km = dist_meters / 1000.0 if dist_meters is not None else 0.0
                    duration_min = duration_sec / 60.0 if duration_sec is not None else 0.0

                    # Combine physical distance and travel duration into generalized cost
                    cost = dist_km * 0.7 + duration_min * 0.3

                    dist_map[(u.id, v.id)] = dist_km
                    cost_map[(u.id, v.id)] = max(cost, 0.0001)

            logger.info("Successfully fetched matrix data from OSRM.")
            return dist_map, cost_map

        except Exception as e:
            logger.warning(f"OSRM query encountered exception ({e}).")
            return None

    def get_matrices_concurrent(
        self,
        node_groups: List[List[Node]],
        max_workers: int = 4
    ) -> List[Optional[Tuple[Dict[Tuple[UUID, UUID], float], Dict[Tuple[UUID, UUID], float]]]]:
        """
        Scenario 2: Concurrent multi-threaded I/O query for multiple regional node clusters.
        Utilizes ThreadPoolExecutor to bypass Python GIL during network I/O waiting periods,
        executing multiple HTTP requests in parallel.
        """
        if not node_groups:
            return []

        from concurrent.futures import ThreadPoolExecutor

        actual_workers = min(len(node_groups), max_workers)
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Map get_matrix across all node groups concurrently
            results = list(executor.map(self.get_matrix, node_groups))

        return results

