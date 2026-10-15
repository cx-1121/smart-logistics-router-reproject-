import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ACOParams:
    num_ants: int = 30
    max_iter: int = 100
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    q: float = 100.0
    initial_radius: float = 30.0
    radius_step: float = 10.0
    max_radius: float = 200.0


@dataclass
class ACOProblem:
    """
    Index-based, picklable ACO problem: plain numpy arrays only (no Node/UUID objects),
    so it can be shipped to a worker process and solved outside the GIL.
    Node i of every matrix corresponds to position i of the caller's node list.
    """
    dist: np.ndarray              # (N, N) physical distance in km, for spatial neighborhood pruning
    cost: np.ndarray              # (N, N) generalized cost, for heuristic eta = 1/cost
    pheromone: np.ndarray         # (N, N) initial pheromone intensities
    prereq_count: np.ndarray      # (N,) number of prerequisites each node must wait for
    dependents: np.ndarray        # (N, N) int: dependents[v, j] = 1 if visiting v releases one prerequisite of j
    eligible_starts: np.ndarray   # (K,) node indices ants may start from
    params: ACOParams
    initial_path: Optional[np.ndarray] = None  # Warm-start closed-loop path as node indices
    seed: Optional[int] = None


@dataclass
class ACOResult:
    best_path: List[int]          # Closed-loop path as node indices; empty if no ant completed a tour
    best_cost: float
    pheromone: np.ndarray         # Final pheromone matrix
    elapsed_ms: float


def _deposit(tau: np.ndarray, paths: np.ndarray, costs: np.ndarray, q: float):
    """Ant-cycle deposit: every edge of path k receives q / costs[k] (repeated edges accumulate)."""
    if len(paths) == 0:
        return
    delta = np.repeat(q / costs, paths.shape[1] - 1)
    np.add.at(tau, (paths[:, :-1].ravel(), paths[:, 1:].ravel()), delta)


def solve_aco(problem: ACOProblem) -> ACOResult:
    """
    Vectorized ACO: all ants of a generation advance one step together on (ants x nodes) arrays.
    Pheromone is only updated between generations, so this is equivalent to building the ants one
    after another. Per step, each ant:
      1. keeps nodes that are unvisited and whose prerequisites have all been visited;
      2. prunes them to the smallest radius (initial_radius + k * radius_step <= max_radius) that
         contains at least one candidate, falling back to all valid candidates beyond max_radius;
      3. picks the next node by roulette wheel on tau^alpha * (1/cost)^beta.
    """
    t_start = time.perf_counter()
    p = problem.params
    rng = np.random.default_rng(problem.seed)
    n = problem.dist.shape[0]
    a = p.num_ants

    tau = problem.pheromone.astype(float, copy=True)
    dist = problem.dist
    cost = problem.cost
    # Diagonal (zero-cost self loops) gets eta = 0; those nodes are always visited anyway
    eta_beta = (1.0 / np.where(cost > 0, cost, np.inf)) ** p.beta
    dependents = problem.dependents
    ants = np.arange(a)

    best_path: Optional[np.ndarray] = None
    best_cost = float("inf")

    # Warm start: seed the global best and pre-reinforce its edges
    init = problem.initial_path
    if init is not None and len(init) >= 2:
        best_path = np.asarray(init, dtype=np.int64)
        best_cost = float(cost[best_path[:-1], best_path[1:]].sum())
        _deposit(tau, best_path[None, :], np.array([best_cost]), p.q * 1.5)

    for _ in range(p.max_iter):
        start = rng.choice(problem.eligible_starts, size=a)
        cur = start.copy()
        visited = np.zeros((a, n), dtype=bool)
        visited[ants, start] = True
        unmet = np.broadcast_to(problem.prereq_count, (a, n)) - dependents[start]
        paths = np.empty((a, n + 1), dtype=np.int64)
        paths[:, 0] = start
        totals = np.zeros(a)
        active = np.ones(a, dtype=bool)

        for step in range(1, n):
            valid = ~visited & (unmet <= 0) & active[:, None]
            active = valid.any(axis=1)
            if not active.any():
                break

            # Adaptive radius: smallest initial_radius + k * radius_step covering the nearest candidate
            d = np.where(valid, dist[cur], np.inf)
            nearest = d.min(axis=1)
            k = np.ceil(np.maximum(nearest - p.initial_radius, 0.0) / p.radius_step - 1e-9)
            radius = p.initial_radius + k * p.radius_step
            beyond = radius > p.max_radius + 1e-9
            avail = valid & ((d <= radius[:, None] + 1e-9) | beyond[:, None])

            # Roulette wheel on tau^alpha * eta^beta
            weights = np.where(avail, tau[cur] ** p.alpha * eta_beta[cur], 0.0)
            zero = weights.sum(axis=1) <= 0
            if zero.any():
                weights[zero] = avail[zero].astype(float)
            cum = np.cumsum(weights, axis=1)
            pick = (1.0 - rng.random(a)) * cum[:, -1]
            nxt = np.minimum((cum < pick[:, None]).sum(axis=1), n - 1)

            moving = np.flatnonzero(active)
            mv_next = nxt[moving]
            totals[moving] += cost[cur[moving], mv_next]
            visited[moving, mv_next] = True
            unmet[moving] -= dependents[mv_next]
            paths[moving, step] = mv_next
            cur[moving] = mv_next

        # Only ants that serviced every node close the circuit back to their origin
        complete = visited.all(axis=1)
        done = np.flatnonzero(complete)
        if len(done):
            totals[done] += cost[cur[done], start[done]]
            paths[done, n] = start[done]
            done_paths, done_costs = paths[done], totals[done]

            i_best = int(done_costs.argmin())
            if done_costs[i_best] < best_cost:
                best_cost = float(done_costs[i_best])
                best_path = done_paths[i_best].copy()
        else:
            done_paths, done_costs = paths[:0], totals[:0]

        tau *= (1.0 - p.rho)
        _deposit(tau, done_paths, done_costs, p.q)
        if best_path is not None:
            _deposit(tau, best_path[None, :], np.array([best_cost]), p.q * 1.5)

    return ACOResult(
        best_path=best_path.tolist() if best_path is not None else [],
        best_cost=best_cost,
        pheromone=tau,
        elapsed_ms=(time.perf_counter() - t_start) * 1000,
    )


def solve_aco_path(problem: ACOProblem) -> Tuple[List[int], float, float]:
    """Process-pool entry point: returns only (path indices, cost, solve ms) to keep IPC payloads small."""
    result = solve_aco(problem)
    return result.best_path, result.best_cost, result.elapsed_ms
