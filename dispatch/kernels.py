"""
Numba kernels for exact route pricing.

A route is an open path that starts at the courier's position and visits n stops. Stop i is a
matrix node (stop_node), changes the on-board load by stop_dw (pickup +w, delivery -w), and may
require another stop of the same route to come first (stop_req, -1 for none: pickups and
deliveries of orders already on board). The kernel finds the cheapest order of the stops that
respects precedence and keeps the load within capacity, by DP over (visited set, last stop).
Only precedence-feasible sets are ever reached, so a route with b on-board deliveries and r
pickup-delivery pairs touches at most 2^b * 3^r sets instead of 2^n.
"""
import numpy as np
from numba import get_num_threads, njit, prange

MAX_ROUTE_STOPS = 16
_EPS = 1e-9


@njit(cache=True)
def _solve_route(courier, n, s0, stop_node, stop_req, stop_dw, start_load, cap,
                 node_dist, courier_dist, dp, parent, load, reach, local, want_path, path_out):
    """
    DP over (visited set, last stop). dp/parent/load/reach are scratch buffers of at least
    (2^n, n) / 2^n entries; dp must hold inf and reach False on entry, and both are restored
    before returning, so a buffer can be reused across columns without a full reset.
    local is an (n, n) scratch for the stop-to-stop distances, gathered once so the hot loop
    reads a small contiguous block instead of scattered entries of the global matrix.
    """
    if n == 0:
        return 0.0
    size = 1 << n
    for i in range(n):
        row = stop_node[s0 + i]
        for j in range(n):
            local[i, j] = node_dist[row, stop_node[s0 + j]]

    for i in range(n):
        if stop_req[s0 + i] >= 0:
            continue
        new_load = start_load + stop_dw[s0 + i]
        if new_load > cap + _EPS:
            continue
        m = 1 << i
        dp[m, i] = courier_dist[courier, stop_node[s0 + i]]
        parent[m, i] = -1
        reach[m] = True
        load[m] = new_load

    for mask in range(1, size):
        if not reach[mask]:
            continue
        mask_load = load[mask]
        for last in range(n):
            base = dp[mask, last]
            if base == np.inf:
                continue
            for nxt in range(n):
                bit = 1 << nxt
                if mask & bit:
                    continue
                req = stop_req[s0 + nxt]
                if req >= 0 and not (mask >> req) & 1:
                    continue
                new_load = mask_load + stop_dw[s0 + nxt]
                if new_load > cap + _EPS:
                    continue
                new_mask = mask | bit
                value = base + local[last, nxt]
                if value < dp[new_mask, nxt]:
                    dp[new_mask, nxt] = value
                    parent[new_mask, nxt] = last
                    reach[new_mask] = True
                    load[new_mask] = new_load

    full = size - 1
    best = np.inf
    best_last = -1
    for last in range(n):
        if dp[full, last] < best:
            best = dp[full, last]
            best_last = last
    if want_path and best_last >= 0:
        mask = full
        cur = best_last
        for pos in range(n - 1, -1, -1):
            path_out[pos] = cur
            prev = parent[mask, cur]
            mask ^= 1 << cur
            cur = prev

    # Restore the scratch invariant (only reached sets were written)
    for mask in range(1, size):
        if reach[mask]:
            reach[mask] = False
            for last in range(n):
                dp[mask, last] = np.inf
    return best


@njit(parallel=True, cache=True)
def route_costs(col_courier, col_ptr, stop_node, stop_req, stop_dw,
                courier_load, courier_cap, node_dist, courier_dist, n_workers):
    """
    Optimal route cost of every column (inf if no capacity-feasible order exists).
    Column c owns stops col_ptr[c]:col_ptr[c+1]; stop_req holds indices local to the column.
    Worker w prices columns w, w + n_workers, ... with its own scratch buffers, which balances
    load (columns of one courier are contiguous and similar in size) and allocates once per worker.
    """
    n_cols = col_courier.shape[0]
    out = np.empty(n_cols)
    n_max = 1
    for c in range(n_cols):
        n_max = max(n_max, col_ptr[c + 1] - col_ptr[c])
    size = 1 << n_max
    no_path = np.empty(0, dtype=np.int64)
    for w in prange(n_workers):
        dp = np.full((size, n_max), np.inf)
        parent = np.empty((size, n_max), dtype=np.int8)
        load = np.empty(size)
        reach = np.zeros(size, dtype=np.bool_)
        local = np.empty((n_max, n_max))
        for c in range(w, n_cols, n_workers):
            k = col_courier[c]
            out[c] = _solve_route(k, col_ptr[c + 1] - col_ptr[c], col_ptr[c], stop_node, stop_req, stop_dw,
                                  courier_load[k], courier_cap[k], node_dist, courier_dist,
                                  dp, parent, load, reach, local, False, no_path)
    return out


@njit(cache=True)
def route_path(courier, stop_node, stop_req, stop_dw, start_load, cap, node_dist, courier_dist):
    """Optimal route for one stop set: (cost, stop indices in visiting order)."""
    n = stop_node.shape[0]
    path = np.empty(n, dtype=np.int64)
    size = 1 << max(n, 1)
    dp = np.full((size, max(n, 1)), np.inf)
    parent = np.empty((size, max(n, 1)), dtype=np.int8)
    load = np.empty(size)
    reach = np.zeros(size, dtype=np.bool_)
    local = np.empty((max(n, 1), max(n, 1)))
    cost = _solve_route(courier, n, 0, stop_node, stop_req, stop_dw, start_load, cap,
                        node_dist, courier_dist, dp, parent, load, reach, local, True, path)
    return cost, path


@njit(cache=True)
def best_pair_insertion(courier, plan_node, plan_dw, start_load, cap, p_node, d_node, weight,
                        node_dist, courier_dist):
    """
    Cheapest capacity-feasible insertion of a pickup/delivery pair into an open itinerary.
    The pickup goes before plan stop i and the delivery before plan stop j (i <= j; index m means
    after the last stop). Returns (delta_km, i, j), with delta_km = inf if no slot is feasible.
    """
    m = plan_node.shape[0]
    loads = np.empty(m + 1)  # loads[k]: on-board weight after the first k plan stops
    loads[0] = start_load
    for k in range(m):
        loads[k + 1] = loads[k] + plan_dw[k]

    best = np.inf
    best_i = -1
    best_j = -1
    for i in range(m + 1):
        if i == 0:
            to_p = courier_dist[courier, p_node]
        else:
            to_p = node_dist[plan_node[i - 1], p_node]
        # Removing edge (i-1 -> i) once the pickup sits in slot i
        if i < m:
            if i == 0:
                old_i = courier_dist[courier, plan_node[0]]
            else:
                old_i = node_dist[plan_node[i - 1], plan_node[i]]
        else:
            old_i = 0.0

        running_max = loads[i]
        for j in range(i, m + 1):
            if j > i:
                running_max = max(running_max, loads[j])
            if running_max + weight > cap + _EPS:
                break  # Loads only grow along the window, so a wider window stays infeasible
            if j == i:
                delta = to_p + node_dist[p_node, d_node]
                if i < m:
                    delta += node_dist[d_node, plan_node[i]] - old_i
            else:
                delta = to_p + node_dist[p_node, plan_node[i]] - old_i
                delta += node_dist[plan_node[j - 1], d_node]
                if j < m:
                    delta += node_dist[d_node, plan_node[j]] - node_dist[plan_node[j - 1], plan_node[j]]
            if delta < best:
                best = delta
                best_i = i
                best_j = j
    return best, best_i, best_j


def default_workers() -> int:
    """Pricing workers: a few per thread so uneven columns still spread across threads."""
    return 4 * get_num_threads()


def warm_up():
    """Trigger JIT compilation (or cache load) so the first timed batch measures solving only."""
    node_dist = np.zeros((4, 4))
    courier_dist = np.zeros((1, 4))
    stop_node = np.array([0, 1], dtype=np.int64)
    stop_req = np.array([-1, 0], dtype=np.int64)
    stop_dw = np.array([1.0, -1.0])
    route_costs(np.zeros(1, dtype=np.int64), np.array([0, 2], dtype=np.int64), stop_node, stop_req,
                stop_dw, np.zeros(1), np.full(1, 10.0), node_dist, courier_dist, default_workers())
    route_path(0, stop_node, stop_req, stop_dw, 0.0, 10.0, node_dist, courier_dist)
    best_pair_insertion(0, np.array([2, 3], dtype=np.int64), np.array([1.0, -1.0]), 0.0, 10.0,
                        0, 1, 1.0, node_dist, courier_dist)
