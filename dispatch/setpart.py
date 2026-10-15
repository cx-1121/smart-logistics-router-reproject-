"""
Exact solver for the batch set-partitioning problem

    min c^T z   s.t.  A z = 1,  z >= 0,  z_j in {0, 1} for column variables (j < n_int)

where every column of A has a 1 in some row, so z <= 1 is implied and needs no explicit bound.
The first n_groups rows are group rows (one per courier, covered by exactly one binary column);
the rest are item rows (one per order, covered by a binary column or its continuous slack).

Strategy (all LPs share one HiGHS instance, so every re-solve is warm-started from the last basis):
  1. LP relaxation by column generation over the explicit column pool: solve on a working set,
     price all columns with one sparse mat-vec, add those with negative reduced cost, repeat.
     The final LP value is the lower bound LB over the full pool.
  2. Reduced-cost fixing: any integer solution costs LB + sum_j rc_j z_j with rc >= 0, so a column
     with rc_j > UB - LB cannot be in a solution at least as good as UB. Branch-and-bound runs on
     the columns with rc <= tau; if the best solution found has UB - LB <= tau, it is optimal over
     the whole pool, otherwise tau grows to UB - LB and the search repeats.
  3. Depth-first branch-and-bound with assignment branching (Ryan-Foster on a group row and an
     item row): take the (group g, item o) pair whose LP share is closest to 0.5 and branch on
     "g serves o" (columns of g without o, and other columns with o, fixed to 0) versus "g does not
     serve o" (columns of g with o fixed to 0). The rule is balanced and complete: when every share
     is 0/1, each group has exactly one positive column. Nodes whose LP value cannot beat the
     incumbent are pruned. If the node budget runs out, HiGHS' own MIP finishes the restricted problem.
"""
import time
from dataclasses import dataclass
from typing import Optional

import highspy
import numpy as np
from scipy.sparse import csc_matrix

_INF = highspy.kHighsInf


@dataclass
class SetPartitionResult:
    z: np.ndarray            # Solution over all columns of the pool
    objective: float
    lower_bound: float       # LP relaxation over the full pool
    proved_optimal: bool
    lp_ms: float
    bb_ms: float
    lp_rounds: int
    bb_nodes: int
    fixing_rounds: int
    restricted_columns: int
    used_fallback: bool


class _Model:
    """
    One HiGHS LP whose columns are a growing subset of the pool. Columns are fixed to zero through
    reference counts, so nested branching fixings and the reduced-cost window compose and unwind
    in any order.
    """
    def __init__(self, n_rows: int, n_pool: int):
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.setOptionValue("simplex_strategy", 1)  # Dual simplex: best for re-solves after bound changes
        ones = np.ones(n_rows)
        self.h.addRows(n_rows, ones, ones, 0, np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0))
        self.pool_index = np.zeros(0, dtype=np.int64)         # Model column -> pool column
        self.model_pos = np.full(n_pool, -1, dtype=np.int64)  # Pool column -> model column (-1 absent)
        self.fix_count = np.zeros(0, dtype=np.int64)

    def add(self, a: csc_matrix, cost: np.ndarray, cols: np.ndarray):
        if len(cols) == 0:
            return
        sub = a[:, cols]
        self.h.addCols(len(cols), cost[cols], np.zeros(len(cols)), np.full(len(cols), _INF),
                       sub.nnz, sub.indptr[:-1].astype(np.int32), sub.indices.astype(np.int32), sub.data)
        self.model_pos[cols] = len(self.pool_index) + np.arange(len(cols))
        self.pool_index = np.concatenate([self.pool_index, cols])
        self.fix_count = np.concatenate([self.fix_count, np.zeros(len(cols), dtype=np.int64)])

    def solve(self) -> Optional[float]:
        self.h.run()
        if self.h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return None
        return self.h.getInfo().objective_function_value

    def values(self) -> np.ndarray:
        return np.asarray(self.h.getSolution().col_value)

    def duals(self) -> np.ndarray:
        return np.asarray(self.h.getSolution().row_dual)

    def fix_zero(self, cols: np.ndarray):
        """Fix model columns to zero (stackable)."""
        if len(cols) == 0:
            return
        newly = cols[self.fix_count[cols] == 0]
        self.fix_count[cols] += 1
        self._set_upper(newly, 0.0)

    def unfix(self, cols: np.ndarray):
        """Undo one fix_zero on the same columns."""
        if len(cols) == 0:
            return
        self.fix_count[cols] -= 1
        self._set_upper(cols[self.fix_count[cols] == 0], _INF)

    def _set_upper(self, cols: np.ndarray, upper: float):
        if len(cols):
            n = len(cols)
            self.h.changeColsBounds(n, cols.astype(np.int32), np.zeros(n), np.full(n, upper))


def solve_set_partition(
    cost: np.ndarray,
    a: csc_matrix,
    n_int: int,
    n_groups: int,
    initial: np.ndarray,
    always: np.ndarray,
    tau: float = 2.0,
    pricing_batch: int = 500,
    node_limit: int = 2000,
    max_fixing_rounds: int = 6,
    fallback_time_limit_s: float = 25.0,
) -> SetPartitionResult:
    """
    :param cost: objective over all pool columns; the first n_int are binary, the rest continuous
    :param a: (rows, columns) 0/1 matrix in CSC form
    :param n_groups: number of leading group rows
    :param initial: bool mask of columns in the first LP working set (must make the LP feasible)
    :param always: bool mask of columns always kept in the branch-and-bound problem; must contain
        a feasible integer solution so every restricted problem is feasible
    """
    n_rows, n_cols = a.shape
    a = a.tocsc()
    a_csr = a.tocsr()
    tol = 1e-7 * max(1.0, float(np.abs(cost).max(initial=1.0)))
    t0 = time.perf_counter()

    model = _Model(n_rows, n_cols)
    in_model = initial.copy()
    model.add(a, cost, np.flatnonzero(in_model))
    lp_rounds = 0
    while True:
        lp_rounds += 1
        lower = model.solve()
        if lower is None:
            raise RuntimeError("Set-partitioning LP relaxation is infeasible")
        reduced = cost - a_csr.T @ model.duals()
        entering = np.flatnonzero(~in_model & (reduced < -tol))
        if len(entering) == 0:
            break
        if len(entering) > pricing_batch:
            entering = entering[np.argpartition(reduced[entering], pricing_batch)[:pricing_batch]]
        in_model[entering] = True
        model.add(a, cost, entering)
    reduced = np.maximum(reduced, 0.0)
    lp_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    is_int = np.zeros(n_cols, dtype=bool)
    is_int[:n_int] = True
    best_obj = np.inf
    best_z: Optional[np.ndarray] = None
    nodes = 0
    used_fallback = False
    fixing_rounds = 0
    proved = False
    restricted = 0
    while True:
        fixing_rounds += 1
        allowed = always | (reduced <= tau + tol)
        restricted = int(allowed.sum())
        missing = np.flatnonzero(allowed & ~in_model)
        in_model[missing] = True
        model.add(a, cost, missing)  # rc >= 0 at the root, so the current basis stays optimal
        outside = np.flatnonzero(~allowed[model.pool_index])
        model.fix_zero(outside)

        found, nodes_used, finished = _branch_and_bound(model, a, a_csr, n_int, n_groups, best_obj, tol,
                                                        node_limit - nodes)
        nodes += nodes_used
        if found is not None:
            best_obj = found[0]
            best_z = np.zeros(n_cols)
            best_z[model.pool_index] = found[1]
        complete = finished
        if not finished:
            used_fallback = True
            best_obj, best_z, complete = _fallback_mip(cost, a, is_int, allowed, best_obj, best_z,
                                                       fallback_time_limit_s)
        model.unfix(outside)

        if not complete:
            break  # The restricted problem itself is unproven; report the incumbent as is
        if best_obj - lower <= tau + tol or allowed.all():
            proved = True
            break
        if fixing_rounds >= max_fixing_rounds:
            break
        tau = best_obj - lower

    if best_z is None:
        raise RuntimeError("No integer solution found for the set-partitioning problem")
    return SetPartitionResult(
        z=best_z,
        objective=float(best_obj),
        lower_bound=float(lower),
        proved_optimal=proved,
        lp_ms=lp_ms,
        bb_ms=(time.perf_counter() - t1) * 1000,
        lp_rounds=lp_rounds,
        bb_nodes=nodes,
        fixing_rounds=fixing_rounds,
        restricted_columns=restricted,
        used_fallback=used_fallback,
    )


def _branch_and_bound(model: _Model, a: csc_matrix, a_csr, n_int: int, n_groups: int,
                      incumbent: float, tol: float, node_budget: int):
    """
    Depth-first branch-and-bound over the model's current columns with assignment branching.
    Returns (best (objective, z in model order) improving on incumbent or None, nodes used,
    finished within budget). Every fixing made here is undone before returning.
    """
    best = None
    best_obj = incumbent
    nodes = 0
    # Frames: None = root, ("fix", cols) = branch to apply, ("undo", cols) = restore on backtrack
    stack = [None]
    while stack:
        frame = stack.pop()
        if frame is not None:
            kind, cols = frame
            if kind == "undo":
                model.unfix(cols)
                continue
            model.fix_zero(cols)
            stack.append(("undo", cols))
        if nodes >= node_budget:
            while stack:
                f = stack.pop()
                if f is not None and f[0] == "undo":
                    model.unfix(f[1])
            return best, nodes, False
        nodes += 1
        value = model.solve()
        if value is None or value >= best_obj - tol:
            continue
        z = model.values()
        branch = _pick_assignment(model, a, z, n_int, n_groups)
        if branch is None:
            best_obj = value
            best = (value, z.copy())
            continue
        g, item_row, share = branch
        serves, not_serves = _assignment_branches(model, a_csr, g, item_row)
        # Depth-first into the likelier side first (pushed last)
        if share >= 0.5:
            stack.append(("fix", not_serves))
            stack.append(("fix", serves))
        else:
            stack.append(("fix", serves))
            stack.append(("fix", not_serves))
    return best, nodes, True


def _pick_assignment(model: _Model, a: csc_matrix, z: np.ndarray, n_int: int, n_groups: int):
    """(group row, item row, LP share) with share closest to 0.5, or None when all shares are 0/1."""
    positive = np.flatnonzero(z > 1e-9)
    pool = model.pool_index[positive]
    binary = pool < n_int
    share = {}
    for j_model, j_pool in zip(positive[binary], pool[binary]):
        rows = a.indices[a.indptr[j_pool]:a.indptr[j_pool + 1]]
        g = int(rows[rows < n_groups][0])
        for r in rows[rows >= n_groups]:
            key = (g, int(r))
            share[key] = share.get(key, 0.0) + z[j_model]
    best_key, best_dist = None, 0.5 - 1e-6
    for key, v in share.items():
        dist = abs(v - 0.5)
        if dist < best_dist:
            best_key, best_dist = key, dist
    if best_key is None:
        return None
    return best_key[0], best_key[1], share[best_key]


def _assignment_branches(model: _Model, a_csr, g: int, item_row: int):
    """
    Model columns to fix to zero in each branch:
    'g serves item'        -> columns of g without the item, and columns of other groups (or the slack) with it
    'g does not serve item' -> columns of g with the item
    """
    def model_cols(row):
        pool = a_csr.indices[a_csr.indptr[row]:a_csr.indptr[row + 1]]
        pos = model.model_pos[pool]
        return pos[pos >= 0]
    of_group = model_cols(g)
    with_item = model_cols(item_row)
    both = np.intersect1d(of_group, with_item, assume_unique=True)
    serves = np.union1d(np.setdiff1d(of_group, both, assume_unique=True),
                        np.setdiff1d(with_item, both, assume_unique=True))
    return serves, both


def _fallback_mip(cost, a, is_int, allowed, incumbent, incumbent_z, time_limit_s):
    """
    HiGHS MIP on the restricted columns, used only when the node budget is exhausted.
    Returns (objective, z, solved to proven optimality).
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    idx = np.flatnonzero(allowed)
    res = milp(cost[idx], integrality=is_int[idx].astype(float), bounds=Bounds(0, 1),
               constraints=LinearConstraint(a[:, idx], 1, 1),
               options={"mip_rel_gap": 0.0, "time_limit": time_limit_s, "disp": False})
    complete = res.status == 0
    if res.x is None or (incumbent_z is not None and res.fun >= incumbent):
        return incumbent, incumbent_z, complete
    z = np.zeros(len(cost))
    z[idx] = res.x
    return float(res.fun), z, complete
