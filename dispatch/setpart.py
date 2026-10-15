"""
Exact solver for the batch set-partitioning problem

    min c^T z   s.t.  A z = 1,  z >= 0,  z_j in {0, 1} for column variables (j < n_int)

where every column of A has a 1 in some row, so z <= 1 is implied and needs no explicit bound.
The first n_groups rows are group rows (one per courier, covered by exactly one binary column);
the rest are item rows (one per order, covered by a binary column or its continuous slack).

Strategy (all LPs share one HiGHS instance, so every re-solve is warm-started from the last basis):
  1. Root LP by column generation over the explicit column pool (solve on a working set, price
     every column with one sparse mat-vec, add those with negative reduced cost), interleaved with
     clique cuts: columns that pairwise share a row can have at most one at 1, so a set of them
     with LP values summing above 1 yields the valid cut sum z <= 1. Set-partitioning LPs are
     fractional mostly through such odd cycles of columns; on the hard batches a handful of cuts
     closes most of the gap that otherwise costs hundreds of branch-and-bound nodes. A pool column
     outside the model has coefficient 0 in every cut, which keeps the cuts valid for the whole pool.
     The final LP value is the lower bound LB over the full pool.
  2. Reduced-cost fixing: any integer solution costs LB + sum_j rc_j z_j with rc >= 0, so a column
     with rc_j > UB - LB cannot be in a solution at least as good as UB. Branch-and-bound runs on
     the columns with rc <= tau; if the best solution found has UB - LB <= tau, it is optimal over
     the whole pool, otherwise tau grows to UB - LB and the search repeats.
  3. The restricted problem splits into independent blocks (rows linked only through shared
     columns: couriers and orders in different parts of the city). A single search tree over all
     blocks multiplies their sizes, because backtracking in one block re-solves the others; so each
     fractional block is searched on its own, its best solution fixed, and the next block taken.
     Blocks share no rows or columns (cuts too stay inside one block), so the sum of their optima
     is the joint optimum. When the window widens, the previous round's solution seeds each block's
     incumbent.
  4. Depth-first branch-and-bound per block with assignment branching (Ryan-Foster on a group row
     and an item row): take the fractional (group g, item o) pair with the largest LP share and
     branch on "g serves o" (columns of g without o, and other columns with o, fixed to 0) first,
     then "g does not serve o" (columns of g with o fixed to 0). The rule is complete: when every
     share is 0/1, each group has exactly one positive column. Each node separates a few more
     clique cuts; nodes whose LP value cannot beat the incumbent are pruned. If the node budget
     runs out, HiGHS' own MIP finishes the restricted problem.
"""
import time
from dataclasses import dataclass
from typing import Optional

import highspy
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.csgraph import connected_components

_INF = highspy.kHighsInf


@dataclass
class SetPartitionResult:
    z: np.ndarray            # Solution over all columns of the pool
    objective: float
    lower_bound: float       # Root LP (with cuts) over the full pool
    proved_optimal: bool
    lp_ms: float
    bb_ms: float
    lp_rounds: int
    cuts: int
    bb_nodes: int
    fixing_rounds: int
    restricted_columns: int
    used_fallback: bool


class _Model:
    """
    One HiGHS LP whose columns are a growing subset of the pool, plus clique-cut rows. Columns are
    fixed to zero through reference counts, so nested branching fixings and the reduced-cost window
    compose and unwind in any order.
    """
    def __init__(self, n_rows: int, n_pool: int):
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.setOptionValue("simplex_strategy", 1)  # Dual simplex: best for re-solves after bound changes
        ones = np.ones(n_rows)
        self.h.addRows(n_rows, ones, ones, 0, np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0))
        self.n_base_rows = n_rows
        self.cuts = 0
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

    def add_cut(self, cols: np.ndarray):
        """sum of z over the given model columns <= 1."""
        self.h.addRow(-_INF, 1.0, len(cols), cols.astype(np.int32), np.ones(len(cols)))
        self.cuts += 1

    def solve(self) -> Optional[float]:
        self.h.run()
        if self.h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return None
        return self.h.getInfo().objective_function_value

    def values(self) -> np.ndarray:
        return np.asarray(self.h.getSolution().col_value)

    def base_row_duals(self) -> np.ndarray:
        return np.asarray(self.h.getSolution().row_dual)[:self.n_base_rows]

    def reduced_costs(self) -> np.ndarray:
        """Reduced costs of the model's columns, including the cut rows' duals."""
        return np.asarray(self.h.getSolution().col_dual)

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
    tau: float = 0.5,
    pricing_batch: int = 500,
    root_cut_rounds: int = 20,
    node_cut_rounds: int = 3,
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
    cut_rounds = 0
    while True:
        lp_rounds += 1
        lower = model.solve()
        if lower is None:
            raise RuntimeError("Set-partitioning LP relaxation is infeasible")
        reduced = cost - a_csr.T @ model.base_row_duals()  # Exact for columns outside the model (in no cut)
        reduced[model.pool_index] = model.reduced_costs()
        entering = np.flatnonzero(~in_model & (reduced < -tol))
        if len(entering):
            if len(entering) > pricing_batch:
                entering = entering[np.argpartition(reduced[entering], pricing_batch)[:pricing_batch]]
            in_model[entering] = True
            model.add(a, cost, entering)
            continue
        if cut_rounds < root_cut_rounds and _separate_cliques(model, a, None) > 0:
            cut_rounds += 1
            continue
        break
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

        block_of_row, block_of_col = _blocks(a, allowed, n_int)
        found, nodes_used, finished = _solve_by_blocks(model, a, a_csr, n_int, n_groups, block_of_row,
                                                       block_of_col, best_z, tol, node_cut_rounds,
                                                       node_limit - nodes)
        nodes += nodes_used
        if found is not None and found[0] < best_obj:
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
        cuts=model.cuts,
        bb_nodes=nodes,
        fixing_rounds=fixing_rounds,
        restricted_columns=restricted,
        used_fallback=used_fallback,
    )


def _separate_cliques(model: _Model, a: csc_matrix, only: Optional[np.ndarray], max_cuts: int = 100) -> int:
    """
    Greedy clique separation on the current LP solution. Columns conflict when they share a row;
    starting from each positive column (largest value first), greedily grow a set of pairwise
    conflicting positive columns and add sum z <= 1 when it is violated and not already one of the
    rows (no single row is common to all members). only (bool per model column) limits candidates.
    Returns the number of cuts added.
    """
    z = model.values()
    positive = z > 1e-6
    if only is not None:
        positive &= only
    pos = np.flatnonzero(positive)
    if len(pos) < 3:
        return 0
    sub = a[:, model.pool_index[pos]]
    conflict = (sub.T @ sub).toarray() > 0
    np.fill_diagonal(conflict, False)
    order = np.argsort(-z[pos], kind="stable")
    seen = set()
    added = 0
    for start in order:
        members = [start]
        candidates = conflict[start].copy()
        for j in order:
            if candidates[j]:
                members.append(j)
                candidates &= conflict[j]
        if len(members) < 3 or z[pos[members]].sum() <= 1.0 + 1e-6:
            continue
        key = tuple(sorted(members))
        if key in seen:
            continue
        seen.add(key)
        common = sub[:, members].sum(axis=1).A1 == len(members)
        if common.any():
            continue  # Already implied by a partitioning row
        model.add_cut(pos[members])
        added += 1
        if added >= max_cuts:
            break
    return added


def _blocks(a: csc_matrix, allowed: np.ndarray, n_int: int):
    """
    (block of each row, block of each pool column): rows are linked when an allowed binary column
    covers both; a column belongs to the block of its first row (a group row, or a slack's item row).
    """
    sub = a[:, np.flatnonzero(allowed[:n_int])]
    _, block_of_row = connected_components((sub @ sub.T).tocsr(), directed=False)
    first_row = a.indices[a.indptr[:-1]]
    return block_of_row, block_of_row[first_row]


def _solve_by_blocks(model: _Model, a: csc_matrix, a_csr, n_int: int, n_groups: int,
                     block_of_row: np.ndarray, block_of_col: np.ndarray, previous: Optional[np.ndarray],
                     tol: float, cut_rounds: int, node_budget: int):
    """
    Solve the model's current restricted problem to integrality one independent block at a time.
    previous (pool order) is an integer solution from a narrower window, or None; it is feasible
    here and gives each block a starting incumbent.
    Returns ((objective, z in model order) or None, nodes used, finished within budget).
    All fixings made here are undone before returning (cuts stay: they are valid everywhere).
    """
    nodes = 1
    value = model.solve()
    if value is None:
        return None, nodes, True
    committed = []
    model_block = block_of_col[model.pool_index]
    finished = True
    while True:
        z = model.values()
        branch = _pick_assignment(model, a, z, n_int, n_groups, None)
        if branch is None:
            result = (value, z.copy())
            break
        block = block_of_row[branch[0]]
        in_block = np.flatnonzero(model_block == block)
        incumbent = np.inf
        fix_previous = None
        if previous is not None:
            # The previous solution's pattern on this block, evaluated with the other blocks at their LP
            fix_previous = in_block[previous[model.pool_index[in_block]] < 0.5]
            model.fix_zero(fix_previous)
            seeded = model.solve()
            nodes += 1
            model.unfix(fix_previous)
            if seeded is not None:
                incumbent = seeded
        best, used, done = _branch_and_bound(model, a, a_csr, n_int, n_groups, block, model_block,
                                             incumbent, tol, cut_rounds, node_budget - nodes)
        nodes += used
        if not done:
            finished = False
            result = None
            break
        if best is not None:
            # Commit the block's optimum: every column of the block that is zero there stays zero
            fix = in_block[best[1][in_block] < 0.5]
        elif np.isfinite(incumbent):
            fix = fix_previous  # Nothing beats the previous pattern on this block
        else:
            raise RuntimeError("A feasible block produced no integer solution")
        model.fix_zero(fix)
        committed.append(fix)
        value = model.solve()
        nodes += 1
        if value is None:
            raise RuntimeError("Committing a block optimum made the LP infeasible")
    for fix in reversed(committed):
        model.unfix(fix)
    return result, nodes, finished


def _branch_and_bound(model: _Model, a: csc_matrix, a_csr, n_int: int, n_groups: int,
                      block: int, model_block: np.ndarray, incumbent: float, tol: float,
                      cut_rounds: int, node_budget: int):
    """
    Depth-first branch-and-bound with assignment branching, restricted to one block: a node is a
    leaf once the block's assignment shares are all 0/1 (other blocks are independent and keep
    their LP values, so comparing whole-LP values compares this block's objective).
    Only solutions strictly better than incumbent are returned.
    Returns (best (objective, z in model order) or None, nodes used, finished within budget).
    Every fixing made here is undone before returning.
    """
    best = None
    best_obj = incumbent
    nodes = 0
    in_block = model_block == block
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
        for _ in range(cut_rounds):
            if value is None or value >= best_obj - tol:
                break
            if _separate_cliques(model, a, in_block) == 0:
                break
            value = model.solve()
        if value is None or value >= best_obj - tol:
            continue
        z = model.values()
        branch = _pick_assignment(model, a, z, n_int, n_groups, (model_block, block))
        if branch is None:
            best_obj = value
            best = (value, z.copy())
            continue
        g, item_row, _ = branch
        serves, not_serves = _assignment_branches(model, a_csr, g, item_row)
        # Dive into "g serves o" first (pushed last)
        stack.append(("fix", not_serves))
        stack.append(("fix", serves))
    return best, nodes, True


def _pick_assignment(model: _Model, a: csc_matrix, z: np.ndarray, n_int: int, n_groups: int, only_block):
    """
    (group row, item row, LP share) of the fractional pair with the largest share, or None when all
    shares are 0/1. only_block = (block of each model column, block id) limits the search to one block.
    """
    positive = np.flatnonzero(z > 1e-9)
    pool = model.pool_index[positive]
    binary = pool < n_int
    if only_block is not None:
        model_block, block = only_block
        binary &= model_block[positive] == block
    share = {}
    for j_model, j_pool in zip(positive[binary], pool[binary]):
        rows = a.indices[a.indptr[j_pool]:a.indptr[j_pool + 1]]
        g = int(rows[rows < n_groups][0])
        for r in rows[rows >= n_groups]:
            key = (g, int(r))
            share[key] = share.get(key, 0.0) + z[j_model]
    best_key, best_share = None, 0.0
    for key, v in share.items():
        if 1e-6 < v < 1.0 - 1e-6 and v > best_share:
            best_key, best_share = key, v
    if best_key is None:
        return None
    return best_key[0], best_key[1], best_share


def _assignment_branches(model: _Model, a_csr, g: int, item_row: int):
    """
    Model columns to fix to zero in each branch:
    'g serves item'         -> columns of g without the item, and columns of other groups (or the slack) with it
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
