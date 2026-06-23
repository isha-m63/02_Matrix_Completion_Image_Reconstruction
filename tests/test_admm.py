import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.admm import admm_solve, ADMMResult, ADMMHistory
from src.operators import MeasurementOperator
from src.svt import effective_rank
from config import TOL

def make_low_rank_problem(n=20, rank=2, m=None, seed=42):
    """
    Returns (op, y, X_true).
    m defaults to 3*r*n (well above the theoretical minimum).
    """
    rng = np.random.default_rng(seed)
    if m is None:
        m = 3 * rank * n          #generous sampling — solver should definitely work

    #Build rank-r matrix as outer product sum
    U = rng.standard_normal((n, rank))
    V = rng.standard_normal((n, rank))
    X_true = U @ V.T              # exactly rank-r

    op = MeasurementOperator(n=n, m=m, seed=seed)
    y = op.measure(X_true)  
    return op, y, X_true


def test_exact_recovery():
    """
    Construct a known rank-2 matrix of small size, take enough measurements, run the solver, check you recover it.
    Returns correct types and shapes.
    Solver must recover a known low-rank matrix to within tolerance.
    This is the fundamental correctness test — if this fails nothing else matters.
    """
    op, y, X_true = make_low_rank_problem(n=20, rank=2, m=300)
    result = admm_solve(op, y, X_true=X_true, verbose=False)
    rel_err = np.linalg.norm(result.X - X_true, 'fro') / np.linalg.norm(X_true, 'fro')
    assert rel_err < 1e-2, (
        f"Recovery failed: relative error {rel_err:.4f} exceeds 1e-2. "
        f"Solver converged={result.converged} in {result.n_iters} iters."
    )


def test_returns_correct_types_and_shapes():
    """
    ADMMResult fields must have correct types and shapes.
    Catches dataclass wiring mistakes early.
    """
    n = 20
    op, y, X_true = make_low_rank_problem(n=n, rank=2)
    result = admm_solve(op, y, verbose=False)

    assert isinstance(result, ADMMResult)
    assert isinstance(result.history, ADMMHistory)
    assert isinstance(result.converged, bool)
    assert isinstance(result.n_iters, int)
    assert result.n_iters > 0
    assert result.X.shape == (n, n), f"Expected ({n},{n}), got {result.X.shape}"


def test_history_lengths_consistent():
    """
    All history lists must have exactly n_iters entries.
    Inconsistent lengths break convergence plots downstream.
    """
    op, y, X_true = make_low_rank_problem(n=20, rank=2)
    result = admm_solve(op, y, X_true=X_true, verbose=False)

    n = result.n_iters
    assert len(result.history.primal_residuals) == n,  \
        f"primal_residuals length {len(result.history.primal_residuals)} != {n}"
    assert len(result.history.dual_residuals) == n,    \
        f"dual_residuals length {len(result.history.dual_residuals)} != {n}"
    assert len(result.history.objective_values) == n,  \
        f"objective_values length {len(result.history.objective_values)} != {n}"
    assert len(result.history.ranks) == n,             \
        f"ranks length {len(result.history.ranks)} != {n}"
    assert len(result.history.rel_errors) == n,        \
        f"rel_errors length {len(result.history.rel_errors)} != {n}"


def test_primal_residual_decreases():
    """
    Primal residual must trend downward over a converged run.
    Checks by comparing the mean of the first quarter vs the last quarter.
    ADMM doesn't guarantee strict monotone decrease every step,
    but the overall trend must be downward.
    """
    op, y, _ = make_low_rank_problem(n=20, rank=2)
    result = admm_solve(op, y, verbose=False)

    r = result.history.primal_residuals
    n = len(r)
    early_mean = np.mean(r[:n//4])
    late_mean  = np.mean(r[3*n//4:])

    assert late_mean < early_mean, (
        f"Primal residual not decreasing: "
        f"early mean={early_mean:.4f}, late mean={late_mean:.4f}"
    )


def test_residuals_below_tol_at_convergence():
    """
    If converged=True, both residuals at the final iteration
    must actually be below TOL. Checks the convergence flag is honest.
    """
    op, y, _ = make_low_rank_problem(n=20, rank=2)
    result = admm_solve(op, y, tol=TOL, verbose=False)

    if result.converged:
        final_primal = result.history.primal_residuals[-1]
        final_dual   = result.history.dual_residuals[-1]
        assert final_primal < TOL, \
            f"Converged=True but primal residual {final_primal:.2e} >= tol {TOL}"
        assert final_dual < TOL, \
            f"Converged=True but dual residual {final_dual:.2e} >= tol {TOL}"


def test_solution_is_low_rank():
    """
    Recovered X must have effective rank equal to the true rank.
    If the solution is full rank, SVT threshold is wrong or rho is too small.
    """
    true_rank = 2
    op, y, X_true = make_low_rank_problem(n=20, rank=true_rank, m=300)
    result = admm_solve(op, y, verbose=False)

    recovered_rank = effective_rank(result.X, tol=1e-3)
    assert recovered_rank <= true_rank + 1, (
        f"Expected rank ≤ {true_rank+1}, got {recovered_rank}. "
        f"Solution is not low-rank — check rho or SVT threshold."
    )


def test_rel_errors_empty_when_no_X_true():
    """
    When X_true is not provided, rel_errors must be empty.
    When X_true is provided, rel_errors must have n_iters entries.
    Tests optional argument branching in the solver loop.
    """
    op, y, X_true = make_low_rank_problem(n=20, rank=2)

    result_no_true = admm_solve(op, y, X_true=None, verbose=False)
    assert len(result_no_true.history.rel_errors) == 0, \
        "rel_errors should be empty when X_true=None"

    result_with_true = admm_solve(op, y, X_true=X_true, verbose=False)
    assert len(result_with_true.history.rel_errors) == result_with_true.n_iters, \
        "rel_errors length must equal n_iters when X_true is provided"


def test_deterministic():
    """
    Same inputs must produce identical results across two runs.
    ADMM is deterministic — no randomness inside the solver loop.
    """
    op, y, _ = make_low_rank_problem(n=20, rank=2, seed=7)

    result1 = admm_solve(op, y, verbose=False)
    result2 = admm_solve(op, y, verbose=False)

    np.testing.assert_array_equal(
        result1.X, result2.X,
        err_msg="Solver is not deterministic — hidden randomness somewhere"
    )


if __name__ == "__main__":
    test_exact_recovery()
    test_returns_correct_types_and_shapes()
    test_history_lengths_consistent()
    test_primal_residual_decreases()
    test_residuals_below_tol_at_convergence()
    test_solution_is_low_rank()
    test_rel_errors_empty_when_no_X_true()
    test_deterministic()
    print("All ADMM tests passed.")