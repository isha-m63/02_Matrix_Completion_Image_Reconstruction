import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from dataclasses import dataclass, field
from typing import List

from src.svt import svt, effective_rank
from src.operators import MeasurementOperator
from config import RHO, MAX_ITER, TOL


@dataclass
class ADMMHistory:
    """
    Stores per-iteration diagnostics for convergence analysis and plotting.
    """
    primal_residuals: list = field(default_factory=list)
    dual_residuals: list = field(default_factory=list)
    objective_values: list = field(default_factory=list)
    ranks: list[int] = field(default_factory=list)
    rel_errors: list = field(default_factory=list)    #Works only when X_true is provided


@dataclass
class ADMMResult:
    """
    Everything the solver returns.
    """
    X: np.ndarray   #Recovered reconstructed matrix
    history: ADMMHistory
    n_iters: int    #Number of iterations solver took to converge
    converged: bool


def admm_solve(
        op: MeasurementOperator, y: np.ndarray,
        rho: float = RHO, max_iter: float = MAX_ITER, 
        tol: float = TOL, X_true: np.ndarray = None, verbose: bool = True,
) -> ADMMResult:
    
    """
    Solves:  min  ||X||_*   s.t.  A(X) = y
    via ADMM with variable splitting X = Z.

    Augmented Lagrangian:
        L(X, Z, Λ) = ||X||_*  +  <Λ, X-Z>  +  (rho/2)||X-Z||_F^2
        subject to  A(Z) = y

    Updates each iteration:
        X =  SVT_{1/rho}( Z - Λ/rho )
        Z = op.z_update(X, Λ, y, rho)      #Projects onto A(Z)=y
        Λ =  Λ + rho * (X - Z)

    Convergence: stop when primal residual ||X-Z||_F < tol
                              AND dual residual rho*||Z_new-Z_old||_F < tol

    Args:
        op: MeasurementOperator instance
        y: (m,) measurement vector
        rho: ADMM penalty parameter (fixed)
        max_iter: iteration cap
        tol: convergence threshold for both residuals
        X_true: ground truth image — if provided, tracks rel error per iter
        verbose: print convergence info every 50 iterations

    Returns:
        ADMMResult with recovered X, diagnostics, iteration count, converged flag
    """

    n = op.n
    history = ADMMHistory()

    #Initialize
    X   = np.zeros((n, n))
    Z   = np.zeros((n, n)) 
    #Z = op.z_update(np.zeros((n, n)), np.zeros((n, n)), y, rho)  # feasible from iter 0
    Lam = np.zeros((n, n))
    converged = False

    '''if rho == RHO:      # only auto-scale if user didn't override
        rho = np.linalg.norm(op.adjoint(y), 'fro') / n
    if verbose:
        print(f"Auto-scaled rho = {rho:.4f}")'''

    if verbose:
        print(f"{'Iter':>6}  {'Primal':>12}  {'Dual':>12}  {'Rank':>6}  {'RelErr':>10}")
        print("-" * 55)

    for i in range(max_iter):
        Z_prev = Z.copy()

        #Update X (proximal operator of nuclear norm)
        #by minimiszing  ||X||_*  +  (rho/2)||X - (Z - Λ/rho)||_F^2
        X = svt(Z - Lam / rho, 1.0 / rho)


        #Update Z (project onto measurement constraint A(Z) = y)
        #by minimizing (rho/2)||Z - (X + Λ/rho)||_F^2   s.t.  A(Z) = y
        Z = op.z_update(X, Lam, y, rho)

        #Dual update
        Lam = Lam + rho * (X - Z)

        #Residuals
        primal_res = np.linalg.norm(X-Z, 'fro')
        dual_res = rho * np.linalg.norm(Z - Z_prev, 'fro')
        objective = np.sum(np.linalg.svd(X, compute_uv=False))   #Nuclear norm
        rank = effective_rank(X)

        history.primal_residuals.append(primal_res)
        history.dual_residuals.append(dual_res)
        history.objective_values.append(objective)
        history.ranks.append(rank)

        if X_true is not None:
            rel_err = (np.linalg.norm(X - X_true, 'fro') /
                       np.linalg.norm(X_true, 'fro'))
            history.rel_errors.append(rel_err)

        if verbose and (i % 50 == 0 or i < 5):
            rel_str = f"{history.rel_errors[-1]:.6f}" if X_true is not None else "n/a"
            print(f"{i+1:>6}  {primal_res:>12.6f}  {dual_res:>12.6f}  "
                  f"{rank:>6}  {rel_str:>10}")

        #Convergence check
        if primal_res < tol and dual_res < tol:
            converged = True
            if verbose:
                rel_str = f"{history.rel_errors[-1]:.6f}" if X_true is not None else "n/a"
                print(f"\nConverged at iteration {i+1}")
                print(f"Primal residual: {primal_res:.2e}")
                print(f"Dual residual: {dual_res:.2e}")
                print(f"Final rank: {rank}")
                if X_true is not None:
                    print(f"Relative error: {rel_str}")
            break

    if not converged and verbose:
        print(f"\nDid not converge in {max_iter} iterations.")
        print(f"Final primal residual : {primal_res:.2e}")
        print(f"Final dual residual   : {dual_res:.2e}")

    return ADMMResult(X=X, history=history, n_iters=i+1, converged=converged)
