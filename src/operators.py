import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from config import IMAGE_SIZE, N_MEASUREMENTS, RANDOM_SEED

class MeasurementOperator:
    """
    Reference: arXiv:0706.4138v1 - Guaranteed Minimum-Rank Solutions of Linear Matrix Equations 
                                    via Nuclear Norm Minimization

    Simulates a one-pixel camera: m random linear projections of an nxn image

    Forward:  A(X)  = A_mat @ vec(X)        shape (m,)
    Adjoint:  A*(u) = reshape(A_mat.T @ u)  shape (n, n)

    The adjoint satisfies <A(X), u> = <X, A*(u)> for all X, u
    This is verified in tests/test_operators.py before use
    """

    def __init__(self, n: int = IMAGE_SIZE, m: int = N_MEASUREMENTS, seed: int = RANDOM_SEED):
        """
        RIP scaling (preserves the Euclidean length of the low-rank signal) and vectorization

        Each row is one random measurement pattern of length n square.
        Dividing by sqrt(m) ensures A^T A ≈ I (RIP scaling),
        so the measurement operator approximately preserves Euclidean geometry.
        This is what makes (I + A A*)^{-1} well-conditioned in the Z-update.
        """
        self.n = n
        self.m = m
        self.n2 = n*n

        rng = np.random.default_rng(seed)
        self.A = rng.standard_normal((m, self.n2)) / np.sqrt(m)     


    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        This implements the mapping y=A(X)
        X = (n, n) array
        It uses X.ravel() to flatten the 2D image into a vector before performing matrix multiplication
        """
        return self.A @ X.ravel()
    

    def measure(self, X:np.ndarray) -> np.ndarray:
        """
        Generate measurements from a ground-truth image.
        Alias for forward — kept separate semantically because
        in experiments you call this once on X_true to get y,
        then never touch X_true again.
        """
        return self.forward(X)


    def adjoint(self, u: np.ndarray) -> np.ndarray:
        """
        Apply A* to measurement vector u. Adjoint A is the mathematical bridge
        that allows the algorithm to move backwards from your measurements (y) to your image pixels (X).
        u : (m,) ndarray
        returns : (n, n) ndarray
        """
        return (self.A.T @ u).reshape(self.n, self.n)
    

    def z_update(self, X: np.ndarray, lam: np.ndarray, y: np.ndarray, rho: float) -> np.ndarray:
        """
        ADMM Z-update: least-squares projection onto measurement constraint.
        This assumes X = Z and A(Z) = y

        The Z-update repairs this by finding the closest matrix to: V = X + lam/rho 
        that exactly satisfies the measurements.

        What the update does: 
        
        Z = argmin_Z  (rho/2) * ||Z - V||_F^2   s.t.  A(Z) = y

        Closed-form via matrix inversion lemma:
        Z = V - A*( (I + A A*)^{-1} (A(V) - y) )

        Since A = A_mat / sqrt(m), we have A A* = A_mat A_mat^T / m
        which is well-conditioned for Gaussian A_mat.

        Args:
        X   : (n, n) current X iterate
        lam : (n, n) scaled dual variable  (lambda in ADMM)
        y   : (m,)   measurements
        rho : float  ADMM penalty parameter

        Returns:
        Z   : (n, n) updated Z
        """

        V = X + lam/rho
        AV = self.forward(V)

        residual = AV - y

        AAT = self.A @ self.A.T
        #coeff = np.linalg.solve(np.eye(self.m) + AAT, residual)    #Understand this better
        coeff = np.linalg.solve(AAT, residual)

        Z = V - self.adjoint(coeff)
        return Z    
    
 
        
