"""
Singular Value Thresholding (SVT): 

- SVT operator is used to recover low rank matrices from noisy/incomplete data
- It is 'spectral' equivalent of LASSO i.e., 
  SVT promotes sparsity in the singular matrix of SVD isntead of all the coefficients
- It reduces singular values by lam (or lambda) and removes all negative values
- It serves as proximal operator for nuclear norm (analogous to l1 norm) which is the sum of singular values

- The proximal operator to nuclear norm is given as: 
 prox(Y) = min_X (lam * ||X||_*  +  (1/2) * ||X - Y||^2_F^2)
                    promotes sparsity        penalty term

- Intutively, singular values represent the strength of different components in the data. 
- SVT promotes low rank structure by shrinking values less than lam towards zero.
- For components larger than lam, it assumes they contain signal but are "contaminated" by noise, 
  so it shrinks them by lam to pull them toward a simpler (lower rank) representation.
"""

import numpy as np

def svt(Y: np.ndarray, lam: float) -> np.ndarray:
    """
    Args:
    Y : (m, n) ndarray - input matrix
    lam : float - threshold (= 1/rho in ADMM)

    Returns:
    X : (m, n) ndarray  — low-rank output matrix
    """

    U, sigma, Vt = np.linalg.svd(Y, full_matrices = False)
    sigma_threshold = np.maximum(sigma - lam, 0.0)
    return (U @ np.diag(sigma_threshold) @ Vt)
    

def effective_rank(Y: np.ndarray, tol: float = 1e-6) -> int:
    """
    Count singular values above tol.
    Useful for tracking how rank evolves during ADMM iterations.

    Args:
    Y : (m, n) ndarray - input matrix
    tol - tolerance

    Returns:
    rank - effective rank of input matrix
    """

    sigma = np.linalg.svd(Y, compute_uv=False)
    rank = int(np.sum(sigma > tol))
    return rank

