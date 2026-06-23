import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.svt import svt, effective_rank

def test_svt_thresholds_correctly():
    """
    Singular values should be soft-thresholded, not hard-thresholded.
    Did the SVT implementation produce a matrix whose singular values are exactly the soft-thresholded ones?
    """
    U, _ = np.linalg.qr(np.random.randn(5, 5))
    sigma = np.array([5.0, 3.0, 1.5, 0.8, 0.2])
    Y = U @ np.diag(sigma) @ U.T
    lam = 1.0
    result = svt(Y, lam)
    #print(result.shape)
    #print(result) 

    _, sigma_out, _ = np.linalg.svd(result, full_matrices=False)
    expected = np.maximum(sigma - lam, 0.0)
    #print(expected.shape)
    #print(expected)

    np.testing.assert_allclose(sorted(sigma_out, reverse=True),
                               sorted(expected, reverse=True),
                               atol=1e-10)
    

def test_svt_zeros_small_singular_values():
    """Singular values below lam should become exactly zero."""
    U, _ = np.linalg.qr(np.random.randn(5,5))
    sigma = np.array([5.0, 3.0, 0.5, 0.3, 0.1])
    Y = U @ np.diag(sigma) @ U.T
    result = svt(Y, lam=1.0)
    #print(result)
    _, sigma_out, _ = np.linalg.svd(result, full_matrices=False)
    #print(sigma_out)
    assert np.all(sigma_out[2:] < 1e-10), "Small singular values should be zeroed"


def test_svt_returns_low_rank():
    """A rank-2 signal + small singular value noise i.e., SVT returns rank-2."""
    rng = np.random.default_rng(42)    #Random number generator
    U, _ = np.linalg.qr(rng.standard_normal((10, 10)))
    sigma = np.array([10.0, 8.0, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005])
    Y = U @ np.diag(sigma) @ U.T
    result = svt(Y, lam=1.0)
    #print(effective_rank(result))
    assert effective_rank(result) == 2


def test_svt_when_lam_zero():
    """SVT with lam=0 should return the input unchanged."""
    Y = np.random.randn(6, 6)
    result = svt(Y, lam=0.0)
    #print(Y[0], result[0])
    np.testing.assert_allclose(result, Y, atol=1e-12)


def test_svt_all_zeros_when_lam_large():
    """SVT with lam larger than all singular values should return zero matrix."""
    Y = np.random.randn(5, 5)
    sigma_max = np.linalg.svd(Y, compute_uv=False).max()
    result = svt(Y, lam=sigma_max + 1.0)
    np.testing.assert_allclose(result, np.zeros_like(Y), atol=1e-10)



if __name__ == "__main__":
    test_svt_thresholds_correctly()
    test_svt_zeros_small_singular_values()
    test_svt_returns_low_rank()
    test_svt_when_lam_zero()
    test_svt_all_zeros_when_lam_large()
    print("All SVT tests passed.")
