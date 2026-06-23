import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.metrics import relative_error, psnr, ssim, print_metrics


def test_relative_error_perfect_recovery():
    """Perfect recovery must give exactly 0.0."""
    X = np.random.rand(10, 10)
    assert relative_error(X, X) == 0.0


def test_relative_error_known_value():
    """
    Scale X_true by 2 → recovered = 2*true.
    ||2X - X||_F / ||X||_F = ||X||_F / ||X||_F = 1.0
    """
    X_true = np.ones((5, 5))
    X_recovered = 2 * X_true
    np.testing.assert_allclose(relative_error(X_recovered, X_true), 1.0, rtol=1e-10)


def test_relative_error_zero_matrix():
    """Recovering zeros from a nonzero true → error = 1.0."""
    X_true = np.ones((5, 5))
    X_recovered = np.zeros((5, 5))
    np.testing.assert_allclose(relative_error(X_recovered, X_true), 1.0, rtol=1e-10)


def test_psnr_perfect_recovery():
    """Perfect recovery must give infinite PSNR."""
    X = np.random.rand(10, 10)
    assert np.isinf(psnr(X, X))


def test_psnr_increases_with_quality():
    """
    Better recovery must give higher PSNR.
    Small noise → higher PSNR than large noise.
    """
    rng = np.random.default_rng(0)
    X_true = rng.random((20, 20))

    X_good = np.clip(X_true + 0.01 * rng.standard_normal((20, 20)), 0, 1)
    X_bad  = np.clip(X_true + 0.20 * rng.standard_normal((20, 20)), 0, 1)

    assert psnr(X_good, X_true) > psnr(X_bad, X_true)


def test_ssim_perfect_recovery():
    """Perfect recovery must give SSIM = 1.0."""
    X = np.random.rand(20, 20)
    np.testing.assert_allclose(ssim(X, X), 1.0, atol=1e-10)


def test_ssim_increases_with_quality():
    """Better recovery must give higher SSIM."""
    rng = np.random.default_rng(1)
    X_true = rng.random((20, 20))

    X_good = np.clip(X_true + 0.01 * rng.standard_normal((20, 20)), 0, 1)
    X_bad  = np.clip(X_true + 0.30 * rng.standard_normal((20, 20)), 0, 1)

    assert ssim(X_good, X_true) > ssim(X_bad, X_true)


def test_psnr_requires_normalised_input():
    """psnr must raise AssertionError if X_true is not in [0,1]."""
    X_true = np.ones((10, 10)) * 255.0
    X_rec  = np.ones((10, 10)) * 200.0
    with pytest.raises(AssertionError):
        psnr(X_rec, X_true)


def test_ssim_requires_normalised_input():
    """ssim must raise AssertionError if X_true is not in [0,1]."""
    X_true = np.ones((10, 10)) * 255.0
    X_rec  = np.ones((10, 10)) * 200.0
    with pytest.raises(AssertionError):
        ssim(X_rec, X_true)


def test_print_metrics_runs_without_error():
    """print_metrics must not raise on valid inputs."""
    X = np.random.rand(20, 20)
    print_metrics(X, X)    # perfect recovery — just checks it doesn't crash


if __name__ == "__main__":
    test_relative_error_perfect_recovery()
    test_relative_error_known_value()
    test_relative_error_zero_matrix()
    test_psnr_perfect_recovery()
    test_psnr_increases_with_quality()
    test_ssim_perfect_recovery()
    test_ssim_increases_with_quality()
    test_psnr_requires_normalised_input()
    test_ssim_requires_normalised_input()
    test_print_metrics_runs_without_error()
    print("All metrics tests passed.")