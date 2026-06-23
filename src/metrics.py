import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def relative_error(X_recovered: np.ndarray, X_true: np.ndarray) -> float:
    """
    Frobenius-norm relative error.

        ||X_recovered - X_true||_F / ||X_true||_F

    Primary metric for matrix recovery quality.
    0.0 = perfect recovery. Anything below 0.01 is excellent.
    """
    return (np.linalg.norm(X_recovered - X_true, 'fro') /
            np.linalg.norm(X_true, 'fro'))


def psnr(X_recovered: np.ndarray, X_true: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).

    Higher is better. Typical thresholds:
        > 40 dB  — excellent, visually indistinguishable
        30-40 dB — good
        20-30 dB — acceptable
        < 20 dB  — poor

    Both inputs must be in [0, 1] — normalise before calling.
    data_range=1.0 reflects this.
    """
    assert X_true.min() >= 0.0 and X_true.max() <= 1.0, \
        "X_true must be normalised to [0, 1] before computing PSNR"

    X_recovered_clipped = np.clip(X_recovered, 0.0, 1.0)
    return peak_signal_noise_ratio(X_true, X_recovered_clipped, data_range=1.0)


def ssim(X_recovered: np.ndarray, X_true: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM).

    Perceptual quality metric — measures luminance, contrast, structure.
    Range: [-1, 1]. In practice for natural images:
        > 0.95 — excellent
        0.8-0.95 — good
        < 0.8   — noticeable degradation

    Both inputs must be in [0, 1].
    """
    assert X_true.min() >= 0.0 and X_true.max() <= 1.0, \
        "X_true must be normalised to [0, 1] before computing SSIM"

    X_recovered_clipped = np.clip(X_recovered, 0.0, 1.0)
    return structural_similarity(X_true, X_recovered_clipped, data_range=1.0)


def print_metrics(X_recovered: np.ndarray, X_true: np.ndarray) -> None:
    """
    Print all three metrics in one call.
    Convenience wrapper for experiments.
    """
    rel = relative_error(X_recovered, X_true)
    p   = psnr(X_recovered, X_true)
    s   = ssim(X_recovered, X_true)

    print(f"Relative error : {rel:.6f}")
    print(f"PSNR: {p:.2f} dB")
    print(f"SSIM: {s:.6f}")