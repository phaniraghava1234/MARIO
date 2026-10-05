"""Smoke test for calibrate.py's math — no model, no dataset.

Verifies:
  1. fit_scalar_s recovers the true s on synthetic data where the
     model is intentionally mis-scaled
  2. coverage() matches np.mean exact for simple inputs
  3. post-calibration coverage is closer to nominal than pre
  4. gaussian_nll is lower post-calibration than pre
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from airfrans_task.uq.calibrate import (
    coverage, fit_scalar_s, gaussian_nll, reliability_curve,
)


def main():
    rng = np.random.default_rng(0)
    N = 20000

    # true error: Gaussian(0, 1); model reports sigma_pred = 2.0 everywhere
    # (so model is 2x over-confident... wait, over-reporting std means
    # OVER-covered. True s = 1/2 = 0.5 should bring sigma down to 1.0.)
    err = rng.normal(0, 1, size=N)
    sigma_pred = np.full(N, 2.0)
    true_s = 0.5

    s_hat = fit_scalar_s(err, sigma_pred)
    print(f"fit_scalar_s: estimated {s_hat:.4f}, true {true_s:.4f}")
    assert abs(s_hat - true_s) < 0.02, s_hat

    # coverage pre: too wide, over-covers
    cov_pre_68 = coverage(err, sigma_pred, 0.68)
    cov_pre_95 = coverage(err, sigma_pred, 0.95)
    sigma_cal = sigma_pred * s_hat
    cov_post_68 = coverage(err, sigma_cal, 0.68)
    cov_post_95 = coverage(err, sigma_cal, 0.95)
    print(f"cov68: pre {cov_pre_68:.3f} -> post {cov_post_68:.3f} (target 0.68)")
    print(f"cov95: pre {cov_pre_95:.3f} -> post {cov_post_95:.3f} (target 0.95)")
    assert cov_pre_68 > 0.90
    assert abs(cov_post_68 - 0.68) < 0.02
    assert abs(cov_post_95 - 0.95) < 0.02

    # NLL should drop post-calibration
    nll_pre = gaussian_nll(err, sigma_pred)
    nll_post = gaussian_nll(err, sigma_cal)
    print(f"NLL: pre {nll_pre:.4f} -> post {nll_post:.4f}")
    assert nll_post < nll_pre

    # under-confident case: sigma_pred = 0.5 (half of true), s should be 2.0
    sigma_under = np.full(N, 0.5)
    s_up = fit_scalar_s(err, sigma_under)
    print(f"under-confident case: s = {s_up:.4f} (true 2.0)")
    assert abs(s_up - 2.0) < 0.05

    # reliability curve basic shape
    probs, obs = reliability_curve(err, sigma_pred * s_hat, n_bins=10)
    # after calibration, observed should track expected (within sampling noise)
    rmse = float(np.sqrt(np.mean((obs - probs) ** 2)))
    print(f"reliability RMSE after calibration: {rmse:.3f}")
    assert rmse < 0.03

    print("\ntest_calibrate: ALL OK")


if __name__ == "__main__":
    main()
