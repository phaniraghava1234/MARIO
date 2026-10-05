"""Smoke test for conformal.py's math — synthetic data, no model.

Verifies:
  1. conformal_q returns correct order statistic (basic sanity)
  2. On Gaussian data: conformal q ≈ z-value (so matches scalar calibration)
  3. On heavy-tailed data (Student-t): conformal achieves nominal coverage
     where Gaussian/scalar calibration undershoots
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm, t as student_t

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from airfrans_task.uq.conformal import conformal_q, coverage


def main():
    rng = np.random.default_rng(0)
    N_cal, N_test = 5000, 10000

    # case 1: errors are N(0,1), model reports sigma=1 → should recover z-values
    err_cal = rng.normal(0, 1, N_cal)
    err_test = rng.normal(0, 1, N_test)
    sigma = np.ones(N_cal)
    nonconf = np.abs(err_cal) / sigma

    q68 = conformal_q(nonconf, 0.32)
    q95 = conformal_q(nonconf, 0.05)
    print(f"Gaussian case: q68={q68:.3f} (expect ~1.0), "
          f"q95={q95:.3f} (expect ~1.96)")
    assert abs(q68 - 1.0) < 0.05
    assert abs(q95 - 1.96) < 0.1

    sigma_t = np.ones(N_test)
    cov68 = coverage(err_test, q68 * sigma_t)
    cov95 = coverage(err_test, q95 * sigma_t)
    print(f"Gaussian coverage: 68={cov68:.3f}, 95={cov95:.3f}")
    assert abs(cov68 - 0.68) < 0.02
    assert abs(cov95 - 0.95) < 0.01

    # case 2: errors from Student-t with 4 dof (heavier tails),
    # model reports Gaussian sigma=1. Gaussian coverage at z=1.96 undershoots;
    # conformal fixes it.
    err_cal_t = student_t.rvs(df=4, size=N_cal, random_state=rng)
    err_test_t = student_t.rvs(df=4, size=N_test, random_state=rng)
    nonconf_t = np.abs(err_cal_t) / sigma     # std still "1" (misspecified)
    q95_t = conformal_q(nonconf_t, 0.05)
    print(f"\nStudent-t df=4 case: conformal q95 = {q95_t:.3f} (> 1.96)")
    assert q95_t > 1.96   # heavier tails need wider intervals

    cov_gauss = coverage(err_test_t, 1.96 * sigma_t)
    cov_conf = coverage(err_test_t, q95_t * sigma_t)
    print(f"  coverage at 95% target: gaussian={cov_gauss:.3f}, "
          f"conformal={cov_conf:.3f}")
    assert cov_gauss < 0.95       # Gaussian under-covers heavy tails
    assert abs(cov_conf - 0.95) < 0.015   # conformal within sampling noise

    # case 3: errors are MORE peaked than Gaussian (mixture-sharp).
    # Simulates our actual data situation: scalar calibration over-covers at 68%.
    err_cal_p = rng.choice([-1, 1], size=N_cal) * np.abs(rng.normal(0, 0.3, N_cal))
    err_test_p = rng.choice([-1, 1], size=N_test) * np.abs(rng.normal(0, 0.3, N_test))
    sigma_mis = np.ones(N_cal)        # model way over-reports uncertainty
    nonconf_p = np.abs(err_cal_p) / sigma_mis
    q68_p = conformal_q(nonconf_p, 0.32)
    print(f"\nSharp case: conformal q68 = {q68_p:.3f} (< 1.0)")
    assert q68_p < 1.0
    cov_conf_p = coverage(err_test_p, q68_p * np.ones(N_test))
    print(f"  conformal cov68 = {cov_conf_p:.3f}")
    assert abs(cov_conf_p - 0.68) < 0.02

    print("\ntest_conformal: ALL OK")


if __name__ == "__main__":
    main()
