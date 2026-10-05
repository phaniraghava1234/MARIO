"""Smoke test for sensitivity.py's Sobol wiring — synthetic model.

Verifies:
  1. Saltelli samples have the correct size: N * (2d + 2)
  2. On a known monotonic function Y = a*X1 + b*X2 (no interaction):
     S1[X1] ≈ a^2 Var(X1) / Var(Y), S1[X2] ≈ b^2 Var(X2) / Var(Y),
     ST[X_i] ≈ S1[X_i] (no interactions).
  3. On a pure-interaction function Y = X1 * X2 (centred): S1 ≈ 0,
     ST > 0 for both.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

try:
    from SALib.sample.sobol import sample as saltelli_sample
except ImportError:
    from SALib.sample.saltelli import sample as saltelli_sample
from SALib.analyze.sobol import analyze


def test_saltelli_size():
    problem = {"num_vars": 2, "names": ["x1", "x2"],
               "bounds": [[0, 1], [0, 1]]}
    N = 128
    s = saltelli_sample(problem, N)
    # Standard Saltelli for d=2: N * (2*d + 2) = 6N rows
    assert s.shape == (N * 6, 2), s.shape
    print(f"saltelli samples shape ok: {s.shape}")


def test_additive_separable():
    """Y = 2*X1 + 1*X2 with X_i ~ U(-1, 1).
    Var(Y) = 4*Var(X1) + 1*Var(X2) = (4 + 1) * (1/3) = 5/3
    S1[X1] = 4/3 / (5/3) = 0.8
    S1[X2] = 1/3 / (5/3) = 0.2
    ST[X_i] = S1[X_i] (no interactions)."""
    problem = {"num_vars": 2, "names": ["x1", "x2"],
               "bounds": [[-1, 1], [-1, 1]]}
    s = saltelli_sample(problem, 2048)
    Y = 2 * s[:, 0] + 1 * s[:, 1]
    Si = analyze(problem, Y, print_to_console=False)
    print(f"additive: S1={Si['S1']} (expect 0.8, 0.2), "
          f"ST={Si['ST']} (expect 0.8, 0.2)")
    assert abs(Si["S1"][0] - 0.8) < 0.05
    assert abs(Si["S1"][1] - 0.2) < 0.05
    assert abs(Si["ST"][0] - Si["S1"][0]) < 0.05
    assert abs(Si["ST"][1] - Si["S1"][1]) < 0.05


def test_pure_interaction():
    """Y = X1 * X2 with X_i ~ U(-1, 1).
    Both inputs enter only via interaction, so S1 ≈ 0, ST > 0."""
    problem = {"num_vars": 2, "names": ["x1", "x2"],
               "bounds": [[-1, 1], [-1, 1]]}
    s = saltelli_sample(problem, 2048)
    Y = s[:, 0] * s[:, 1]
    Si = analyze(problem, Y, print_to_console=False)
    print(f"interaction: S1={Si['S1']} (expect ~0), "
          f"ST={Si['ST']} (expect ~1)")
    assert abs(Si["S1"][0]) < 0.1
    assert abs(Si["S1"][1]) < 0.1
    assert Si["ST"][0] > 0.5
    assert Si["ST"][1] > 0.5


def main():
    test_saltelli_size()
    test_additive_separable()
    test_pure_interaction()
    print("\ntest_sensitivity: ALL OK")


if __name__ == "__main__":
    main()
