"""Smoke test for propagate.py (updated with aleatoric/epistemic + Cl/Cd).

Verifies:
  1. Priors A/B statistics
  2. compute_qois returns five finite values including Cl_proxy, Cd_proxy
  3. decompose_variance: aleatoric + epistemic == total
  4. recover_normals inverts the min-max normalization
  5. On synthetic data: wider prior gives larger aleatoric variance
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from airfrans_task.uq.propagate import (
    QOIS, PRIORS, compute_qois, decompose_variance,
    propagate_one_geom, recover_normals,
    sample_prior_A, sample_prior_B,
)


def test_priors_statistics():
    rng = np.random.default_rng(0)
    U_A, A_A = sample_prior_A(5000, rng)
    U_B, A_B = sample_prior_B(5000, rng)
    print(f"Prior A: U {U_A.mean():.2f}±{U_A.std():.2f}, "
          f"AoA {A_A.mean():.2f}±{A_A.std():.2f}")
    print(f"Prior B: U {U_B.mean():.2f}±{U_B.std():.2f}, "
          f"AoA {A_B.mean():.2f}±{A_B.std():.2f}")
    assert abs(U_A.mean() - 50) < 0.5 and abs(U_A.std() - 2) < 0.2
    assert 48 < U_B.mean() < 52 and 5.5 < U_B.std() < 6.0
    assert U_B.std() > U_A.std() and A_B.std() > A_A.std()


def test_qoi_computation():
    rng = np.random.default_rng(1)
    N = 500
    fields = rng.normal(0, 1, size=(N, 4))
    normals = rng.normal(0, 1, size=(N, 2))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)   # unit normals
    is_airfoil = np.zeros(N, dtype=bool); is_airfoil[:50] = True
    q = compute_qois(fields, is_airfoil, normals, U_inf=50.0)
    assert set(q) == set(QOIS), q.keys()
    for k in QOIS:
        assert np.isfinite(q[k]), (k, q[k])
    print(f"QoIs on synthetic: Cl={q['Cl_proxy']:.3f}, Cd={q['Cd_proxy']:.3f}")


def test_recover_normals_inverse():
    rng = np.random.default_rng(2)
    N = 1000
    # fake physical normals in [-1, 2] for nx, [-2, 1] for ny
    nx_phys = rng.uniform(-1, 2, size=N)
    ny_phys = rng.uniform(-2, 1, size=N)
    nmin = np.array([-1.0, -2.0])
    nmax = np.array([2.0, 1.0])
    nrange = nmax - nmin
    coef_norm = {"normal": {"min": nmin, "max": nmax, "range": nrange}}
    # apply dataset normalization: n_n = 2*(n - min)/range - 1
    nx_n = 2 * (nx_phys - nmin[0]) / nrange[0] - 1
    ny_n = 2 * (ny_phys - nmin[1]) / nrange[1] - 1
    input_arr = np.zeros((N, 6)); input_arr[:, 3] = nx_n; input_arr[:, 4] = ny_n
    recovered = recover_normals(input_arr, coef_norm)
    assert np.allclose(recovered[:, 0], nx_phys, atol=1e-6)
    assert np.allclose(recovered[:, 1], ny_phys, atol=1e-6)
    print("recover_normals: inverse matches")


def test_variance_decomposition_identity():
    """Total variance identity: Var[Q] = E[Var[Q|θ]] + Var[E[Q|θ]]."""
    rng = np.random.default_rng(3)
    n_draws, T, n_q = 500, 20, 3
    # synthetic: per-draw latent mean μ_i ~ N(0,1), passes = μ_i + N(0, s^2)
    mu = rng.normal(0, 1, size=(n_draws, n_q))
    noise_std = np.array([0.3, 0.5, 0.1])
    passes = mu[:, None, :] + rng.normal(0, 1, size=(n_draws, T, n_q)) * noise_std
    d = decompose_variance(passes)
    # expected: aleatoric ≈ 1.0, epistemic ≈ noise_std^2
    print(f"aleatoric var (expect ~1.0, 1.0, 1.0): {d['aleatoric_var']}")
    print(f"epistemic var (expect {noise_std ** 2}): {d['epistemic_var']}")
    # loose tolerances due to finite sampling
    assert np.allclose(d["aleatoric_var"], 1.0, atol=0.1)
    assert np.allclose(d["epistemic_var"], noise_std ** 2, atol=0.05)
    # identity: total = aleatoric + epistemic
    assert np.allclose(d["total_var"], d["aleatoric_var"] + d["epistemic_var"])


class _ToyModel(torch.nn.Module):
    def modulated_forward(self, x, z):
        base = x.sum(dim=-1, keepdim=True) * 0.0
        u_in = z[:, -2:-1]; v_in = z[:, -1:]
        return torch.cat([base + u_in, base + v_in,
                          base - 0.5 * (u_in ** 2 + v_in ** 2),  # pressure
                          base + 0.01 * u_in], dim=-1)


def test_propagation_spread_scales_with_prior_width():
    device = torch.device("cpu")
    model = _ToyModel()
    N_pts = 300
    input_t = torch.randn(N_pts, 6, device=device)
    is_airfoil = np.zeros(N_pts, dtype=bool); is_airfoil[:30] = True
    normals = np.ones((N_pts, 2)) / np.sqrt(2)
    geom_latent = np.zeros(8)
    coef_norm = {
        "cond": {"mean": np.zeros(10), "std": np.ones(10)},
        "output": {},
        "normal": {"min": np.array([-1.0, -1.0]),
                   "max": np.array([1.0, 1.0]),
                   "range": np.array([2.0, 2.0])},
    }
    out_mean = np.zeros(4); out_std = np.ones(4)
    rng = np.random.default_rng(4)
    passes_A, _, _ = propagate_one_geom(
        model, input_t, is_airfoil, normals, geom_latent, coef_norm,
        sample_prior_A, n_draws=100, T=5, device=device, rng=rng,
        out_mean=out_mean, out_std=out_std)
    passes_B, _, _ = propagate_one_geom(
        model, input_t, is_airfoil, normals, geom_latent, coef_norm,
        sample_prior_B, n_draws=100, T=5, device=device, rng=rng,
        out_mean=out_mean, out_std=out_std)
    assert passes_A.shape == (100, 5, len(QOIS))
    dA = decompose_variance(passes_A)
    dB = decompose_variance(passes_B)
    print(f"aleatoric_var A vs B (vmag_max): "
          f"{dA['aleatoric_var'][2]:.3f} vs {dB['aleatoric_var'][2]:.3f}")
    # Prior B wider → larger aleatoric variance on inflow-sensitive QoI
    assert dB["aleatoric_var"][2] > dA["aleatoric_var"][2]


def main():
    test_priors_statistics()
    test_qoi_computation()
    test_recover_normals_inverse()
    test_variance_decomposition_identity()
    test_propagation_spread_scales_with_prior_width()
    print("\ntest_propagate: ALL OK")


if __name__ == "__main__":
    main()
