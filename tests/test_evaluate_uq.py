"""Smoke test for airfrans_task/uq/evaluate_uq.py.

Tests mc_evaluate() directly on a tiny synthetic dataset + tiny model,
without needing a real train_uq run dir. Verifies:
  - result dict has the expected keys and shapes
  - stds are non-negative
  - coverages are in [0, 1] and roughly plausible on synthetic data
  - correlation between |error| and std returns a finite scalar
  - per-sim stats match n_sims

Run from repo root:
    python tests/test_evaluate_uq.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "airfrans_task"))

from dataset import AirfransFlowDataset
from airfrans_task.uq.models_uq import MultiScaleModulatedFourierFeaturesUQ
from airfrans_task.uq.evaluate_uq import mc_evaluate, FIELDS

rng = np.random.default_rng(3)


def fake_sim(n_points):
    data = rng.normal(size=(n_points, 12)).astype(np.float64)
    x, y = data[:, 0], data[:, 1]
    data[:, 2] = 50.0 + rng.normal(scale=2.0)
    data[:, 3] = rng.normal(scale=2.0)
    data[:, 4] = np.abs(data[:, 4])
    data[:, 7] = np.sin(x) + data[:, 2] * 0.01
    data[:, 8] = np.cos(y)
    data[:, 9] = x * y * 0.1
    data[:, 10] = np.abs(np.sin(x * y)) * 0.01
    data[:, 11] = (data[:, 4] < 0.05)
    return data


def main():
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_sims, latent_dim, n_points = 4, 8, 300
    data_list = [fake_sim(n_points) for _ in range(n_sims)]
    names = [f"fake_{i}" for i in range(n_sims)]
    latents = rng.normal(size=(n_sims, latent_dim))

    train_ds = AirfransFlowDataset(data_list, names, latents, mode='train')
    coef_norm = train_ds.coef_norm

    model = MultiScaleModulatedFourierFeaturesUQ(
        input_dim=6, output_dim=4, num_frequencies=16, latent_dim=latent_dim + 2,
        width=64, depth=3, depth_hnn=2, include_input=True,
        scales=(0.5, 1.0), scalar_out_dim=0,
        dropout_hnn=0.2, dropout_trunk=0.2,
    ).to(device)

    T, n_pool = 10, 100
    result = mc_evaluate(model, train_ds, coef_norm, T, n_pool, device, seed=0)

    # Keys
    for k in ("T", "n_sims", "fields", "per_sim", "overall", "pool",
              "coef_norm_out"):
        assert k in result, f"missing key: {k}"
    assert result["fields"] == FIELDS
    assert result["n_sims"] == n_sims
    assert len(result["per_sim"]) == n_sims
    print(f"keys ok, n_sims={n_sims}")

    # Pool shapes
    pool = result["pool"]
    expected = min(n_pool, n_points) * n_sims
    for k in ("mean_norm", "std_norm", "gt_norm"):
        arr = pool[k]
        assert arr.shape == (expected, 4), (k, arr.shape)
        assert np.isfinite(arr).all(), f"{k} has non-finite"
    print(f"pool shapes ok: ({expected}, 4) for mean/std/gt")

    # Stds non-negative
    assert (pool["std_norm"] >= 0).all()
    assert pool["std_norm"].mean() > 0
    print(f"pool std_norm min={pool['std_norm'].min():.4f} "
          f"mean={pool['std_norm'].mean():.4f}")

    # Overall metrics in expected ranges
    o = result["overall"]
    for kname in ("mse_norm", "rmse_phys", "cov_68", "cov_95",
                  "corr_abs_err_std"):
        assert len(o[kname]) == 4, kname
    for c in o["cov_68"] + o["cov_95"]:
        assert 0.0 <= c <= 1.0, f"coverage out of [0,1]: {c}"
    for r in o["corr_abs_err_std"]:
        assert np.isfinite(r), f"corr not finite: {r}"
    print("overall metrics ok:")
    for k, f in enumerate(FIELDS):
        print(f"  {f:<22} rmse_phys={o['rmse_phys'][k]:.3e} "
              f"cov68={o['cov_68'][k]:.3f} cov95={o['cov_95'][k]:.3f} "
              f"r={o['corr_abs_err_std'][k]:+.3f}")

    # Per-sim entries have all expected fields
    for s in result["per_sim"]:
        for k in ("idx", "name", "n_points", "mse_norm",
                  "cov_68", "cov_95", "mean_std_norm"):
            assert k in s, f"per_sim missing {k}"
    print("per_sim entries ok")

    print("\ntest_evaluate_uq: ALL OK")


if __name__ == "__main__":
    main()
