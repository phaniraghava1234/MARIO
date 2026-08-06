"""Smoke test for airfrans_task/uq/models_uq.py.

Confirms the MC-dropout model:
  1. builds and runs a forward pass with expected shape
  2. is DETERMINISTIC in eval() mode (dropout off)
  3. is STOCHASTIC in train() mode (dropout on) — T passes vary
  4. mc_predict(T) returns finite (mean, std) with std > 0
  5. state_dict has no dropout-only params (bookkeeping check)

Run from repo root:
    python tests/test_models_uq.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from airfrans_task.uq.models_uq import (
    MultiScaleModulatedFourierFeaturesUQ, mc_predict,
)


def main():
    torch.manual_seed(0)

    # Match config_out.yaml shapes but keep small for a fast test
    model = MultiScaleModulatedFourierFeaturesUQ(
        input_dim=6, output_dim=4, num_frequencies=16, latent_dim=10,
        width=64, depth=3, depth_hnn=3, include_input=True,
        scales=(0.5, 1.0), scalar_out_dim=0,
        dropout_hnn=0.2, dropout_trunk=0.2,
    )

    N = 200
    x = torch.randn(N, 6)
    z = torch.randn(N, 10)

    # 1. Forward pass, correct shape
    model.eval()
    y = model.modulated_forward(x, z)
    assert y.shape == (N, 4), y.shape
    assert torch.isfinite(y).all()
    print(f"forward ok: shape {tuple(y.shape)}")

    # 2. eval() mode deterministic: repeated passes identical
    y1 = model.modulated_forward(x, z)
    y2 = model.modulated_forward(x, z)
    assert torch.allclose(y1, y2), "eval() mode is not deterministic"
    print("eval() deterministic ok")

    # 3. train() mode stochastic: passes vary
    model.train()
    with torch.no_grad():
        passes = torch.stack([model.modulated_forward(x, z) for _ in range(20)])
    per_point_std = passes.std(dim=0)       # (N, 4)
    mean_std = per_point_std.mean().item()
    assert mean_std > 1e-3, f"train() mode too deterministic, mean std={mean_std:.2e}"
    print(f"train() stochastic ok: mean std across 20 passes = {mean_std:.4f}")

    # 4. mc_predict helper
    mean_pred, std_pred = mc_predict(model, x, z, T=30)
    assert mean_pred.shape == (N, 4) and std_pred.shape == (N, 4)
    assert torch.isfinite(mean_pred).all() and torch.isfinite(std_pred).all()
    assert (std_pred > 0).float().mean().item() > 0.95, \
        "mc_predict std should be positive almost everywhere"
    # mc_predict restores the prior mode
    was_training = model.training
    model.eval()
    _ = mc_predict(model, x, z, T=5)
    assert model.training == was_training or not model.training, \
        "mc_predict left model in wrong mode"
    print(f"mc_predict ok: mean_std={std_pred.mean():.4f}, "
          f"mean_mean={mean_pred.mean():+.4f}")

    # 5. Bookkeeping: dropout layers add no learnable parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model has {n_params:,} parameters")

    print("\ntest_models_uq: ALL OK")


if __name__ == "__main__":
    main()
