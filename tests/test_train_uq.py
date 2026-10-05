"""Smoke test for airfrans_task/uq/train_uq.py's building blocks.

Exercises the same code paths as train_uq.py without hydra or the real
dataset:
  - _remove_outliers + _train_val_split
  - AirfransFlowDataset with mode='train'/'val' using synthetic sims
  - MultiScaleModulatedFourierFeaturesUQ + training_step for a few epochs
  - checkpoint save + load roundtrip with the same payload keys
  - best.pt reload yields a model that reproduces predictions

Run from repo root:
    python tests/test_train_uq.py
"""
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "airfrans_task"))

from dataset import AirfransFlowDataset, subsample_dataset
from src.utils_training import training_step
from airfrans_task.uq.models_uq import MultiScaleModulatedFourierFeaturesUQ
from airfrans_task.uq.train_uq import _remove_outliers, _train_val_split

rng = np.random.default_rng(2)


def fake_sim(n_points, u_in=None):
    data = rng.normal(size=(n_points, 12)).astype(np.float64)
    x, y = data[:, 0], data[:, 1]
    data[:, 2] = 50.0 if u_in is None else u_in
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

    # --- fake sims + outlier + latents ---
    n_total, latent_dim = 10, 8
    data_list = [fake_sim(500) for _ in range(n_total)]
    names = [f"fake_{i}" for i in range(n_total - 1)] + [
        "airFoil2D_SST_50.077_-4.416_2.834_4.029_1.0_5.156"]   # outlier
    latents = rng.normal(size=(n_total, latent_dim))

    data_list, names, latents = _remove_outliers(data_list, names, latents)
    assert len(data_list) == n_total - 1
    print(f"outlier removal ok: {n_total} -> {len(data_list)}")

    # --- split ---
    val_size = 3
    (tr_d, tr_n, tr_l, tr_i), (va_d, va_n, va_l, va_i) = _train_val_split(
        data_list, names, latents, val_size=val_size, seed=0)
    assert set(tr_i).isdisjoint(set(va_i))
    assert len(tr_i) + len(va_i) == len(data_list)
    assert len(va_i) == val_size
    # deterministic seed
    (_, _, _, tr_i2), (_, _, _, va_i2) = _train_val_split(
        data_list, names, latents, val_size=val_size, seed=0)
    assert tr_i == tr_i2 and va_i == va_i2, "split not reproducible"
    print(f"split ok: train {len(tr_i)}, val {len(va_i)}, disjoint, reproducible")

    # --- datasets ---
    train_ds = AirfransFlowDataset(tr_d, tr_n, tr_l, mode='train')
    coef_norm = train_ds.coef_norm
    val_ds = AirfransFlowDataset(va_d, va_n, va_l, mode='val',
                                 coef_norm=coef_norm)
    print(f"datasets ok: train {len(train_ds)}, val {len(val_ds)}")

    # --- model + short train loop ---
    model = MultiScaleModulatedFourierFeaturesUQ(
        input_dim=6, output_dim=4, num_frequencies=16, latent_dim=latent_dim + 2,
        width=64, depth=3, depth_hnn=2, include_input=True,
        scales=(0.5, 1.0), scalar_out_dim=0,
        dropout_hnn=0.1, dropout_trunk=0.1,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    tr_losses, va_losses = [], []
    for epoch in range(6):
        model.train()
        sub = subsample_dataset(train_ds, 256)
        loader = DataLoader(sub, batch_size=2, shuffle=True)
        ep_tr = 0.0
        for g in loader:
            g = g.to(device)
            out = training_step(model, g, latent_cond=True, predict_scalars=False)
            opt.zero_grad(); out["loss"].backward(); opt.step()
            ep_tr += out["loss"].item() * len(g)
        tr_losses.append(ep_tr / len(train_ds))

        model.train()   # dropout ON during val, matching train_uq.py
        with torch.no_grad():
            ep_va = 0.0
            vloader = DataLoader(subsample_dataset(val_ds, 256), batch_size=2)
            for g in vloader:
                g = g.to(device)
                out = training_step(model, g, latent_cond=True, predict_scalars=False)
                ep_va += out["loss"].item() * len(g)
            va_losses.append(ep_va / len(val_ds))
    print(f"train loss {tr_losses[0]:.3f} -> {tr_losses[-1]:.3f}")
    print(f"val   loss {va_losses[0]:.3f} -> {va_losses[-1]:.3f}")
    assert tr_losses[-1] < tr_losses[0], "train loss did not decrease"

    # --- checkpoint payload roundtrip ---
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "best.pt"
        payload = {
            "cfg": {"inr": {"dropout_hnn": 0.1, "dropout_trunk": 0.1}},
            "epoch": 5,
            "model_state": model.state_dict(),
            "optim_state": opt.state_dict(),
            "coef_norm": coef_norm,
            "is_best": True,
        }
        torch.save(payload, ckpt_path)

        reloaded = torch.load(ckpt_path, map_location=device, weights_only=False)
        for k in ("cfg", "epoch", "model_state", "optim_state",
                  "coef_norm", "is_best"):
            assert k in reloaded, f"missing key: {k}"

        # rebuild model, load weights, compare eval-mode predictions
        model2 = MultiScaleModulatedFourierFeaturesUQ(
            input_dim=6, output_dim=4, num_frequencies=16,
            latent_dim=latent_dim + 2, width=64, depth=3, depth_hnn=2,
            include_input=True, scales=(0.5, 1.0), scalar_out_dim=0,
            dropout_hnn=0.1, dropout_trunk=0.1,
        ).to(device)
        model2.load_state_dict(reloaded["model_state"])
        model.eval(); model2.eval()
        g0 = val_ds[0]
        y1 = model.modulated_forward(g0.input.to(device),
                                     g0.cond.to(device)[torch.zeros(g0.input.size(0), dtype=torch.long, device=device)])
        y2 = model2.modulated_forward(g0.input.to(device),
                                     g0.cond.to(device)[torch.zeros(g0.input.size(0), dtype=torch.long, device=device)])
        assert torch.allclose(y1, y2, atol=1e-6), "reloaded model differs"
    print("checkpoint save/load roundtrip ok (all keys, eval preds match)")

    print("\ntest_train_uq: ALL OK")


if __name__ == "__main__":
    main()
