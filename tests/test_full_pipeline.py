"""Smoke test: tiny end-to-end flow-model training on synthetic data.

Mirrors airfrans_task/train.py (dataset -> DataLoader -> training_step
-> checkpointed weights -> de-normalized prediction) without hydra,
wandb, or the AirfRANS download. CPU, ~30s. Run from repo root:
    python tests/test_full_pipeline.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "airfrans_task"))

from dataset import AirfransFlowDataset, subsample_dataset
from src.models import MultiScaleModulatedFourierFeatures
from src.utils_training import training_step

rng = np.random.default_rng(1)


def fake_sim(n_points):
    data = rng.normal(size=(n_points, 12)).astype(np.float64)
    x, y = data[:, 0], data[:, 1]
    data[:, 2] = 50.0 + rng.normal(scale=2.0)   # u_in, sim-constant-ish
    data[:, 3] = rng.normal(scale=2.0)          # v_in
    data[:, 4] = np.abs(data[:, 4])             # sdf
    # smooth learnable targets
    data[:, 7] = np.sin(x) + data[:, 2] * 0.01
    data[:, 8] = np.cos(y)
    data[:, 9] = x * y * 0.1
    data[:, 10] = np.abs(np.sin(x * y)) * 0.01
    data[:, 11] = (data[:, 4] < 0.05)
    return data


def main():
    torch.manual_seed(0)
    n_sims, latent_dim = 4, 8
    data_list = [fake_sim(600) for _ in range(n_sims)]
    names = [f"fake_sim_{i}" for i in range(n_sims)]
    latents = rng.normal(size=(n_sims, latent_dim))

    train_ds = AirfransFlowDataset(data_list, names, latents, mode="train")
    coef_norm = train_ds.coef_norm
    test_ds = AirfransFlowDataset(data_list, names, latents, mode="test",
                                  coef_norm=coef_norm)

    model = MultiScaleModulatedFourierFeatures(
        input_dim=6, output_dim=4, num_frequencies=16,
        latent_dim=latent_dim + 2, width=64, depth=3, depth_hnn=2,
        include_input=True, scales=[0.5, 1], scalar_out_dim=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for epoch in range(15):
        sub = subsample_dataset(train_ds, 256)
        loader = DataLoader(sub, batch_size=2, shuffle=True)
        epoch_loss = 0.0
        for graph in loader:
            model.train()
            out = training_step(model, graph, latent_cond=True,
                                predict_scalars=False)
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += out["loss"].item() * len(graph)
        losses.append(epoch_loss / len(train_ds))
    print(f"loss epoch 0: {losses[0]:.4f} -> epoch {len(losses)-1}: "
          f"{losses[-1]:.4f}")
    assert losses[-1] < losses[0], "loss did not decrease"

    # checkpoint roundtrip (as train.py does)
    ckpt = {"inr_in": model.state_dict()}
    model2 = MultiScaleModulatedFourierFeatures(
        input_dim=6, output_dim=4, num_frequencies=16,
        latent_dim=latent_dim + 2, width=64, depth=3, depth_hnn=2,
        include_input=True, scales=[0.5, 1], scalar_out_dim=0)
    model2.load_state_dict(ckpt["inr_in"])
    model2.eval()

    # test-time inference + de-normalization (as train.py / evaluate.py)
    fields = ["Velocity-x", "Velocity-y", "Pressure", "Turbulent-viscosity"]
    mean = np.array([coef_norm["output"]["mean"][f] for f in fields])
    std = np.array([coef_norm["output"]["std"][f] for f in fields])
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    preds = []
    with torch.no_grad():
        for graph in loader:
            field_n = model2.modulated_forward(
                graph.input, graph.cond[graph.batch]).numpy()
            preds.append(field_n * std + mean)
    assert len(preds) == n_sims
    assert preds[0].shape == (600, 4)
    assert all(np.isfinite(p).all() for p in preds)
    # de-normalized predictions should live on the physical scale
    assert abs(preds[0][:, 0].mean() - mean[0]) < 5 * std[0]
    print("inference + de-normalization ok")

    print("\ntest_full_pipeline: ALL OK")


if __name__ == "__main__":
    main()
