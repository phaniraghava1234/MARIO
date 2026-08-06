"""Smoke test: dataset.py + src/models.py on synthetic data.

No AirfRANS download, no GPU, no wandb. Run from repo root:
    python tests/test_pipeline.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "airfrans_task"))

from dataset import AirfransFlowDataset, SDFDataset, make_bl_layer
from src.models import ModulatedFourierFeatures, MultiScaleModulatedFourierFeatures

rng = np.random.default_rng(0)


def fake_sim(n_points):
    """Synthetic (N, 12) array matching the AirfRANS column layout."""
    data = rng.normal(size=(n_points, 12)).astype(np.float64)
    data[:, 4] = np.abs(data[:, 4])          # sdf >= 0
    data[:, 11] = (data[:, 4] < 0.05)        # is_airfoil boolean
    return data


def main():
    n_sims, latent_dim = 4, 8
    data_list = [fake_sim(rng.integers(500, 800)) for _ in range(n_sims)]
    names = [f"fake_sim_{i}" for i in range(n_sims)]
    latents = rng.normal(size=(n_sims, latent_dim))

    # --- make_bl_layer ---
    bl = make_bl_layer({"point_cloud_field/distance_function": data_list[0][:, 4]})
    assert bl.shape == (len(data_list[0]),)
    assert bl.min() >= 0 and bl.max() <= 1
    print("make_bl_layer ok")

    # --- SDFDataset (stage 1) ---
    sdf_train = SDFDataset(data_list, names, is_train=True, num_points=256)
    sdf_test = SDFDataset(data_list, names, is_train=False,
                          coef_norm=sdf_train.coef_norm)
    d0 = sdf_train.processed_dataset[0]
    assert d0.input.shape[1] == 2 and d0.output.shape[1] == 1
    assert d0.input.shape[0] == 256
    print("SDFDataset ok")

    # --- AirfransFlowDataset (stage 2) ---
    train_ds = AirfransFlowDataset(data_list, names, latents, mode="train")
    test_ds = AirfransFlowDataset(data_list, names, latents, mode="test",
                                  coef_norm=train_ds.coef_norm)
    g = train_ds[0]
    assert g.input.shape[1] == 6, g.input.shape       # x,y,sdf,nx,ny,bl
    assert g.output.shape[1] == 4, g.output.shape     # u,v,p,nut
    assert g.cond.shape == (1, latent_dim + 2), g.cond.shape
    # inputs min-max normalized to [-1, 1]
    assert g.input.min() >= -1.0001 and g.input.max() <= 1.0001

    # norm/denorm roundtrip on outputs
    c = train_ds.coef_norm
    fields = ["Velocity-x", "Velocity-y", "Pressure", "Turbulent-viscosity"]
    mean = np.array([c["output"]["mean"][f] for f in fields])
    std = np.array([c["output"]["std"][f] for f in fields])
    raw = data_list[0][:, [7, 8, 9, 10]]
    back = g.output.numpy() * std + mean
    assert np.allclose(back, raw, atol=1e-4), "denorm roundtrip failed"
    print("AirfransFlowDataset ok (denorm roundtrip passed)")

    # --- ModulatedFourierFeatures (SDF encoder architecture) ---
    torch.manual_seed(0)
    enc = ModulatedFourierFeatures(
        input_dim=2, output_dim=1, num_frequencies=32, latent_dim=8,
        width=64, depth=3, frequency_embedding="gaussian",
        include_input=True, scale=0.5)
    x = torch.randn(100, 2)
    z = torch.randn(100, 8)
    out = enc.modulated_forward(x, z)
    assert out.shape == (100, 1) and torch.isfinite(out).all()
    print("ModulatedFourierFeatures ok")

    # --- MultiScaleModulatedFourierFeatures (flow model) ---
    flow = MultiScaleModulatedFourierFeatures(
        input_dim=6, output_dim=4, num_frequencies=16, latent_dim=10,
        width=64, depth=3, depth_hnn=2, include_input=True,
        scales=[0.5, 1], scalar_out_dim=0)
    x = torch.randn(200, 6)
    z = torch.randn(200, 10)
    out = flow.modulated_forward(x, z)
    assert out.shape == (200, 4) and torch.isfinite(out).all()
    print("MultiScaleModulatedFourierFeatures ok")

    print("\ntest_pipeline: ALL OK")


if __name__ == "__main__":
    main()
