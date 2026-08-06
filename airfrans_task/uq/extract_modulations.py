"""Standalone modulation extractor.

Recreates the modulation-extraction phase that ends train_sdf.py — reads
best_inr.pt, fits the per-sim SDF latent for every train and test sim via
the CAVIA inner loop, and writes scarce_train_8.npz / scarce_test_8.npz.

Use this when train_sdf.py's own extraction phase hangs or is silently
too slow — this script prints progress every sim and works from an
existing checkpoint (no retraining).

Usage (from airfrans_task/):
    python uq/extract_modulations.py \
        --run_dir E:/OneDrive/Github_Projects/MARIO/airfrans_task/trainings/training_sdf_20260802-234646 \
        --dataset_root C:/Users/phani/OneDrive/Aerodynamic-_Prediction_with_GNN/data/Dataset

Optional: --num_points N  (subsample each sim to N points instead of
using full point clouds; dramatically faster, minor latent-quality cost)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "airfrans_task"))

import airfrans as af
from dataset import SDFDataset
from src.load_models import load_inr
from src.utils_training import graph_outer_step


def extract(model, ds, inner_steps, alpha, latent_dim, device, name):
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    mods = []
    t0 = time.time()
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        batch.modulations = torch.zeros(batch.num_graphs, latent_dim,
                                        device=device)
        out = graph_outer_step(model, batch, inner_steps, alpha,
                               is_train=False)
        mods.append(out["modulations"].detach().cpu().numpy())
        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(ds) - i - 1) / rate
            n_pts = batch.input.shape[0]
            print(f"  [{name}] {i+1}/{len(ds)}  "
                  f"pts={n_pts}  rate={rate:.2f} sim/s  eta={eta:.0f}s")
    return np.concatenate(mods, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="training_sdf_<ts> folder with best_inr.pt + config.yaml")
    ap.add_argument("--dataset_root", required=True,
                    help="AirfRANS Dataset root")
    ap.add_argument("--num_points", type=int, default=None,
                    help="Subsample each sim to this many points "
                         "(default: use full point clouds)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    ckpt_path = run_dir / "best_inr.pt"
    cfg_path = run_dir / "config.yaml"
    assert ckpt_path.exists(), f"missing {ckpt_path}"
    assert cfg_path.exists(), f"missing {cfg_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    with open(cfg_path) as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"checkpoint from epoch {ckpt['epoch']}")

    print("loading raw dataset (may hydrate OneDrive files)...")
    t0 = time.time()
    train_data, train_names = af.dataset.load(
        root=args.dataset_root, task=cfg.dataset.task, train=True)
    test_data, test_names = af.dataset.load(
        root=args.dataset_root, task=cfg.dataset.task, train=False)
    print(f"  loaded in {time.time()-t0:.0f}s: "
          f"train {len(train_data)}, test {len(test_data)}")

    print("building SDFDatasets...")
    train_dataset = SDFDataset(train_data, train_names, is_train=True,
                               num_points=args.num_points)
    test_dataset = SDFDataset(test_data, test_names, is_train=False,
                              coef_norm=train_dataset.coef_norm,
                              num_points=args.num_points)
    train_ds = train_dataset.processed_dataset
    test_ds = test_dataset.processed_dataset

    print("loading model...")
    model = load_inr(ckpt["inr_in"], cfg,
                     cfg.inr.input_dim, cfg.inr.output_dim, device=device)
    model.eval()
    alpha = ckpt["alpha_in"].to(device)
    inner_steps = cfg.optim.inner_steps
    latent_dim = cfg.inr.latent_dim
    print(f"inner_steps={inner_steps}  latent_dim={latent_dim}")

    print("\nextracting train modulations...")
    train_mods = extract(model, train_ds, inner_steps, alpha,
                         latent_dim, device, "train")
    print(f"  train_mods shape: {train_mods.shape}")

    print("\nextracting test modulations...")
    test_mods = extract(model, test_ds, inner_steps, alpha,
                        latent_dim, device, "test")
    print(f"  test_mods shape: {test_mods.shape}")

    mod_dir = run_dir / "modulations"
    mod_dir.mkdir(exist_ok=True)
    train_out = mod_dir / f"{cfg.dataset.task}_train_{latent_dim}.npz"
    test_out = mod_dir / f"{cfg.dataset.task}_test_{latent_dim}.npz"
    np.savez(train_out, train_modulations=train_mods)
    np.savez(test_out, val_modulations=test_mods)
    print(f"\nsaved:\n  {train_out}\n  {test_out}")


if __name__ == "__main__":
    main()
