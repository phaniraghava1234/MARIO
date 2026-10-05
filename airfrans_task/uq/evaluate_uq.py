"""Stage 3 — MC dropout evaluation of a trained UQ checkpoint.

Loads a train_uq.py run dir, runs T stochastic forward passes per sim,
computes per-field metrics:
  - RMSE in physical units (mean-prediction accuracy)
  - Coverage at 68% / 95% Gaussian (calibration diagnostic)
  - Pearson r between |error| and predicted std (uncertainty
    informativeness)
plus per-sim breakdowns, and pools a subsample of (mean, std, GT)
triples for downstream calibration (stage 4).

Run from repo root:
    python airfrans_task/uq/evaluate_uq.py \
        --run-dir airfrans_task/uq/trainings/uq_YYYYMMDD-HHMMSS \
        --root-path C:/Users/phani/OneDrive/Aerodynamic-_Prediction_with_GNN/data/Dataset \
        --split test --T 50

CLI args stay explicit — this is inference, no hydra ceremony needed.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "airfrans_task"))

from airfrans_task.uq.models_uq import MultiScaleModulatedFourierFeaturesUQ

FIELDS = ["Velocity-x", "Velocity-y", "Pressure", "Turbulent-viscosity"]


def _build_model(cfg, device):
    inr = cfg["inr"]
    m = MultiScaleModulatedFourierFeaturesUQ(
        input_dim=inr["in_dim"],
        output_dim=inr["out_dim"],
        num_frequencies=inr["num_frequencies"],
        latent_dim=inr["latent_dim"],
        width=inr["hidden_dim"],
        depth=inr["depth"],
        depth_hnn=inr["hypernet_depth"],
        include_input=inr["include_input"],
        scales=tuple(inr["scale"]),
        scalar_out_dim=inr["out_scalar_dim"],
        dropout_hnn=inr["dropout_hnn"],
        dropout_trunk=inr["dropout_trunk"],
    ).to(device)
    return m


def mc_evaluate(model, dataset, coef_norm, T, n_pool_per_sim, device,
                seed=0):
    """Core evaluation loop. Model must already have weights loaded.

    Returns a dict with per-sim stats, overall stats, and pooled
    (mean, std, GT) arrays for downstream calibration.
    """
    out_mean = np.array([coef_norm["output"]["mean"][f] for f in FIELDS])
    out_std = np.array([coef_norm["output"]["std"][f] for f in FIELDS])

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    per_sim, pool_mean, pool_std, pool_gt = [], [], [], []
    rng = np.random.default_rng(seed)

    model.train()  # dropout ON for MC sampling
    with torch.no_grad():
        for i, graph in enumerate(loader):
            graph = graph.to(device)
            batch_idx = graph.batch
            passes = torch.stack([
                model.modulated_forward(graph.input, graph.cond[batch_idx])
                for _ in range(T)
            ])  # (T, N, 4) in normalized space
            mean_n = passes.mean(dim=0).cpu().numpy()   # (N, 4)
            std_n = passes.std(dim=0).cpu().numpy()     # (N, 4)
            gt_n = graph.output.cpu().numpy()           # (N, 4)
            err_n = mean_n - gt_n

            # per-sim stats (in normalized space; ratios invariant)
            mse_n = (err_n ** 2).mean(axis=0)
            cov68 = (np.abs(err_n) <= 1.0 * std_n).mean(axis=0)
            cov95 = (np.abs(err_n) <= 1.96 * std_n).mean(axis=0)
            per_sim.append({
                "idx": i,
                "name": getattr(dataset, "data_names",
                                [f"sim_{i}"] * (i + 1))[i]
                        if hasattr(dataset, "data_names")
                        else f"sim_{i}",
                "n_points": int(mean_n.shape[0]),
                "mse_norm": mse_n.tolist(),
                "cov_68": cov68.tolist(),
                "cov_95": cov95.tolist(),
                "mean_std_norm": std_n.mean(axis=0).tolist(),
            })

            # subsample for pool
            N = mean_n.shape[0]
            k = min(n_pool_per_sim, N)
            sel = rng.choice(N, size=k, replace=False)
            pool_mean.append(mean_n[sel])
            pool_std.append(std_n[sel])
            pool_gt.append(gt_n[sel])

    pool_mean = np.vstack(pool_mean)
    pool_std = np.vstack(pool_std)
    pool_gt = np.vstack(pool_gt)
    pool_err = pool_mean - pool_gt

    overall_mse_norm = (pool_err ** 2).mean(axis=0)
    overall_rmse_phys = np.sqrt(overall_mse_norm) * out_std
    overall_cov68 = (np.abs(pool_err) <= 1.0 * pool_std).mean(axis=0)
    overall_cov95 = (np.abs(pool_err) <= 1.96 * pool_std).mean(axis=0)
    corr = []
    for k in range(len(FIELDS)):
        if pool_std[:, k].std() > 0:
            r = pearsonr(np.abs(pool_err[:, k]), pool_std[:, k]).statistic
        else:
            r = float("nan")
        corr.append(float(r))

    return {
        "T": T,
        "n_sims": len(per_sim),
        "fields": FIELDS,
        "per_sim": per_sim,
        "overall": {
            "mse_norm": overall_mse_norm.tolist(),
            "rmse_phys": overall_rmse_phys.tolist(),
            "cov_68": overall_cov68.tolist(),
            "cov_95": overall_cov95.tolist(),
            "corr_abs_err_std": corr,
        },
        "pool": {
            "mean_norm": pool_mean,
            "std_norm": pool_std,
            "gt_norm": pool_gt,
        },
        "coef_norm_out": {
            "mean": out_mean,
            "std": out_std,
        },
    }


def evaluate_from_run_dir(run_dir, root_path, split, T, n_pool_per_sim,
                          out_name=None):
    """Load a train_uq.py run dir, build the requested split dataset,
    run mc_evaluate, and save the result under run_dir/results/."""
    import airfrans as af
    from dataset import AirfransFlowDataset
    from airfrans_task.uq.train_uq import _remove_outliers

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    run_dir = Path(run_dir)
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(run_dir / "coef_norm.pkl", "rb") as f:
        coef_norm = pickle.load(f)
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "last.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"checkpoint: {ckpt_path} (epoch {ckpt['epoch']})")

    model = _build_model(cfg, device)
    model.load_state_dict(ckpt["model_state"])

    task = cfg["dataset"]["task"]
    if split == "test":
        raw, names = af.dataset.load(root=root_path, task=task, train=False)
        latents = np.load(cfg["dataset"]["test_latents_path"])["val_modulations"]
        dataset = AirfransFlowDataset(raw, names, latents, mode='test',
                                      coef_norm=coef_norm)
    elif split == "val":
        raw, names = af.dataset.load(root=root_path, task=task, train=True)
        latents = np.load(cfg["dataset"]["train_latents_path"])["train_modulations"]
        raw, names, latents = _remove_outliers(raw, names, latents)
        with open(run_dir / "split.yaml") as f:
            split_info = yaml.safe_load(f)
        val_idx = split_info["val_idx"]
        raw = [raw[i] for i in val_idx]
        names = [names[i] for i in val_idx]
        latents = latents[val_idx]
        dataset = AirfransFlowDataset(raw, names, latents, mode='val',
                                      coef_norm=coef_norm)
    else:
        raise ValueError(f"split must be 'test' or 'val', got {split!r}")
    print(f"split={split}: {len(dataset)} sims")

    result = mc_evaluate(model, dataset, coef_norm, T, n_pool_per_sim, device)
    result["split"] = split
    result["run_dir"] = str(run_dir)
    result["checkpoint_epoch"] = int(ckpt["epoch"])

    out_dir = run_dir / "results"
    out_dir.mkdir(exist_ok=True)
    if out_name is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_name = f"eval_{split}_T{T}_{ts}.pt"
    torch.save(result, out_dir / out_name)

    _print_summary(result)
    print(f"\nsaved: {out_dir / out_name}")
    return result


def _print_summary(result):
    o = result["overall"]
    print(f"\n=== {result['split']} evaluation, T={result['T']}, "
          f"n_sims={result['n_sims']} ===")
    print(f"{'field':<22} {'RMSE (phys)':>12} {'cov68':>8} {'cov95':>8} "
          f"{'r(|err|,std)':>14}")
    for k, f in enumerate(FIELDS):
        print(f"{f:<22} {o['rmse_phys'][k]:>12.4e} "
              f"{o['cov_68'][k]:>8.3f} {o['cov_95'][k]:>8.3f} "
              f"{o['corr_abs_err_std'][k]:>14.3f}")
    print("(ideal cov68=0.680, cov95=0.950; r>0 means std tracks error)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="train_uq.py run directory")
    p.add_argument("--root-path", required=True, help="AirfRANS Dataset root")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--T", type=int, default=50, help="MC forward passes")
    p.add_argument("--n-pool", type=int, default=5000,
                   help="points subsampled per sim for pooled arrays")
    p.add_argument("--out-name", default=None,
                   help="custom output filename (default: eval_<split>_T<T>_<ts>.pt)")
    args = p.parse_args()
    evaluate_from_run_dir(args.run_dir, args.root_path, args.split, args.T,
                          args.n_pool, args.out_name)
