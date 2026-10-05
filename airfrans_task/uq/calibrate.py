"""Stage 4 — variance calibration of MC-dropout uncertainty.

Fits a per-field scalar `s_k` by minimizing Gaussian NLL on the
validation split, then applies it to the test split and reports
before/after coverage, NLL, and reliability.

Closed-form MLE for a scalar on Gaussian likelihood:
    s_k = sqrt(mean_i((err_ik / sigma_ik)^2))
This is the standard isotropic variance scaling (Levi et al. 2022,
"Evaluating and Calibrating Uncertainty Prediction in Regression
Tasks"; also used in Laves et al. 2020, "Well-calibrated regression
uncertainty in medical imaging").

Run from repo root:
    python airfrans_task/uq/calibrate.py \
        --run-dir airfrans_task/uq/trainings/uq_YYYYMMDD-HHMMSS \
        --root-path <AirfRANS Dataset path> \
        --T 50
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import norm, pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "airfrans_task"))

from airfrans_task.uq.evaluate_uq import mc_evaluate, _build_model, FIELDS


def _load_val_dataset(cfg, coef_norm, run_dir, root_path):
    import airfrans as af
    from dataset import AirfransFlowDataset
    from airfrans_task.uq.train_uq import _remove_outliers

    raw, names = af.dataset.load(root=root_path, task=cfg["dataset"]["task"],
                                 train=True)
    latents = np.load(cfg["dataset"]["train_latents_path"])["train_modulations"]
    raw, names, latents = _remove_outliers(raw, names, latents)
    with open(run_dir / "split.yaml") as f:
        val_idx = yaml.safe_load(f)["val_idx"]
    raw = [raw[i] for i in val_idx]
    names = [names[i] for i in val_idx]
    latents = latents[val_idx]
    return AirfransFlowDataset(raw, names, latents, mode='val',
                               coef_norm=coef_norm)


def _load_test_dataset(cfg, coef_norm, root_path):
    import airfrans as af
    from dataset import AirfransFlowDataset

    raw, names = af.dataset.load(root=root_path, task=cfg["dataset"]["task"],
                                 train=False)
    latents = np.load(cfg["dataset"]["test_latents_path"])["val_modulations"]
    return AirfransFlowDataset(raw, names, latents, mode='test',
                               coef_norm=coef_norm)


def gaussian_nll(err, sigma):
    """Mean per-point Gaussian NLL for a single field (1D arrays)."""
    sigma = np.maximum(sigma, 1e-8)
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2)
                         + err ** 2 / (2 * sigma ** 2)))


def fit_scalar_s(err, sigma):
    """Closed-form MLE for sigma_cal = s * sigma."""
    sigma = np.maximum(sigma, 1e-8)
    z2 = (err / sigma) ** 2
    return float(np.sqrt(np.mean(z2)))


def coverage(err, sigma, nominal):
    """Fraction of points inside the two-sided Gaussian interval."""
    z = norm.ppf(0.5 + nominal / 2)   # 1.0 for 68%, 1.96 for 95%
    return float(np.mean(np.abs(err) <= z * sigma))


def reliability_curve(err, sigma, n_bins=20):
    """Expected vs observed CDF of |err|/sigma against a half-normal.

    Returns (expected, observed) arrays each of length n_bins.
    Perfect calibration: observed == expected along the diagonal.
    """
    sigma = np.maximum(sigma, 1e-8)
    z = np.abs(err / sigma)
    # expected quantiles of |N(0,1)| at probabilities p ∈ (0, 1)
    probs = np.linspace(0.05, 0.95, n_bins)
    expected = norm.ppf(0.5 + probs / 2)       # cutoffs in standard units
    observed = np.array([(z <= e).mean() for e in expected])
    return probs, observed


def calibrate(run_dir, root_path, T, n_pool_per_sim, seed=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    run_dir = Path(run_dir)
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(run_dir / "coef_norm.pkl", "rb") as f:
        coef_norm = pickle.load(f)
    ckpt_path = run_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"checkpoint: {ckpt_path} (epoch {ckpt['epoch']})")

    model = _build_model(cfg, device)
    model.load_state_dict(ckpt["model_state"])

    # --- VAL: fit s_k ---
    print("\nrunning MC evaluation on VAL split (for fitting s)...")
    val_ds = _load_val_dataset(cfg, coef_norm, run_dir, root_path)
    print(f"val: {len(val_ds)} sims")
    val = mc_evaluate(model, val_ds, coef_norm, T, n_pool_per_sim, device,
                      seed=seed)
    v_mean = val["pool"]["mean_norm"]
    v_std = val["pool"]["std_norm"]
    v_gt = val["pool"]["gt_norm"]
    v_err = v_mean - v_gt

    s_per_field = np.array([fit_scalar_s(v_err[:, k], v_std[:, k])
                            for k in range(len(FIELDS))])
    print(f"\nfitted s per field (val):")
    for k, f in enumerate(FIELDS):
        print(f"  {f:<22} s = {s_per_field[k]:.4f}")

    # --- TEST: apply s_k and report before/after ---
    print("\nrunning MC evaluation on TEST split (for reporting)...")
    test_ds = _load_test_dataset(cfg, coef_norm, root_path)
    print(f"test: {len(test_ds)} sims")
    test = mc_evaluate(model, test_ds, coef_norm, T, n_pool_per_sim, device,
                       seed=seed)
    t_mean = test["pool"]["mean_norm"]
    t_std = test["pool"]["std_norm"]
    t_gt = test["pool"]["gt_norm"]
    t_err = t_mean - t_gt
    t_std_cal = t_std * s_per_field[None, :]

    out_std_phys = np.array([coef_norm["output"]["std"][f] for f in FIELDS])

    # metrics before/after
    def summarize(sigma_pool, label):
        rows = []
        for k, f in enumerate(FIELDS):
            e, s = t_err[:, k], sigma_pool[:, k]
            rows.append({
                "field": f,
                "nll": gaussian_nll(e, s),
                "cov_68": coverage(e, s, 0.68),
                "cov_95": coverage(e, s, 0.95),
                "mean_sigma_phys": float(s.mean() * out_std_phys[k]),
                "r_abs_err_std": (
                    float(pearsonr(np.abs(e), s).statistic)
                    if s.std() > 0 else float("nan")
                ),
            })
        return label, rows

    pre_label, pre_rows = summarize(t_std, "pre-calibration (s=1)")
    post_label, post_rows = summarize(t_std_cal, "post-calibration")

    print(f"\n=== TEST metrics, before vs after (T={T}, n_sims={len(test_ds)}) ===")
    print(f"{'field':<22} {'cov68 pre→post':>18} {'cov95 pre→post':>18} "
          f"{'NLL pre→post':>18} {'r pre→post':>18}")
    for pre, post in zip(pre_rows, post_rows):
        print(f"{pre['field']:<22} "
              f"{pre['cov_68']:>7.3f} → {post['cov_68']:>5.3f}    "
              f"{pre['cov_95']:>7.3f} → {post['cov_95']:>5.3f}    "
              f"{pre['nll']:>7.3f} → {post['nll']:>5.3f}    "
              f"{pre['r_abs_err_std']:>7.3f} → {post['r_abs_err_std']:>5.3f}")
    print("(ideal cov68=0.680, cov95=0.950)")

    # reliability plots (one subplot per field)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for k, f in enumerate(FIELDS):
        ax = axes[k // 2, k % 2]
        probs, obs_pre = reliability_curve(t_err[:, k], t_std[:, k])
        _, obs_post = reliability_curve(t_err[:, k], t_std_cal[:, k])
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='ideal')
        ax.plot(probs, obs_pre, 'o-', color='tab:red', label='pre (s=1)')
        ax.plot(probs, obs_post, 's-', color='tab:blue',
                label=f'post (s={s_per_field[k]:.2f})')
        ax.set_xlabel("expected"); ax.set_ylabel("observed")
        ax.set_title(f)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    fig.suptitle("Reliability diagrams — predicted vs observed CDF of |err|/σ")
    fig.tight_layout()

    out_dir = run_dir / "results"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    fig_path = out_dir / f"calibration_reliability_{ts}.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"\nreliability plot: {fig_path}")

    # save all state
    payload = {
        "s_per_field": s_per_field,
        "fields": FIELDS,
        "val": {
            "n_sims": val["n_sims"],
            "pool_err": v_err,
            "pool_std": v_std,
        },
        "test_pre": {
            "pool_err": t_err,
            "pool_std": t_std,
            "rows": pre_rows,
        },
        "test_post": {
            "pool_std_cal": t_std_cal,
            "rows": post_rows,
        },
        "T": T,
        "seed": seed,
        "run_dir": str(run_dir),
        "checkpoint_epoch": int(ckpt["epoch"]),
    }
    out_path = out_dir / f"calibration_T{T}_{ts}.pt"
    torch.save(payload, out_path)
    print(f"calibration data: {out_path}")
    return payload


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--root-path", required=True)
    p.add_argument("--T", type=int, default=50)
    p.add_argument("--n-pool", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    calibrate(args.run_dir, args.root_path, args.T, args.n_pool, args.seed)
