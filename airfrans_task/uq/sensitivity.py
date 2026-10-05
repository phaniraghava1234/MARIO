"""Stage 6 — variance-based global sensitivity analysis via Sobol indices.

For each prior over inflow (U_inf, AoA) and each scalar QoI (same five
as stage 5), decomposes the output variance into contributions from
each input:

  - S1[i] : first-order index, fraction of Var[Y] from X_i alone
  - ST[i] : total-order index, S1[i] plus all interactions involving X_i

Interpretation: if S1[U_inf] = 0.85 and S1[AoA] = 0.10 then free-stream
speed explains 85% of QoI variance by itself; angle of attack 10%; the
remaining ~5% comes from the U*AoA interaction (visible as ST - S1).

Method: Saltelli sampling (Sobol' sequence with a specific permutation
scheme) via the SALib library. Needs N * (2*d + 2) model evaluations
per prior per geometry, where d = number of uncertain inputs (here 2).

Reference: Saltelli et al., Global Sensitivity Analysis: The Primer,
Wiley 2008. SALib: Herman & Usher, "SALib: An open-source Python
library for Sensitivity Analysis," JOSS 2017.

Run from repo root:
    python airfrans_task/uq/sensitivity.py \
        --run-dir <run_dir> --root-path <Dataset> \
        --n-geom 5 --N 256 --T 5 --num-points 10000
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

# SALib API: prefer the new location, fall back to old
try:
    from SALib.sample.sobol import sample as saltelli_sample
except ImportError:
    from SALib.sample.saltelli import sample as saltelli_sample
from SALib.analyze.sobol import analyze as sobol_analyze

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "airfrans_task"))

from airfrans_task.uq.evaluate_uq import _build_model, FIELDS
from airfrans_task.uq.propagate import (
    QOIS, compute_qois, recover_normals,
)

# For each prior, define bounds for Saltelli sampling (SALib needs a
# bounded hyper-rectangle). For Prior A (Gaussian), use [mu - 3*sigma,
# mu + 3*sigma] which contains 99.7% of the Gaussian; for Prior B
# (uniform) use the exact support.
PRIOR_BOUNDS = {
    "A_sensor":   {"U": (44.0, 56.0), "AoA": (2.5, 5.5)},
    "B_envelope": {"U": (40.0, 60.0), "AoA": (2.0, 8.0)},
}


def mc_mean_qois_batched(model, input_t, is_airfoil, normals_phys,
                         geom_latent, coef_norm, U_samples, A_samples,
                         T, device, out_mean, out_std):
    """Return QoI arrays of shape (n_samples, n_qois), one row per input
    sample (MC-dropout mean over T passes used as the point estimate)."""
    cond_mean = coef_norm["cond"]["mean"]
    cond_std = coef_norm["cond"]["std"]
    n_samples = len(U_samples)
    rad = np.deg2rad(A_samples)
    u_in = U_samples * np.cos(rad)
    v_in = U_samples * np.sin(rad)
    batch_idx = torch.zeros(input_t.shape[0], dtype=torch.long, device=device)
    Y = np.zeros((n_samples, len(QOIS)))

    model.train()
    with torch.no_grad():
        for i in range(n_samples):
            cond_raw = np.concatenate([geom_latent, [u_in[i], v_in[i]]])
            cond_n = (cond_raw - cond_mean) / cond_std
            cond_t = torch.tensor(cond_n, dtype=torch.float,
                                  device=device).unsqueeze(0)
            # MC mean over T passes = point estimate for Sobol
            pred_sum = torch.zeros_like(input_t[:, :len(FIELDS)])
            for t in range(T):
                pred_sum += model.modulated_forward(input_t, cond_t[batch_idx])
            pred_norm = (pred_sum / T).cpu().numpy()
            fields_phys = pred_norm * out_std + out_mean
            q = compute_qois(fields_phys, is_airfoil,
                             normals_phys, U_samples[i])
            for k, name in enumerate(QOIS):
                Y[i, k] = q[name]
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--root-path", required=True)
    ap.add_argument("--n-geom", type=int, default=5)
    ap.add_argument("--N", type=int, default=256,
                    help="Saltelli base sample size; total evals per prior "
                         "per geom = N * (2d+2) = 6N for d=2")
    ap.add_argument("--T", type=int, default=5,
                    help="MC-dropout passes per input sample (averaged)")
    ap.add_argument("--num-points", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import airfrans as af
    from dataset import AirfransFlowDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    run_dir = Path(args.run_dir)
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(run_dir / "coef_norm.pkl", "rb") as f:
        coef_norm = pickle.load(f)
    ckpt = torch.load(run_dir / "best.pt", map_location=device,
                      weights_only=False)
    print(f"checkpoint: epoch {ckpt['epoch']}")
    model = _build_model(cfg, device)
    model.load_state_dict(ckpt["model_state"])

    raw, names = af.dataset.load(root=args.root_path,
                                 task=cfg["dataset"]["task"], train=False)
    latents_raw = np.load(cfg["dataset"]["test_latents_path"])["val_modulations"]
    test_ds = AirfransFlowDataset(raw, names, latents_raw, mode='test',
                                  coef_norm=coef_norm)
    print(f"test: {len(test_ds)} sims; using first {args.n_geom}")

    out_mean = np.array([coef_norm["output"]["mean"][f] for f in FIELDS])
    out_std = np.array([coef_norm["output"]["std"][f] for f in FIELDS])

    rng = np.random.default_rng(args.seed)
    aggregates = {prior: {q: {"S1": [], "ST": []} for q in QOIS}
                  for prior in PRIOR_BOUNDS}

    for prior_name, bounds in PRIOR_BOUNDS.items():
        problem = {
            "num_vars": 2,
            "names": ["U", "AoA"],
            "bounds": [list(bounds["U"]), list(bounds["AoA"])],
        }
        samples = saltelli_sample(problem, args.N)
        print(f"\nprior {prior_name}: N={args.N}, "
              f"{len(samples):,} evals/geom, bounds={bounds}")

        for gi in range(args.n_geom):
            graph = test_ds[gi]
            name = test_ds.data_names[gi]
            input_t = graph.input.to(device)
            input_arr = graph.input.cpu().numpy()
            is_airfoil = graph.is_airfoil.cpu().numpy() > 0.5
            normals_phys = recover_normals(input_arr, coef_norm)

            # subsample as in propagate.py (keep all surface points)
            if args.num_points > 0 and input_arr.shape[0] > args.num_points:
                n_surf = int(is_airfoil.sum())
                n_bulk = max(args.num_points - n_surf, 0)
                bulk_idx = np.where(~is_airfoil)[0]
                sub_bulk = (rng.choice(bulk_idx, size=n_bulk, replace=False)
                            if n_bulk < len(bulk_idx) else bulk_idx)
                keep = np.sort(np.concatenate([np.where(is_airfoil)[0],
                                                sub_bulk]))
                input_t = input_t[keep]
                is_airfoil = is_airfoil[keep]
                normals_phys = normals_phys[keep]

            t0 = time.time()
            Y = mc_mean_qois_batched(
                model, input_t, is_airfoil, normals_phys,
                test_ds.geom_latents[gi], coef_norm,
                samples[:, 0], samples[:, 1], args.T, device,
                out_mean, out_std)
            dt = time.time() - t0

            for k, q in enumerate(QOIS):
                yk = Y[:, k]
                if yk.var() < 1e-20:
                    aggregates[prior_name][q]["S1"].append(np.array([np.nan,
                                                                     np.nan]))
                    aggregates[prior_name][q]["ST"].append(np.array([np.nan,
                                                                     np.nan]))
                    continue
                Si = sobol_analyze(problem, yk, print_to_console=False)
                aggregates[prior_name][q]["S1"].append(np.array(Si["S1"]))
                aggregates[prior_name][q]["ST"].append(np.array(Si["ST"]))
            print(f"  geom {gi+1:>2}/{args.n_geom}  {name[:30]:<30}  "
                  f"{dt:.1f}s  var(Y)>0: {int((Y.var(0) > 1e-20).sum())}/{len(QOIS)}")

    # ---- Summary table ----
    print(f"\n=== Sobol indices (averaged over n_geom={args.n_geom}) ===")
    for prior_name in PRIOR_BOUNDS:
        print(f"\n-- prior {prior_name} --")
        print(f"{'QoI':<14}  {'S1[U]':>8} {'S1[AoA]':>8}  "
              f"{'ST[U]':>8} {'ST[AoA]':>8}  {'interaction':>12}")
        for q in QOIS:
            S1 = np.nanmean(np.vstack(aggregates[prior_name][q]["S1"]), axis=0)
            ST = np.nanmean(np.vstack(aggregates[prior_name][q]["ST"]), axis=0)
            interaction = float(np.sum(ST) - np.sum(S1))
            print(f"{q:<14}  {S1[0]:>8.3f} {S1[1]:>8.3f}  "
                  f"{ST[0]:>8.3f} {ST[1]:>8.3f}  {interaction:>12.3f}")
    print("\n(S1 close to ST → input acts independently;")
    print(" interaction > 0 → inputs matter jointly, not just alone.)")

    # ---- Plot: grouped bars per QoI per prior ----
    fig, axes = plt.subplots(2, len(QOIS), figsize=(3 * len(QOIS), 6),
                             sharey='row')
    for row, (prior_name, _) in enumerate(PRIOR_BOUNDS.items()):
        for col, q in enumerate(QOIS):
            ax = axes[row, col]
            S1 = np.nanmean(np.vstack(aggregates[prior_name][q]["S1"]), axis=0)
            ST = np.nanmean(np.vstack(aggregates[prior_name][q]["ST"]), axis=0)
            x = np.arange(2)
            w = 0.35
            ax.bar(x - w/2, S1, w, label='S1 (first-order)', color='tab:blue')
            ax.bar(x + w/2, ST, w, label='ST (total)', color='tab:orange')
            ax.set_xticks(x); ax.set_xticklabels(["U", "AoA"])
            ax.set_title(f"{q}\n({prior_name})", fontsize=9)
            ax.set_ylim(0, max(1.0, float(np.nanmax(ST) * 1.1)))
            if col == 0 and row == 0:
                ax.legend(fontsize=8, loc='upper right')
            if col == 0:
                ax.set_ylabel("Sobol index")
    fig.suptitle("Stage 6 — Sobol sensitivity indices")
    fig.tight_layout()

    out_dir = run_dir / "results"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    fig_path = out_dir / f"sensitivity_{ts}.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"\nsensitivity plot: {fig_path}")

    payload = {
        "priors": PRIOR_BOUNDS,
        "variables": ["U", "AoA"],
        "qois": QOIS,
        "n_geom": args.n_geom,
        "N_saltelli": args.N,
        "T": args.T,
        "aggregates": aggregates,
        "seed": args.seed,
        "run_dir": str(run_dir),
    }
    out_path = out_dir / f"sensitivity_{ts}.pt"
    torch.save(payload, out_path)
    print(f"sensitivity data: {out_path}")


if __name__ == "__main__":
    main()
