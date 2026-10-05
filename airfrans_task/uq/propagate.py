"""Stage 5 — propagate inflow uncertainty through the trained surrogate.

Two priors over inflow (U_inf, AoA), both reported side-by-side:

  Prior A — sensor noise at nominal cruise.
    U_inf ~ N(50, 2) m/s      ~ 4% airspeed-sensor noise
    AoA   ~ N(4, 0.5) deg     ~ 0.5 deg AoA-sensor noise
  Prior B — mission envelope.
    U_inf ~ U(40, 60) m/s     ~ gust / speed-variation envelope
    AoA   ~ U(2, 8) deg       ~ small maneuvers

Both lie inside the AirfRANS training envelope.

Scalar QoIs (five, physical units):
  - p_surf_min  : suction peak pressure on airfoil surface (Pa)
  - p_surf_mean : mean surface pressure (Pa)
  - vmag_max    : peak velocity magnitude in domain (m/s)
  - Cl_proxy    : lift coeff from surface pressure integral (dim.less)
  - Cd_proxy    : pressure-drag coeff from surface integral (dim.less)
                  (viscous drag not included — surrogate predicts only
                   mean pressure, so this is the pressure-drag part)

Aleatoric vs epistemic decomposition via the total-variance identity:
    Var[Q]  =  E[Var[Q|U,A]]  +  Var[E[Q|U,A]]
            = epistemic (model)+ aleatoric (inflow)
Per draw we run T MC-dropout passes and record the T QoI samples;
mean of their variances across draws = epistemic var; variance of
their per-draw means across draws = aleatoric var; sum = total var.
Reference: Kendall & Gal, "What Uncertainties Do We Need in Bayesian
Deep Learning for Computer Vision?", NeurIPS 2017.

Reports empirical 68% / 95% intervals (p16-p84, p05-p95) alongside
total mean and std.

Run from repo root:
    python airfrans_task/uq/propagate.py \
        --run-dir <run_dir> --root-path <Dataset> \
        --n-geom 20 --n-draws 500 --T 10
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "airfrans_task"))

from airfrans_task.uq.evaluate_uq import _build_model, FIELDS

QOIS = ["p_surf_min", "p_surf_mean", "vmag_max", "Cl_proxy", "Cd_proxy"]
RHO_AIR = 1.225   # kg/m^3 standard conditions
CHORD = 1.0       # m (AirfRANS airfoils are normalized to unit chord)


# ---- Priors --------------------------------------------------------------

def sample_prior_A(n, rng):
    return rng.normal(50.0, 2.0, size=n), rng.normal(4.0, 0.5, size=n)


def sample_prior_B(n, rng):
    return rng.uniform(40.0, 60.0, size=n), rng.uniform(2.0, 8.0, size=n)


PRIORS = {"A_sensor": sample_prior_A, "B_envelope": sample_prior_B}


# ---- QoI computation -----------------------------------------------------

def compute_qois(fields_phys, is_airfoil, normals_phys, U_inf,
                 rho=RHO_AIR, c=CHORD):
    """fields_phys: (N, 4) physical units [u, v, p, nu_t].
    normals_phys: (N, 2) physical surface normals (used only on is_airfoil).
    U_inf: scalar free-stream speed used to non-dimensionalize Cl/Cd."""
    p = fields_phys[:, 2]
    u = fields_phys[:, 0]
    v = fields_phys[:, 1]
    vmag = np.sqrt(u ** 2 + v ** 2)
    surf = is_airfoil
    if surf.sum() == 0:
        surf = np.ones_like(surf, dtype=bool)

    nx = normals_phys[surf, 0]
    ny = normals_phys[surf, 1]
    p_surf = p[surf]
    # Pressure force per unit chord with unit area weighting per surface
    # point (uniform mesh, so this is proportional to true force; UQ ratios
    # unaffected).
    # Sign convention: AirfRANS surface normals point INWARD (into the
    # airfoil, derived from SDF gradient). Force on body from a pressure
    # field is F = -∫ p * n_outward dA = +∫ p * n_inward dA.
    Fx = float((p_surf * nx).sum())       # drag (streamwise)
    Fy = float((p_surf * ny).sum())       # lift (perpendicular to flow)
    q_inf = 0.5 * rho * max(U_inf, 1e-6) ** 2
    denom = q_inf * c * max(surf.sum(), 1)   # normalize by surface pts count
    return {
        "p_surf_min": float(p_surf.min()),
        "p_surf_mean": float(p_surf.mean()),
        "vmag_max": float(vmag.max()),
        "Cl_proxy": Fy / denom,
        "Cd_proxy": Fx / denom,
    }


# ---- Per-sim propagation -------------------------------------------------

def propagate_one_geom(model, input_t, is_airfoil, normals_phys,
                       geom_latent, coef_norm, prior_fn,
                       n_draws, T, device, rng, out_mean, out_std):
    """Returns:
        qoi_per_pass : (n_draws, T, n_qois) QoI value at every single pass
        (U, A) : the sampled inflow arrays
    """
    cond_mean = coef_norm["cond"]["mean"]
    cond_std = coef_norm["cond"]["std"]
    U, A = prior_fn(n_draws, rng)
    rad = np.deg2rad(A)
    u_in = U * np.cos(rad)
    v_in = U * np.sin(rad)

    qoi_per_pass = np.zeros((n_draws, T, len(QOIS)))
    batch_idx = torch.zeros(input_t.shape[0], dtype=torch.long, device=device)

    model.train()  # dropout ON for MC
    with torch.no_grad():
        for i in range(n_draws):
            cond_raw = np.concatenate([geom_latent, [u_in[i], v_in[i]]])
            cond_n = (cond_raw - cond_mean) / cond_std
            cond_t = torch.tensor(cond_n, dtype=torch.float,
                                  device=device).unsqueeze(0)
            # run T dropout passes individually to keep each separate
            for t in range(T):
                pred_norm = model.modulated_forward(input_t,
                                                    cond_t[batch_idx])
                pred_norm = pred_norm.cpu().numpy()
                fields_phys = pred_norm * out_std + out_mean
                q = compute_qois(fields_phys, is_airfoil,
                                 normals_phys, U[i])
                for k, name in enumerate(QOIS):
                    qoi_per_pass[i, t, k] = q[name]
    return qoi_per_pass, U, A


def decompose_variance(qoi_per_pass):
    """Total-variance decomposition per QoI.

    qoi_per_pass: (n_draws, T, n_qois)
    returns dict of (n_qois,) arrays:
        total_mean       = overall mean of all passes
        aleatoric_var    = Var over draws of per-draw mean (inflow)
        epistemic_var    = Mean over draws of per-draw var (model)
        total_var        = sum
        per_draw_mean    = (n_draws, n_qois) means per draw (for quantiles)
    """
    per_draw_mean = qoi_per_pass.mean(axis=1)        # (n_draws, n_qois)
    per_draw_var = qoi_per_pass.var(axis=1, ddof=0)  # (n_draws, n_qois)
    aleatoric_var = per_draw_mean.var(axis=0, ddof=0)   # (n_qois,)
    epistemic_var = per_draw_var.mean(axis=0)           # (n_qois,)
    total_var = aleatoric_var + epistemic_var
    return {
        "total_mean": per_draw_mean.mean(axis=0),
        "aleatoric_var": aleatoric_var,
        "epistemic_var": epistemic_var,
        "total_var": total_var,
        "per_draw_mean": per_draw_mean,
    }


# ---- Normals de-normalization --------------------------------------------

def recover_normals(input_arr, coef_norm):
    """Input was min-max normalized to [-1, 1]. Recover physical normals."""
    nmin = np.asarray(coef_norm["normal"]["min"])
    nrange = np.asarray(coef_norm["normal"]["range"])
    nx = (input_arr[:, 3] + 1) / 2 * nrange[0] + nmin[0]
    ny = (input_arr[:, 4] + 1) / 2 * nrange[1] + nmin[1]
    return np.stack([nx, ny], axis=-1)


# ---- Main ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--root-path", required=True)
    ap.add_argument("--n-geom", type=int, default=20)
    ap.add_argument("--n-draws", type=int, default=500)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-points", type=int, default=10000,
                    help="subsample each sim to this many points "
                         "(0 = full point cloud, slow)")
    ap.add_argument("--out-tag", type=str, default=None,
                    help="output filename tag; reused for resume")
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
    out_dir = run_dir / "results"
    out_dir.mkdir(exist_ok=True)
    tag = args.out_tag or time.strftime("%Y%m%d-%H%M%S")
    partial_path = out_dir / f"propagation_{tag}_partial.pt"

    # resume from a partial run if present with the same tag
    if partial_path.exists():
        prev = torch.load(partial_path, map_location="cpu",
                          weights_only=False)
        results = prev["results"]
        geom_names = prev["geom_names"]
        start_gi = len(geom_names)
        print(f"resuming from {partial_path}: {start_gi} geoms already done")
    else:
        results = {prior: [] for prior in PRIORS}
        geom_names = []
        start_gi = 0

    t0 = time.time()
    for gi in range(start_gi, args.n_geom):
        graph = test_ds[gi]
        name = test_ds.data_names[gi]
        geom_names.append(name)
        input_t = graph.input.to(device)
        input_arr = graph.input.cpu().numpy()
        is_airfoil = graph.is_airfoil.cpu().numpy() > 0.5
        normals_phys = recover_normals(input_arr, coef_norm)

        # subsample per sim: keep ALL surface points (small), subsample bulk
        if args.num_points > 0 and input_arr.shape[0] > args.num_points:
            n_surf = int(is_airfoil.sum())
            n_bulk_keep = max(args.num_points - n_surf, 0)
            bulk_idx = np.where(~is_airfoil)[0]
            if n_bulk_keep > 0 and n_bulk_keep < len(bulk_idx):
                sub_bulk = rng.choice(bulk_idx, size=n_bulk_keep, replace=False)
            else:
                sub_bulk = bulk_idx
            keep = np.concatenate([np.where(is_airfoil)[0], sub_bulk])
            keep = np.sort(keep)
            input_t = input_t[keep]
            is_airfoil = is_airfoil[keep]
            normals_phys = normals_phys[keep]

        geom_latent = test_ds.geom_latents[gi]

        for prior_name, prior_fn in PRIORS.items():
            qoi_per_pass, _, _ = propagate_one_geom(
                model, input_t, is_airfoil, normals_phys, geom_latent,
                coef_norm, prior_fn, args.n_draws, args.T, device, rng,
                out_mean, out_std)
            decomp = decompose_variance(qoi_per_pass)
            # quantiles over per-draw means (what the end user sees)
            per_draw_mean = decomp["per_draw_mean"]
            p05 = np.percentile(per_draw_mean, 5, axis=0)
            p16 = np.percentile(per_draw_mean, 16, axis=0)
            p84 = np.percentile(per_draw_mean, 84, axis=0)
            p95 = np.percentile(per_draw_mean, 95, axis=0)
            results[prior_name].append({
                "name": name,
                "total_mean": decomp["total_mean"],
                "aleatoric_var": decomp["aleatoric_var"],
                "epistemic_var": decomp["epistemic_var"],
                "total_var": decomp["total_var"],
                "p05": p05, "p16": p16, "p84": p84, "p95": p95,
                "per_draw_mean": per_draw_mean,
            })
        elapsed = time.time() - t0
        done_now = (gi - start_gi) + 1
        rate = done_now / max(elapsed, 1e-9)
        eta = (args.n_geom - gi - 1) / max(rate, 1e-9)
        print(f"  geom {gi+1:>2}/{args.n_geom}  n_pts={input_t.shape[0]:,}  "
              f"{name[:30]:<30}  rate={rate:.2f} geom/s  eta={eta:.0f}s")
        # incremental save after each geometry
        torch.save({
            "priors": {
                "A_sensor": "U~N(50,2) m/s, AoA~N(4,0.5) deg",
                "B_envelope": "U~U(40,60) m/s, AoA~U(2,8) deg",
            },
            "n_geom_planned": args.n_geom,
            "n_geom_done": gi + 1,
            "n_draws": args.n_draws,
            "T": args.T,
            "num_points": args.num_points,
            "qois": QOIS,
            "geom_names": geom_names,
            "results": results,
            "seed": args.seed,
            "run_dir": str(run_dir),
        }, partial_path)

    # ---- Summary table ----
    print(f"\n=== propagation summary "
          f"(n_geom={args.n_geom}, n_draws={args.n_draws}, T={args.T}) ===")
    for prior_name in PRIORS:
        print(f"\n-- prior {prior_name} --")
        print(f"{'QoI':<14} {'mean(mean)':>12} {'tot_std':>10} "
              f"{'CV%':>7} {'ale%':>6} {'epi%':>6} "
              f"{'[p16,p84] avg':>24} {'[p05,p95] avg':>24}")
        for k, q in enumerate(QOIS):
            per_geom = results[prior_name]
            m_mean = np.mean([r["total_mean"][k] for r in per_geom])
            m_tot_var = np.mean([r["total_var"][k] for r in per_geom])
            m_ale_var = np.mean([r["aleatoric_var"][k] for r in per_geom])
            m_epi_var = np.mean([r["epistemic_var"][k] for r in per_geom])
            m_tot_std = float(np.sqrt(m_tot_var))
            cv = 100 * m_tot_std / (abs(m_mean) + 1e-12)
            ale_pct = 100 * m_ale_var / max(m_tot_var, 1e-12)
            epi_pct = 100 * m_epi_var / max(m_tot_var, 1e-12)
            p16_avg = np.mean([r["p16"][k] for r in per_geom])
            p84_avg = np.mean([r["p84"][k] for r in per_geom])
            p05_avg = np.mean([r["p05"][k] for r in per_geom])
            p95_avg = np.mean([r["p95"][k] for r in per_geom])
            print(f"{q:<14} {m_mean:>12.3e} {m_tot_std:>10.3e} "
                  f"{cv:>6.2f}% {ale_pct:>5.1f}% {epi_pct:>5.1f}% "
                  f"[{p16_avg:>8.2e},{p84_avg:>8.2e}] "
                  f"[{p05_avg:>8.2e},{p95_avg:>8.2e}]")
    print("\n(ale% = aleatoric (inflow) share of total variance;")
    print(" epi% = epistemic (model) share. Together they sum to 100%.)")

    # ---- Plot: aleatoric vs epistemic shares per QoI per prior ----
    fig, axes = plt.subplots(1, len(QOIS), figsize=(3 * len(QOIS), 4))
    for k, q in enumerate(QOIS):
        ax = axes[k] if len(QOIS) > 1 else axes
        labels, ale, epi = [], [], []
        for prior_name in PRIORS:
            per_geom = results[prior_name]
            tot_var = np.mean([r["total_var"][k] for r in per_geom])
            ale_var = np.mean([r["aleatoric_var"][k] for r in per_geom])
            epi_var = np.mean([r["epistemic_var"][k] for r in per_geom])
            labels.append(prior_name)
            ale.append(100 * ale_var / max(tot_var, 1e-12))
            epi.append(100 * epi_var / max(tot_var, 1e-12))
        x = np.arange(len(labels))
        ax.bar(x, ale, color='tab:blue', label='aleatoric (inflow)')
        ax.bar(x, epi, bottom=ale, color='tab:orange',
               label='epistemic (model)')
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha='right')
        ax.set_ylim(0, 105)
        ax.set_ylabel("% of total variance")
        ax.set_title(q)
        if k == 0:
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Stage 5 — variance decomposition (averaged over geometries)")
    fig.tight_layout()

    ts = tag
    fig_path = out_dir / f"propagation_decomp_{ts}.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"\ndecomposition plot: {fig_path}")

    # ---- Comparison histograms: priors overlay (first geom) ----
    fig, axes = plt.subplots(1, len(QOIS), figsize=(3 * len(QOIS), 4))
    for k, q in enumerate(QOIS):
        ax = axes[k] if len(QOIS) > 1 else axes
        for prior_name, color in [("A_sensor", "tab:blue"),
                                  ("B_envelope", "tab:orange")]:
            arr = results[prior_name][0]["per_draw_mean"][:, k]
            ax.hist(arr, bins=40, alpha=0.5, color=color,
                    label=f"{prior_name} (std={arr.std():.3g})")
        ax.set_xlabel(q); ax.set_ylabel("count")
        ax.legend(fontsize=8); ax.set_title(q)
    fig.suptitle(f"Stage 5 — per-draw QoI distributions (geom 0: "
                 f"{geom_names[0][:30]})")
    fig.tight_layout()
    fig_path2 = out_dir / f"propagation_hist_{ts}.png"
    fig.savefig(fig_path2, dpi=120)
    plt.close(fig)
    print(f"histogram plot:     {fig_path2}")

    payload = {
        "priors": {
            "A_sensor": "U~N(50,2) m/s, AoA~N(4,0.5) deg",
            "B_envelope": "U~U(40,60) m/s, AoA~U(2,8) deg",
        },
        "n_geom": args.n_geom,
        "n_draws": args.n_draws,
        "T": args.T,
        "qois": QOIS,
        "geom_names": geom_names,
        "results": results,
        "seed": args.seed,
        "run_dir": str(run_dir),
    }
    out_path = out_dir / f"propagation_{ts}.pt"
    torch.save(payload, out_path)
    print(f"propagation data:   {out_path}")


if __name__ == "__main__":
    main()
