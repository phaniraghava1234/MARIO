"""Stage 4b — split conformal prediction on top of MC dropout.

Distribution-free intervals with guaranteed marginal coverage.
Where scalar calibration (stage 4) assumes errors are Gaussian around
the predicted mean/std, conformal uses the empirical distribution of
normalized residuals |y - mean| / std on a held-out calibration set.

Standard split conformal for regression with scaled scores:
  1. On val: compute s_i = |y_i - mean_i| / std_i for each point.
  2. Pick miscoverage level α (α=0.32 for 68%, α=0.05 for 95%).
  3. Take q = ⌈(n+1)(1-α)⌉-th order statistic of {s_i}.
  4. For test point j: interval = [mean_j - q * std_j, mean_j + q * std_j].
  5. Marginal coverage ≥ 1 - α by exchangeability (distribution-free).

References:
  Vovk, Gammerman & Shafer, "Algorithmic Learning in a Random World,"
    Springer 2005 (foundational).
  Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction
    and Distribution-Free Uncertainty Quantification," FnTML 2023.
  Papadopoulos et al., "Inductive confidence machines for regression,"
    ECML 2002 (split conformal for regression).

Reuses pool arrays from a calibrate.py output (so no extra MC passes).

Run from repo root:
    python airfrans_task/uq/conformal.py \
        --calibration-pt airfrans_task/uq/trainings/uq_YYYYMMDD-HHMMSS/results/calibration_T50_YYYYMMDD-HHMMSS.pt
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm

FIELDS = ["Velocity-x", "Velocity-y", "Pressure", "Turbulent-viscosity"]


def conformal_q(nonconf, alpha):
    """Split-conformal quantile with the standard (n+1)/(1-α) correction.

    Args:
        nonconf : (n,) nonconformity scores on calibration set
        alpha   : miscoverage (e.g. 0.32 for 68% target, 0.05 for 95%)
    """
    n = len(nonconf)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    return float(np.sort(nonconf)[k - 1])


def coverage(err, half_width):
    """Fraction of points where |err| ≤ half_width."""
    return float(np.mean(np.abs(err) <= half_width))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-pt", required=True,
                    help="output .pt from calibrate.py (has val+test pool arrays)")
    ap.add_argument("--out-dir", default=None,
                    help="where to save results (defaults to run_dir/results/)")
    args = ap.parse_args()

    cal_path = Path(args.calibration_pt).resolve()
    print(f"loading {cal_path}")
    cal = torch.load(cal_path, map_location="cpu", weights_only=False)

    v_err = cal["val"]["pool_err"]              # (N_val, 4)
    v_std = cal["val"]["pool_std"]
    t_err = cal["test_pre"]["pool_err"]         # (N_test, 4)
    t_std = cal["test_pre"]["pool_std"]
    t_std_cal = cal["test_post"]["pool_std_cal"]
    s_scalar = cal["s_per_field"]

    print(f"val pool:  {v_err.shape[0]:,} points")
    print(f"test pool: {t_err.shape[0]:,} points")

    # alpha values for 68% and 95% intervals
    alphas = {"68%": 0.32, "95%": 0.05}

    print("\n=== conformal quantiles fitted on VAL ===")
    print(f"{'field':<22} {'q_68':>8} {'q_95':>8} {'scalar s':>10}")
    q = {level: np.zeros(len(FIELDS)) for level in alphas}
    for k, f in enumerate(FIELDS):
        nonconf = np.abs(v_err[:, k]) / np.maximum(v_std[:, k], 1e-8)
        for level, a in alphas.items():
            q[level][k] = conformal_q(nonconf, a)
        print(f"{f:<22} {q['68%'][k]:>8.3f} {q['95%'][k]:>8.3f} "
              f"{s_scalar[k]:>10.3f}")
    print("(for reference: Gaussian would use 1.000 / 1.960 at 68% / 95%)")

    # test-time: three sets of intervals -- raw, scalar-calibrated, conformal
    print("\n=== TEST coverage: raw vs scalar vs conformal ===")
    print(f"{'field':<22}  {'target':>8} {'raw':>7} {'scalar':>7} {'conf':>7} "
          f"{'raw w':>10} {'scalar w':>10} {'conf w':>10}")

    rows = []
    for k, f in enumerate(FIELDS):
        for level, a in alphas.items():
            target = 1 - a
            # raw Gaussian interval: ±z * std (z=1 for 68%, z=1.96 for 95%)
            z = norm.ppf(0.5 + target / 2)
            hw_raw = z * t_std[:, k]
            hw_scalar = z * t_std_cal[:, k]
            hw_conf = q[level][k] * t_std[:, k]
            cov_raw = coverage(t_err[:, k], hw_raw)
            cov_scalar = coverage(t_err[:, k], hw_scalar)
            cov_conf = coverage(t_err[:, k], hw_conf)
            print(f"{f:<22}  {target:>8.2f} {cov_raw:>7.3f} {cov_scalar:>7.3f} "
                  f"{cov_conf:>7.3f} {hw_raw.mean():>10.3f} "
                  f"{hw_scalar.mean():>10.3f} {hw_conf.mean():>10.3f}")
            rows.append({
                "field": f, "level": level, "target": target,
                "cov_raw": cov_raw, "cov_scalar": cov_scalar,
                "cov_conformal": cov_conf,
                "mean_half_width_raw": float(hw_raw.mean()),
                "mean_half_width_scalar": float(hw_scalar.mean()),
                "mean_half_width_conformal": float(hw_conf.mean()),
                "q_conformal": float(q[level][k]),
                "s_scalar": float(s_scalar[k]),
                "z_gaussian": float(z),
            })
    print("(w = mean half-width of test intervals; smaller is better "
          "when coverage matches)")

    # reliability plot: compare the three methods per field
    probs = np.linspace(0.05, 0.95, 20)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for k, f in enumerate(FIELDS):
        ax = axes[k // 2, k % 2]
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='ideal')
        for method, color, use_std in [
            ("raw (s=1)", "tab:red", t_std[:, k]),
            (f"scalar (s={s_scalar[k]:.2f})", "tab:blue", t_std_cal[:, k]),
            ("conformal", "tab:green", None),
        ]:
            obs = []
            for p in probs:
                a = 1 - p
                if method == "conformal":
                    # use conformal quantile at this level
                    nonconf = np.abs(v_err[:, k]) / np.maximum(v_std[:, k], 1e-8)
                    qp = conformal_q(nonconf, a)
                    hw = qp * t_std[:, k]
                else:
                    z = norm.ppf(0.5 + p / 2)
                    hw = z * use_std
                obs.append((np.abs(t_err[:, k]) <= hw).mean())
            ax.plot(probs, obs, 'o-', color=color, label=method, markersize=4)
        ax.set_xlabel("expected"); ax.set_ylabel("observed")
        ax.set_title(f); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.suptitle("Reliability — raw vs scalar vs conformal")
    fig.tight_layout()

    out_dir = Path(args.out_dir) if args.out_dir else cal_path.parent
    out_dir.mkdir(exist_ok=True, parents=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    fig_path = out_dir / f"conformal_reliability_{ts}.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"\nreliability plot: {fig_path}")

    payload = {
        "fields": FIELDS,
        "q_per_field": {level: q[level].tolist() for level in alphas},
        "s_scalar_per_field": s_scalar.tolist()
                              if hasattr(s_scalar, "tolist") else list(s_scalar),
        "rows": rows,
        "calibration_source": str(cal_path),
    }
    out_path = out_dir / f"conformal_{ts}.pt"
    torch.save(payload, out_path)
    print(f"conformal data: {out_path}")


if __name__ == "__main__":
    main()
