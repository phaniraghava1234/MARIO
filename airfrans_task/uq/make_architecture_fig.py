"""Generate architecture diagrams for the MARIO flow model.

Two variants:
    deterministic : original MARIO (no dropout)
    uq            : our MC-dropout variant (dropout in hypernetwork + trunk)

Usage:
    python airfrans_task/uq/make_architecture_fig.py --variant deterministic
    python airfrans_task/uq/make_architecture_fig.py --variant uq
    python airfrans_task/uq/make_architecture_fig.py --both

Outputs land in airfrans_task/uq/figures/.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


def _box(ax, xy, w, h, text, color="#dceeff", edgecolor="#2b6cb0",
         fontsize=9, bold=False):
    patch = FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.08",
                           linewidth=1.4, facecolor=color,
                           edgecolor=edgecolor)
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(xy[0], xy[1], text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight)


def _arrow(ax, src, dst, color="#555", lw=1.0, style="-|>"):
    arr = FancyArrowPatch(src, dst, arrowstyle=style, mutation_scale=12,
                          linewidth=lw, color=color,
                          shrinkA=4, shrinkB=4)
    ax.add_patch(arr)


def draw(variant="deterministic"):
    """variant in {'deterministic', 'uq'}."""
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.set_axis_off()

    is_uq = variant == "uq"
    dropout_color = "#fed7aa"       # orange for dropout
    dropout_edge = "#dd6b20"

    # ---- Inputs (left column) ----
    _box(ax, (1.0, 7.0), 2.0, 0.8,
         "Point input\n[x, y, sdf, n_x, n_y, bl]\n(6D)",
         color="#e9d8fd", edgecolor="#6b46c1")
    _box(ax, (1.0, 2.5), 2.0, 0.8,
         "Condition\n[geom_latent(8), u_in, v_in]\n(10D)",
         color="#fed7d7", edgecolor="#c53030")

    # ---- Fourier embedding branches (multi-scale) ----
    _box(ax, (4.0, 7.6), 2.0, 0.6,
         "Gaussian Fourier\nscale = 0.5", color="#dceeff", edgecolor="#2b6cb0")
    _box(ax, (4.0, 6.4), 2.0, 0.6,
         "Gaussian Fourier\nscale = 1.0", color="#dceeff", edgecolor="#2b6cb0")
    _arrow(ax, (1.0 + 1.0, 7.0 + 0.2), (4.0 - 1.0, 7.6))
    _arrow(ax, (1.0 + 1.0, 7.0 - 0.2), (4.0 - 1.0, 6.4))

    # ---- Hypernetwork ----
    hnet_top = 2.8
    hnet_bottom = 1.8
    hnet_x = 4.0
    _box(ax, (hnet_x, 2.3), 2.4, 1.1,
         "Hypernetwork\n4× Linear + SiLU",
         color="#fff5f5", edgecolor="#c53030")
    _arrow(ax, (1.0 + 1.0, 2.5), (hnet_x - 1.2, 2.3))

    # dropout markers inside hypernet
    if is_uq:
        for xi in (3.3, 4.0, 4.7):
            _box(ax, (xi, 2.0), 0.5, 0.25, "drop",
                 color=dropout_color, edgecolor=dropout_edge, fontsize=7)

    # ---- Modulations output of hypernet ----
    _box(ax, (7.2, 2.3), 1.6, 0.6,
         "modulations\n(width × (depth-1))",
         color="#fef5e7", edgecolor="#b7791f")
    _arrow(ax, (hnet_x + 1.2, 2.3), (7.2 - 0.8, 2.3))

    # ---- FiLM-modulated trunk (two scales share depth-6 trunk) ----
    trunk_y_top = 7.6
    trunk_y_bot = 6.4
    layer_w = 1.0
    layer_h = 0.55
    gap_x = 1.1
    x_start = 5.4
    layer_boxes_top = []
    layer_boxes_bot = []
    for d in range(6):
        xc = x_start + d * gap_x
        label = f"L{d+1}"
        _box(ax, (xc, trunk_y_top), layer_w, layer_h,
             label + "\n+FiLM" if d < 5 else label,
             color="#dceeff", edgecolor="#2b6cb0")
        _box(ax, (xc, trunk_y_bot), layer_w, layer_h,
             label + "\n+FiLM" if d < 5 else label,
             color="#dceeff", edgecolor="#2b6cb0")
        layer_boxes_top.append(xc)
        layer_boxes_bot.append(xc)
        if is_uq and d < 5:
            # dropout after each modulated layer
            _box(ax, (xc + gap_x / 2, trunk_y_top), 0.5, 0.3,
                 "drop", color=dropout_color, edgecolor=dropout_edge,
                 fontsize=7)
            _box(ax, (xc + gap_x / 2, trunk_y_bot), 0.5, 0.3,
                 "drop", color=dropout_color, edgecolor=dropout_edge,
                 fontsize=7)

    # modulation feeds trunk layers (vertical lines going up/down from
    # modulations box to each FiLM point)
    for xc in layer_boxes_top[:-1]:
        _arrow(ax, (xc, 2.3 + 0.3), (xc, trunk_y_bot - layer_h / 2),
               color="#d69e2e", lw=0.8, style="-|>")
    _arrow(ax, (7.2, 2.3 + 0.3), (7.2, trunk_y_bot - layer_h / 2),
           color="#d69e2e", lw=0.8, style="-|>")

    # chain arrows across each trunk row
    for row_y, boxes in [(trunk_y_top, layer_boxes_top),
                         (trunk_y_bot, layer_boxes_bot)]:
        for i in range(len(boxes) - 1):
            x0 = boxes[i] + layer_w / 2
            x1 = boxes[i + 1] - layer_w / 2
            _arrow(ax, (x0, row_y), (x1, row_y))

    # Fourier -> first trunk layer
    _arrow(ax, (4.0 + 1.0, 7.6), (layer_boxes_top[0] - layer_w / 2, 7.6))
    _arrow(ax, (4.0 + 1.0, 6.4), (layer_boxes_bot[0] - layer_w / 2, 6.4))

    # ---- Concatenation + final linear ----
    concat_x = x_start + 5 * gap_x + 1.3
    _box(ax, (concat_x, 7.0), 1.4, 1.1,
         "Concat\nscales\n(2 × width)",
         color="#e6fffa", edgecolor="#2c7a7b")
    _arrow(ax, (layer_boxes_top[-1] + layer_w / 2, 7.6),
           (concat_x - 0.7, 7.2))
    _arrow(ax, (layer_boxes_bot[-1] + layer_w / 2, 6.4),
           (concat_x - 0.7, 6.8))

    final_x = concat_x + 1.6
    _box(ax, (final_x, 7.0), 1.0, 0.55,
         "Linear",
         color="#dceeff", edgecolor="#2b6cb0")
    _arrow(ax, (concat_x + 0.7, 7.0), (final_x - 0.5, 7.0))

    _box(ax, (final_x + 1.5, 7.0), 1.3, 0.9,
         "Output\n[u, v, p, ν_t]\n(4D)",
         color="#e9d8fd", edgecolor="#6b46c1")
    _arrow(ax, (final_x + 0.5, 7.0), (final_x + 1.5 - 0.65, 7.0))

    # ---- Title and legend ----
    title = ("MARIO flow model — deterministic" if not is_uq
             else "MARIO flow model — MC-dropout UQ variant")
    ax.text(6.5, 0.55, title, ha="center", fontsize=14, fontweight="bold")
    ax.text(6.5, 0.1,
            ("One forward pass per query point. Multi-scale Gaussian "
             "Fourier features; hypernetwork produces per-layer FiLM "
             "shift modulations from the condition vector."),
            ha="center", fontsize=9, color="#333")

    # legend
    handles = [
        mpatches.Patch(facecolor="#dceeff", edgecolor="#2b6cb0",
                       label="Trunk layer (Linear + FiLM shift + ReLU)"),
        mpatches.Patch(facecolor="#fff5f5", edgecolor="#c53030",
                       label="Hypernetwork (cond → modulations)"),
        mpatches.Patch(facecolor="#fef5e7", edgecolor="#b7791f",
                       label="FiLM modulations (yellow arrows)"),
    ]
    if is_uq:
        handles.append(
            mpatches.Patch(facecolor=dropout_color, edgecolor=dropout_edge,
                           label="Dropout (active at train AND inference)"))
    ax.legend(handles=handles, loc="lower left", fontsize=8,
              frameon=False, bbox_to_anchor=(0.0, 0.14))

    fig.tight_layout()
    out_path = OUT_DIR / f"architecture_{variant}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def draw_sdf_encoder():
    """Stage 1 — SDF encoder (ModulatedFourierFeatures + CAVIA inner loop)."""
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.set_axis_off()

    # ---- Input coords ----
    _box(ax, (1.0, 6.5), 1.8, 0.8,
         "Point input\n(x, y)\n(2D)",
         color="#e9d8fd", edgecolor="#6b46c1")

    # ---- Gaussian Fourier (single scale) ----
    _box(ax, (4.0, 6.5), 2.0, 0.8,
         "Gaussian Fourier\n32 freqs, scale 0.5",
         color="#dceeff", edgecolor="#2b6cb0")
    _arrow(ax, (1.0 + 0.9, 6.5), (4.0 - 1.0, 6.5))

    # ---- Trunk with FiLM ----
    x_start = 6.4
    gap = 0.95
    layer_w = 0.8
    layer_h = 0.55
    xs_trunk = []
    for d in range(5):
        xc = x_start + d * gap
        _box(ax, (xc, 6.5), layer_w, layer_h,
             f"L{d+1}" + ("\n+FiLM" if d < 4 else ""),
             color="#dceeff", edgecolor="#2b6cb0")
        xs_trunk.append(xc)
    _arrow(ax, (4.0 + 1.0, 6.5), (xs_trunk[0] - layer_w / 2, 6.5))
    for i in range(len(xs_trunk) - 1):
        _arrow(ax, (xs_trunk[i] + layer_w / 2, 6.5),
               (xs_trunk[i + 1] - layer_w / 2, 6.5))

    # ---- Output ----
    out_x = xs_trunk[-1] + 1.0
    _box(ax, (out_x, 6.5), 1.2, 0.8,
         "SDF value\n(1D)",
         color="#e9d8fd", edgecolor="#6b46c1")
    _arrow(ax, (xs_trunk[-1] + layer_w / 2, 6.5), (out_x - 0.6, 6.5))

    # ---- CAVIA inner loop panel ----
    _box(ax, (1.5, 3.5), 2.6, 0.8,
         "Per-shape latent\nz_shape (8D)\ninit = 0 at outer step",
         color="#fed7d7", edgecolor="#c53030", fontsize=9)
    _box(ax, (5.3, 3.5), 2.2, 0.8,
         "Hypernetwork\n1× Linear + SiLU",
         color="#fff5f5", edgecolor="#c53030")
    _box(ax, (8.3, 3.5), 1.8, 0.8,
         "modulations\n(width × 4)",
         color="#fef5e7", edgecolor="#b7791f")
    _arrow(ax, (1.5 + 1.3, 3.5), (5.3 - 1.1, 3.5))
    _arrow(ax, (5.3 + 1.1, 3.5), (8.3 - 0.9, 3.5))
    # Modulations fed up into trunk layers
    for xc in xs_trunk[:-1]:
        _arrow(ax, (8.3, 3.5 + 0.4), (xc, 6.5 - layer_h / 2),
               color="#d69e2e", lw=0.8)

    # ---- CAVIA inner loop annotation ----
    _box(ax, (6.5, 1.4), 7.5, 1.2,
         ("CAVIA inner loop (3 steps) during training:\n"
          "   fit z_shape with inner lr α   —   outer loop updates INR weights + α"),
         color="#fff5f5", edgecolor="#c53030", fontsize=10)
    _arrow(ax, (6.5, 1.4 + 0.6), (3.0, 3.5 - 0.4),
           color="#999", lw=0.8, style="->")

    ax.text(6.5, 0.5,
            "MARIO — Stage 1 — SDF Encoder (ModulatedFourierFeatures)",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(6.5, 0.1,
            "Deterministic. 500 training epochs. Produces one 8-dim latent per airfoil shape.",
            ha="center", fontsize=9, color="#333")

    handles = [
        mpatches.Patch(facecolor="#dceeff", edgecolor="#2b6cb0",
                       label="Trunk layer (Linear + FiLM + ReLU)"),
        mpatches.Patch(facecolor="#fff5f5", edgecolor="#c53030",
                       label="Hypernetwork / CAVIA machinery"),
        mpatches.Patch(facecolor="#fef5e7", edgecolor="#b7791f",
                       label="FiLM modulations (yellow arrows)"),
        mpatches.Patch(facecolor="#fed7d7", edgecolor="#c53030",
                       label="Per-shape latent (what we extract)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              frameon=False, bbox_to_anchor=(1.0, 0.0))

    fig.tight_layout()
    out_path = OUT_DIR / "architecture_sdf_encoder.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def draw_pipeline():
    """End-to-end 6-stage pipeline overview."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.set_axis_off()

    # Row 1 — training lane (stages 1-2)
    _box(ax, (1.3, 7.5), 2.0, 1.0,
         "AirfRANS\n1000 CFD sims\n(scarce: 200+200)",
         color="#e9d8fd", edgecolor="#6b46c1", fontsize=9)
    _box(ax, (4.6, 7.5), 2.0, 1.0,
         "STAGE 1\nSDF Encoder\n500 ep", color="#dceeff",
         edgecolor="#2b6cb0", bold=True)
    _box(ax, (7.9, 7.5), 2.0, 1.0,
         "geom_latents\n.npz\n(8D × 200 shapes)",
         color="#f0f0f0", edgecolor="#555", fontsize=9)
    _box(ax, (11.2, 7.5), 2.0, 1.0,
         "STAGE 2\nFlow Model\n+ MC Dropout\n500 ep",
         color="#dceeff", edgecolor="#2b6cb0", bold=True)
    _arrow(ax, (1.3 + 1.0, 7.5), (4.6 - 1.0, 7.5))
    _arrow(ax, (4.6 + 1.0, 7.5), (7.9 - 1.0, 7.5))
    _arrow(ax, (7.9 + 1.0, 7.5), (11.2 - 1.0, 7.5))
    # geom_latents also go down to inference
    ax.text(1.3, 6.6, "— training (one-time) —", fontsize=9,
            color="#6b46c1", style="italic")

    # Mid: checkpoint from stage 2
    _box(ax, (11.2, 5.5), 2.0, 0.8,
         "best.pt\n(epoch 142)",
         color="#f0f0f0", edgecolor="#555", fontsize=9)
    _arrow(ax, (11.2, 7.0), (11.2, 5.9))

    # Row 2 — UQ branches
    ax.text(1.3, 4.6, "— inference (minutes) —", fontsize=9,
            color="#2c7a7b", style="italic")

    # Stage 3
    _box(ax, (2.0, 3.5), 2.0, 1.0,
         "STAGE 3\nMC Dropout Eval\nT=50", color="#e6fffa",
         edgecolor="#2c7a7b", bold=True)
    _arrow(ax, (11.2 - 0.9, 5.5), (2.0 + 1.0, 3.5 + 0.3),
           color="#777")
    _box(ax, (2.0, 1.6), 2.0, 0.8,
         "per-point\n(mean, std)\nr(|err|,std)≈0.6",
         color="#f0f0f0", edgecolor="#555", fontsize=8)
    _arrow(ax, (2.0, 3.0), (2.0, 2.0))

    # Stage 4 scalar
    _box(ax, (5.0, 3.5), 2.0, 1.0,
         "STAGE 4\nScalar Calibration\n(Gaussian NLL)",
         color="#e6fffa", edgecolor="#2c7a7b", bold=True)
    _arrow(ax, (2.0 + 1.0, 3.5), (5.0 - 1.0, 3.5))
    # Stage 4b conformal
    _box(ax, (7.5, 3.5), 2.0, 1.0,
         "STAGE 4b\nConformal\n(distribution-free)",
         color="#e6fffa", edgecolor="#2c7a7b", bold=True)
    _arrow(ax, (5.0 + 1.0, 3.5), (7.5 - 1.0, 3.5))
    _box(ax, (6.25, 1.6), 2.8, 0.8,
         "calibrated intervals\ncov68=0.67, cov95=0.95",
         color="#f0f0f0", edgecolor="#555", fontsize=8)
    _arrow(ax, (6.25, 3.0), (6.25, 2.0))

    # Stage 5 propagation
    _box(ax, (10.5, 3.5), 1.6, 1.0,
         "STAGE 5\nPropagation\n(Priors A,B)",
         color="#e6fffa", edgecolor="#2c7a7b", bold=True)
    _arrow(ax, (11.2 - 0.3, 5.5), (10.5 + 0.5, 3.5 + 0.3),
           color="#777")
    _box(ax, (10.5, 1.6), 1.6, 0.8,
         "QoI distributions\naleatoric / epistemic",
         color="#f0f0f0", edgecolor="#555", fontsize=8)
    _arrow(ax, (10.5, 3.0), (10.5, 2.0))

    # Stage 6 sensitivity
    _box(ax, (12.8, 3.5), 1.2, 1.0,
         "STAGE 6\nSobol\n(SALib)",
         color="#e6fffa", edgecolor="#2c7a7b", bold=True)
    _arrow(ax, (11.2 + 0.5, 5.5), (12.8 - 0.4, 3.5 + 0.3),
           color="#777")
    _box(ax, (12.8, 1.6), 1.2, 0.8,
         "S1[AoA]≈1.0\n(for Cl)",
         color="#f0f0f0", edgecolor="#555", fontsize=8)
    _arrow(ax, (12.8, 3.0), (12.8, 2.0))

    ax.text(7.0, 0.6,
            "MARIO + Monte Carlo UQ — end-to-end pipeline (6 stages)",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(7.0, 0.2,
            "Training (top) is run once; inference stages (bottom) all reuse the "
            "same best.pt checkpoint.",
            ha="center", fontsize=9, color="#333")

    handles = [
        mpatches.Patch(facecolor="#dceeff", edgecolor="#2b6cb0",
                       label="Training stage"),
        mpatches.Patch(facecolor="#e6fffa", edgecolor="#2c7a7b",
                       label="Inference / UQ stage"),
        mpatches.Patch(facecolor="#f0f0f0", edgecolor="#555",
                       label="Artifact (file or result)"),
        mpatches.Patch(facecolor="#e9d8fd", edgecolor="#6b46c1",
                       label="External data"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8,
              frameon=False, bbox_to_anchor=(0.0, 0.0))

    fig.tight_layout()
    out_path = OUT_DIR / "pipeline_overview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def draw_mc_mechanism():
    """MC dropout at inference — T stochastic forward passes to mean + std."""
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.set_axis_off()

    # Input
    _box(ax, (1.4, 4.5), 2.0, 0.9,
         "Query input\n(point coords +\ngeom latent + inflow)",
         color="#e9d8fd", edgecolor="#6b46c1", fontsize=9)

    # Model instances (T stochastic masks)
    model_x = 5.5
    T_shown = 4
    model_ys = [6.5, 5.5, 4.5, 3.5]
    for i, y in enumerate(model_ys):
        _box(ax, (model_x, y), 2.2, 0.6,
             f"Pass t={i+1}: model.train() (dropout ON)",
             color="#dceeff", edgecolor="#2b6cb0", fontsize=8)
        # dropout mask icon
        _box(ax, (model_x - 1.4, y), 0.4, 0.3, "mask",
             color="#fed7aa", edgecolor="#dd6b20", fontsize=6)
        _arrow(ax, (1.4 + 1.0, 4.5), (model_x - 1.6, y), color="#aaa")
    # dots for T=50
    ax.text(model_x, 2.6, "⋮", fontsize=18, ha="center")
    ax.text(model_x, 2.0, "(total T = 50 passes)", fontsize=8,
            ha="center", color="#555")

    # Per-pass outputs
    for i, y in enumerate(model_ys):
        _box(ax, (8.5, y), 1.4, 0.5,
             f"y_t ∈ ℝ^4",
             color="#f0f0f0", edgecolor="#555", fontsize=8)
        _arrow(ax, (model_x + 1.1, y), (8.5 - 0.7, y), color="#aaa")

    # Aggregation
    _box(ax, (11.0, 5.5), 1.6, 0.9,
         "mean\nover T\n= point estimate",
         color="#e6fffa", edgecolor="#2c7a7b", fontsize=9)
    _box(ax, (11.0, 3.5), 1.6, 0.9,
         "std\nover T\n= uncertainty",
         color="#e6fffa", edgecolor="#2c7a7b", fontsize=9)
    for y in model_ys:
        _arrow(ax, (8.5 + 0.7, y), (11.0 - 0.8, 5.5),
               color="#aaa", lw=0.6)
        _arrow(ax, (8.5 + 0.7, y), (11.0 - 0.8, 3.5),
               color="#aaa", lw=0.6)

    # Downstream: calibration applied to std
    _box(ax, (11.0, 1.8), 1.6, 0.8,
         "std_cal\n= s · std\nor q · std",
         color="#fef5e7", edgecolor="#b7791f", fontsize=9)
    _arrow(ax, (11.0, 3.5 - 0.45), (11.0, 1.8 + 0.4))
    ax.text(11.0 + 0.9, 1.8, "stage 4 / 4b", fontsize=7, color="#b7791f")

    # Title
    ax.text(6.5, 0.7,
            "MC Dropout at inference — T stochastic forward passes",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(6.5, 0.3,
            "Weights are fixed. Only the dropout mask changes per pass. "
            "Reference: Gal & Ghahramani, ICML 2016.",
            ha="center", fontsize=9, color="#333")

    handles = [
        mpatches.Patch(facecolor="#dceeff", edgecolor="#2b6cb0",
                       label="Forward pass of trained model (dropout ON)"),
        mpatches.Patch(facecolor="#fed7aa", edgecolor="#dd6b20",
                       label="Random dropout mask (varies per pass)"),
        mpatches.Patch(facecolor="#f0f0f0", edgecolor="#555",
                       label="Per-pass output y_t"),
        mpatches.Patch(facecolor="#e6fffa", edgecolor="#2c7a7b",
                       label="Aggregated prediction + uncertainty"),
        mpatches.Patch(facecolor="#fef5e7", edgecolor="#b7791f",
                       label="Calibrated std for downstream UQ"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8,
              frameon=False, bbox_to_anchor=(0.0, 1.0))

    fig.tight_layout()
    out_path = OUT_DIR / "mc_dropout_mechanism.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def _naca0012(x):
    """NACA 0012 half-thickness as function of chord x in [0, 1]."""
    import numpy as np
    return 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                  + 0.2843 * x ** 3 - 0.1015 * x ** 4)


def _sdf_field(nx=180, ny=120, aoa_deg=4.0):
    """Approximate SDF field around a NACA 0012 at AoA for the figure."""
    import numpy as np
    x = np.linspace(-0.4, 1.4, nx)
    y = np.linspace(-0.55, 0.55, ny)
    X, Y = np.meshgrid(x, y)
    # rotate by -AoA so the airfoil visually tilts up
    a = np.deg2rad(aoa_deg)
    Xr = X * np.cos(a) + Y * np.sin(a)
    Yr = -X * np.sin(a) + Y * np.cos(a)
    # distance to nearest surface point along chord (approximation)
    xs = np.linspace(0.0, 1.0, 400)
    yt = _naca0012(xs)
    dist = np.full_like(X, 1e9)
    for xc, yc in zip(xs, yt):
        d_up = np.sqrt((Xr - xc) ** 2 + (Yr - yc) ** 2)
        d_dn = np.sqrt((Xr - xc) ** 2 + (Yr + yc) ** 2)
        dist = np.minimum(dist, np.minimum(d_up, d_dn))
    # inside airfoil
    inside = (Xr >= 0) & (Xr <= 1) & (np.abs(Yr) <= _naca0012(np.clip(Xr, 0, 1)))
    dist[inside] = 0.0
    return X, Y, dist


def _pressure_field(X, Y, aoa_deg=4.0):
    """Toy pressure field around the airfoil for visualization only."""
    import numpy as np
    a = np.deg2rad(aoa_deg)
    Xr = X * np.cos(a) + Y * np.sin(a)
    Yr = -X * np.sin(a) + Y * np.cos(a)
    # suction above (Yr>0 near chord), overpressure below
    r = np.sqrt((Xr - 0.3) ** 2 + Yr ** 2 + 0.05)
    p = -np.sign(Yr) * np.exp(-6 * r) * (Xr > -0.2) * (Xr < 1.2)
    inside = (Xr >= 0) & (Xr <= 1) & (np.abs(Yr) <= _naca0012(np.clip(Xr, 0, 1)))
    p[inside] = np.nan
    return p


def _draw_mlp_neurons(ax, x0, y0, layer_sizes, dx=0.6, dy=0.2,
                      neuron_r=0.07, color="#b8c6ff", edge="#4a5fb8",
                      arrow_color="#555"):
    """Draw a mini-MLP of neurons with full connections between layers."""
    import numpy as np
    L = len(layer_sizes)
    positions = []
    for li, size in enumerate(layer_sizes):
        xs = x0 + li * dx
        ys_center = y0
        total_h = (size - 1) * dy
        ys = np.linspace(ys_center + total_h / 2, ys_center - total_h / 2, size)
        pts = [(xs, yy) for yy in ys]
        positions.append(pts)
    # connections
    for li in range(L - 1):
        for p0 in positions[li]:
            for p1 in positions[li + 1]:
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                        color=arrow_color, lw=0.25, alpha=0.55, zorder=1)
    # neurons
    for pts in positions:
        for px, py in pts:
            circ = plt.Circle((px, py), neuron_r,
                              facecolor=color, edgecolor=edge,
                              linewidth=0.8, zorder=2)
            ax.add_patch(circ)
    return positions


def draw_paper_style(variant="uq"):
    """Paper-style schematic for our AirfRANS problem (deterministic or UQ).

    Reproduces the look of the MARIO paper figure (SDF viz + encoder +
    hypernetwork drawn as neurons + modulated trunk drawn as neurons +
    shift modulation arrows + output field) but tailored to our exact
    inputs/outputs, with dropout markers added in the UQ variant.
    """
    import numpy as np

    is_uq = variant == "uq"
    fig, ax = plt.subplots(figsize=(14, 9.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_axis_off()

    dropout_color = "#fed7aa"
    dropout_edge = "#dd6b20"
    modulation_color = "#ec6a9c"
    arrow_grey = "#555"

    # Top title band (paper-figure style)
    title = ("MARIO on AirfRANS — Stage 2 flow model (deterministic)"
             if not is_uq
             else "MARIO on AirfRANS — Stage 2 flow model with MC-dropout UQ")
    ax.text(7.0, 9.65, title, ha="center", fontsize=14, fontweight="bold")

    # ======== 1. Top-left: SDF field around NACA airfoil ========
    sdf_ax = fig.add_axes([0.045, 0.55, 0.22, 0.31])
    Xg, Yg, D = _sdf_field()
    sdf_ax.pcolormesh(Xg, Yg, D, cmap="viridis", shading="auto")
    sdf_ax.contour(Xg, Yg, D, levels=[0.0], colors="white", linewidths=1.5)
    sdf_ax.set_xticks([]); sdf_ax.set_yticks([])
    sdf_ax.set_title("SDF around airfoil (stage 1 input)",
                     fontsize=9, pad=4)
    for s in sdf_ax.spines.values():
        s.set_edgecolor("#2b6cb0"); s.set_linewidth(1.6)

    # Encoder E_in
    _box(ax, (4.8, 7.3), 1.4, 1.2, "E_in\n(stage 1)",
         color="#c6f6d5", edgecolor="#2f855a", fontsize=11, bold=True)
    _arrow(ax, (3.65, 7.3), (4.1, 7.3), color=arrow_grey, lw=1.5)

    # μ_geom box
    _box(ax, (6.4, 7.3), 1.1, 0.8, r"$\mu_{geom}$" + "\n(8D)",
         color="#dceeff", edgecolor="#2b6cb0", fontsize=10)
    _arrow(ax, (5.5, 7.3), (5.85, 7.3), color=arrow_grey, lw=1.5)

    # Inflow condition box
    _box(ax, (6.4, 5.9), 1.1, 0.8, r"$\mu_{inlet}$" + "\n[u_in, v_in]",
         color="#fed7d7", edgecolor="#c53030", fontsize=10)

    # z vector
    _box(ax, (7.9, 6.6), 0.9, 1.6, "z\n(10D)\n[μ_geom\nμ_inlet]",
         color="#dceeff", edgecolor="#2b6cb0", fontsize=9)
    _arrow(ax, (6.95, 7.3), (7.45, 6.8), color=arrow_grey, lw=1.5)
    _arrow(ax, (6.95, 5.9), (7.45, 6.4), color=arrow_grey, lw=1.5)

    # ======== 2. Hypernetwork — drawn with neurons ========
    hnet_x0, hnet_y0 = 9.0, 6.6
    _draw_mlp_neurons(ax, hnet_x0, hnet_y0, [3, 4, 4, 3],
                      dx=0.52, dy=0.32, neuron_r=0.08,
                      color="#b8c6ff", edge="#4a5fb8")
    # dropout markers on hypernet layers (UQ variant)
    if is_uq:
        for k in range(3):
            xc = hnet_x0 + (k + 0.5) * 0.52
            _box(ax, (xc, hnet_y0 - 0.95), 0.42, 0.22, "drop",
                 color=dropout_color, edgecolor=dropout_edge, fontsize=6)
    _arrow(ax, (8.4, 6.6), (8.95, 6.6), color=arrow_grey, lw=1.5)

    # φ (shift modulations output)
    phi_x = 11.4
    _box(ax, (phi_x, 6.6), 0.7, 0.6, r"$\phi$",
         color="#fce4ec", edgecolor=modulation_color, fontsize=12, bold=True)
    _arrow(ax, (hnet_x0 + 3 * 0.52 + 0.2, 6.6), (phi_x - 0.4, 6.6),
           color=arrow_grey, lw=1.5)

    # ======== 3. Middle-left: Point input X and γ (Fourier) ========
    _box(ax, (0.9, 3.5), 1.1, 1.6,
         "X =\n[x, y,\n sdf,\n n_x, n_y,\n bl_mask]",
         color="#c6f6d5", edgecolor="#2f855a", fontsize=9)
    _box(ax, (2.6, 3.5), 0.8, 0.9, r"$\gamma$" + "\n(multi-\n scale\n Fourier)",
         color="#dceeff", edgecolor="#2b6cb0", fontsize=8)
    _arrow(ax, (1.45, 3.5), (2.25, 3.5), color=arrow_grey, lw=1.5)

    # ======== 4. Trunk — two scales, each 6 layers of drawn neurons ========
    trunk_x0 = 3.6
    trunk_y_top = 4.7
    trunk_y_bot = 2.3
    layer_sizes = [4, 5, 5, 5, 5, 4]

    pos_top = _draw_mlp_neurons(ax, trunk_x0, trunk_y_top, layer_sizes,
                                dx=0.75, dy=0.26, neuron_r=0.085,
                                color="#b8c6ff", edge="#4a5fb8")
    pos_bot = _draw_mlp_neurons(ax, trunk_x0, trunk_y_bot, layer_sizes,
                                dx=0.75, dy=0.26, neuron_r=0.085,
                                color="#b8c6ff", edge="#4a5fb8")

    # scale labels on the left
    ax.text(trunk_x0 - 0.5, trunk_y_top, "scale 0.5", fontsize=8,
            ha="right", va="center", color="#2b6cb0", style="italic")
    ax.text(trunk_x0 - 0.5, trunk_y_bot, "scale 1.0", fontsize=8,
            ha="right", va="center", color="#2b6cb0", style="italic")

    # γ → trunk arrows (both scales)
    _arrow(ax, (3.0, 3.6), (trunk_x0 - 0.1, trunk_y_top), color=arrow_grey)
    _arrow(ax, (3.0, 3.4), (trunk_x0 - 0.1, trunk_y_bot), color=arrow_grey)

    # Shift modulation arrows from φ down into each trunk layer
    for li in range(len(layer_sizes) - 1):
        xc = trunk_x0 + li * 0.75
        ax.plot([phi_x, xc], [6.3, trunk_y_top + 0.6], color=modulation_color,
                lw=1.1, ls="--", alpha=0.75, zorder=3)
        ax.plot([phi_x, xc], [6.3, trunk_y_bot + 0.6], color=modulation_color,
                lw=1.1, ls="--", alpha=0.75, zorder=3)

    # Dropout markers between trunk layers (UQ variant)
    if is_uq:
        for li in range(len(layer_sizes) - 1):
            xc = trunk_x0 + (li + 0.5) * 0.75
            _box(ax, (xc, trunk_y_top - 0.95), 0.42, 0.22, "drop",
                 color=dropout_color, edgecolor=dropout_edge, fontsize=6)
            _box(ax, (xc, trunk_y_bot - 0.95), 0.42, 0.22, "drop",
                 color=dropout_color, edgecolor=dropout_edge, fontsize=6)

    # Concat and final linear → output
    concat_x = trunk_x0 + 5 * 0.75 + 1.1
    _box(ax, (concat_x, 3.5), 0.9, 1.6,
         "concat\n2× width\n→ Linear",
         color="#e6fffa", edgecolor="#2c7a7b", fontsize=8)
    _arrow(ax, (trunk_x0 + 5 * 0.75 + 0.2, trunk_y_top),
           (concat_x - 0.45, 3.8), color=arrow_grey)
    _arrow(ax, (trunk_x0 + 5 * 0.75 + 0.2, trunk_y_bot),
           (concat_x - 0.45, 3.2), color=arrow_grey)

    # Output
    out_x = concat_x + 1.5
    _box(ax, (out_x, 3.5), 1.3, 1.1,
         "u(x) = \n[u, v,\n p, ν_t]\n(4D)",
         color="#fce4ec", edgecolor="#c53030", fontsize=9)
    _arrow(ax, (concat_x + 0.45, 3.5), (out_x - 0.65, 3.5),
           color=arrow_grey, lw=1.5)

    # ======== 5. Bottom-right: predicted pressure field viz ========
    pred_ax = fig.add_axes([0.76, 0.07, 0.21, 0.24])
    P = _pressure_field(Xg, Yg)
    pred_ax.pcolormesh(Xg, Yg, P, cmap="RdBu_r", shading="auto",
                       vmin=-0.5, vmax=0.5)
    pred_ax.contour(Xg, Yg, P, levels=10, colors="k", linewidths=0.3)
    pred_ax.set_xticks([]); pred_ax.set_yticks([])
    pred_ax.set_title("predicted p field (de-normalized)",
                      fontsize=9, pad=4)
    for s in pred_ax.spines.values():
        s.set_edgecolor("#c53030"); s.set_linewidth(1.6)

    # ======== 6. Labels & annotations ========
    ax.text(10.3, 7.4, "Shift Modulation", fontsize=10,
            color=modulation_color, style="italic")
    ax.text(0.15, 9.3, "Non-parametric geometry — AirfRANS airfoils",
            fontsize=10, color="#2f855a", style="italic")

    sub1 = ("Multi-scale Fourier embedding γ; hypernetwork maps condition z "
            "to per-layer FiLM shift modulations φ.")
    sub2 = ("Dropout stays active at inference → T stochastic passes give "
            "(mean, std) per point."
            if is_uq else
            "Deterministic — one forward pass per query point.")
    ax.text(7.0, 0.75, sub1, ha="center", fontsize=9, color="#333")
    ax.text(7.0, 0.4, sub2, ha="center", fontsize=9, color="#333")

    # Legend
    handles = [
        mpatches.Patch(facecolor="#c6f6d5", edgecolor="#2f855a",
                       label="Inputs / encoder"),
        mpatches.Patch(facecolor="#dceeff", edgecolor="#2b6cb0",
                       label="Latent / Fourier / trunk layers"),
        mpatches.Patch(facecolor="#fed7d7", edgecolor="#c53030",
                       label="Inflow / output"),
        mpatches.Patch(facecolor="#fce4ec", edgecolor=modulation_color,
                       label="Shift modulations φ (pink dashed arrows)"),
    ]
    if is_uq:
        handles.append(
            mpatches.Patch(facecolor=dropout_color, edgecolor=dropout_edge,
                           label="Dropout (active at inference)"))
    # Legend placed in dedicated space below the diagram, horizontal
    ax.legend(handles=handles, loc="lower center", fontsize=8,
              frameon=False, bbox_to_anchor=(0.5, 0.08),
              ncol=5 if is_uq else 4)

    out_path = OUT_DIR / f"paper_style_{variant}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant",
                    choices=["deterministic", "uq", "sdf", "pipeline",
                             "mc", "paper_det", "paper_uq"],
                    default=None)
    ap.add_argument("--both", action="store_true",
                    help="draw just the two flow-model variants")
    ap.add_argument("--all", action="store_true",
                    help="draw every diagram")
    args = ap.parse_args()
    if args.all:
        draw("deterministic")
        draw("uq")
        draw_sdf_encoder()
        draw_pipeline()
        draw_mc_mechanism()
        draw_paper_style("deterministic")
        draw_paper_style("uq")
    elif args.both:
        draw("deterministic")
        draw("uq")
    elif args.variant == "sdf":
        draw_sdf_encoder()
    elif args.variant == "pipeline":
        draw_pipeline()
    elif args.variant == "mc":
        draw_mc_mechanism()
    elif args.variant == "paper_det":
        draw_paper_style("deterministic")
    elif args.variant == "paper_uq":
        draw_paper_style("uq")
    elif args.variant in ("deterministic", "uq"):
        draw(args.variant)
    else:
        # default behaviour: all
        draw("deterministic")
        draw("uq")
        draw_sdf_encoder()
        draw_pipeline()
        draw_mc_mechanism()
        draw_paper_style("deterministic")
        draw_paper_style("uq")


if __name__ == "__main__":
    main()
