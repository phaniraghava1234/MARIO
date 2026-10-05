"""Train MultiScaleModulatedFourierFeaturesUQ on the AirfRANS flow task.

Copy of airfrans_task/train.py adapted for UQ:
  - dropout-enabled model from airfrans_task.uq.models_uq
  - validation split carved OUT of the AirfRANS train set (test set
    untouched during training; upstream validates on test)
  - device-agnostic (CPU / CUDA / anywhere)
  - wandb defaults to offline
  - periodic checkpointing every N epochs for Colab session recovery
  - checkpoint payload includes coef_norm so evaluate_uq can reload
    normalization without touching the raw dataset again

Run from airfrans_task/uq (see config_uq.yaml header for CLI example).
"""
import os
import sys
import time
import pickle
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "airfrans_task"))

import airfrans as af
import wandb
from dataset import AirfransFlowDataset, subsample_dataset
from src.utils_training import training_step
from airfrans_task.uq.models_uq import MultiScaleModulatedFourierFeaturesUQ

# outlier removed by AirfransFlowDataset — mirror here so the split
# reflects true train size, not raw airfrans size
_OUTLIERS = {"airFoil2D_SST_50.077_-4.416_2.834_4.029_1.0_5.156"}


def _remove_outliers(data_list, names, latents):
    keep = [i for i, n in enumerate(names) if n not in _OUTLIERS]
    return ([data_list[i] for i in keep],
            [names[i] for i in keep],
            latents[keep])


def _train_val_split(data_list, names, latents, val_size, seed):
    rng = np.random.RandomState(seed)
    n = len(data_list)
    val_idx = sorted(rng.choice(n, size=val_size, replace=False).tolist())
    train_idx = [i for i in range(n) if i not in val_idx]
    tri = lambda a: [a[i] for i in train_idx] if isinstance(a, list) else a[train_idx]
    vai = lambda a: [a[i] for i in val_idx] if isinstance(a, list) else a[val_idx]
    return ((tri(data_list), tri(names), tri(latents), train_idx),
            (vai(data_list), vai(names), vai(latents), val_idx))


@hydra.main(config_path=".", config_name="config_uq", version_base=None)
def main(cfg: DictConfig) -> None:
    os.environ.setdefault("WANDB_MODE", cfg.wandb.mode)
    os.environ.setdefault("WANDB__SERVICE_WAIT", "300")
    torch.set_default_dtype(torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    wandb.init(project=cfg.wandb.project,
               config=OmegaConf.to_container(cfg, resolve=True))

    # ---- Data ----
    train_raw, train_names = af.dataset.load(
        root=cfg.dataset.root_path, task=cfg.dataset.task, train=True)
    train_latents = np.load(cfg.dataset.train_latents_path)['train_modulations']
    train_raw, train_names, train_latents = _remove_outliers(
        train_raw, train_names, train_latents)

    (tr_data, tr_names, tr_lat, tr_idx), (va_data, va_names, va_lat, va_idx) = \
        _train_val_split(train_raw, train_names, train_latents,
                         cfg.optim.val_split_size, cfg.optim.val_split_seed)

    train_dataset = AirfransFlowDataset(tr_data, tr_names, tr_lat, mode='train')
    coef_norm = train_dataset.coef_norm
    val_dataset = AirfransFlowDataset(va_data, va_names, va_lat, mode='val',
                                      coef_norm=coef_norm)

    ntrain, nval = len(train_dataset), len(val_dataset)
    print(f"train: {ntrain} sims, val: {nval} sims (test held for eval)")

    # ---- Model ----
    model = MultiScaleModulatedFourierFeaturesUQ(
        input_dim=cfg.inr.in_dim,
        output_dim=cfg.inr.out_dim,
        num_frequencies=cfg.inr.num_frequencies,
        latent_dim=cfg.inr.latent_dim,
        width=cfg.inr.hidden_dim,
        depth=cfg.inr.depth,
        depth_hnn=cfg.inr.hypernet_depth,
        include_input=cfg.inr.include_input,
        scales=tuple(cfg.inr.scale),
        scalar_out_dim=cfg.inr.out_scalar_dim,
        dropout_hnn=cfg.inr.dropout_hnn,
        dropout_trunk=cfg.inr.dropout_trunk,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.optim.lr_inr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=10, min_lr=1e-5)

    # ---- Run dir ----
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(__file__).resolve().parent / "trainings" / f"uq_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir: {run_dir}")

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), f,
                  default_flow_style=False)
    with open(run_dir / "coef_norm.pkl", "wb") as f:
        pickle.dump(coef_norm, f)
    with open(run_dir / "split.yaml", "w") as f:
        yaml.dump({"train_idx": tr_idx, "val_idx": va_idx,
                   "seed": cfg.optim.val_split_seed}, f)

    # ---- Restart ----
    start_epoch = 0
    if cfg.restart_training and cfg.saved_model_path \
            and os.path.exists(cfg.saved_model_path):
        ckpt = torch.load(cfg.saved_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"resumed from epoch {start_epoch}")

    # ---- Training loop ----
    best_val = float('inf')
    train_hist, val_hist = [], []
    loss_log = run_dir / "loss.txt"
    with open(loss_log, "w") as f:
        f.write("epoch\ttrain\tval\n")

    def save_ckpt(path, epoch, is_best=False):
        payload = {
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "coef_norm": coef_norm,
            "is_best": is_best,
        }
        torch.save(payload, path)
        print(f"    SAVED {path.name} at epoch {epoch}"
              + (" (best)" if is_best else ""))

    for epoch in tqdm(range(start_epoch, cfg.optim.epochs)):
        # --- train ---
        model.train()
        train_sub = subsample_dataset(train_dataset, cfg.optim.num_points)
        train_loader = DataLoader(train_sub, batch_size=cfg.optim.batch_size,
                                  shuffle=True)
        fit_tr = fit_tr_field = 0.0
        for graph in train_loader:
            n_samples = len(graph)
            graph = graph.to(device)
            out = training_step(model, graph, latent_cond=True,
                                predict_scalars=False)
            optimizer.zero_grad()
            out["loss"].backward()
            nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            fit_tr += out["loss"].item() * n_samples
            fit_tr_field += out["field_loss"].item() * n_samples
        tr_loss = fit_tr / ntrain
        train_hist.append({"epoch": epoch, "loss": tr_loss,
                           "field_loss": fit_tr_field / ntrain})
        wandb.log({"epoch": epoch, "train_loss": tr_loss,
                   "train_field_loss": fit_tr_field / ntrain})

        # --- val (model in train() so dropout is ON, matching training) ---
        model.train()
        val_sub = subsample_dataset(val_dataset, cfg.optim.num_points)
        val_loader = DataLoader(val_sub, batch_size=cfg.optim.batch_size,
                                shuffle=False)
        fit_va = fit_va_field = 0.0
        comp = np.zeros(cfg.inr.out_dim)
        with torch.no_grad():
            for graph in val_loader:
                n_samples = len(graph)
                graph = graph.to(device)
                out = training_step(model, graph, latent_cond=True,
                                    predict_scalars=False)
                fit_va += out["loss"].item() * n_samples
                fit_va_field += out["field_loss"].item() * n_samples
                comp += out["field_loss_components"].cpu().numpy() * n_samples
        va_loss = fit_va / nval
        va_field = fit_va_field / nval
        comp = comp / nval
        val_hist.append({"epoch": epoch, "loss": va_loss, "field_loss": va_field})
        scheduler.step(va_loss)
        wandb.log({"epoch": epoch, "val_loss": va_loss, "val_field_loss": va_field,
                   "val_field_loss_u": comp[0], "val_field_loss_v": comp[1],
                   "val_field_loss_p": comp[2], "val_field_loss_nut": comp[3]})
        print(f"epoch {epoch:4d} | train {tr_loss:.6f} | val {va_loss:.6f}")
        with open(loss_log, "a") as f:
            f.write(f"{epoch}\t{tr_loss:.6f}\t{va_loss:.6f}\n")

        # --- checkpointing ---
        if va_loss < best_val:
            best_val = va_loss
            save_ckpt(run_dir / "best.pt", epoch, is_best=True)
        if (epoch + 1) % cfg.optim.checkpoint_every_n_epochs == 0:
            save_ckpt(run_dir / f"epoch_{epoch:04d}.pt", epoch)

        # loss curve (overwrite each epoch)
        plt.figure()
        plt.plot([h["epoch"] for h in train_hist],
                 [h["loss"] for h in train_hist], label="train")
        plt.plot([h["epoch"] for h in val_hist],
                 [h["loss"] for h in val_hist], label="val")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.yscale("log")
        plt.legend()
        plt.savefig(run_dir / "loss.png")
        plt.close()

    # final: save "last" copy alongside best.pt
    save_ckpt(run_dir / "last.pt", cfg.optim.epochs - 1)
    print("=" * 60)
    print("TRAINING COMPLETE")
    print(f"  best val loss: {best_val:.6f}")
    print(f"  run_dir:       {run_dir}")
    print(f"  checkpoints:   best.pt, last.pt, epoch_*.pt")
    print("=" * 60)
    wandb.finish()
    print("wandb.finish() returned")


if __name__ == "__main__":
    main()
