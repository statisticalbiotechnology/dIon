#!/usr/bin/env python
"""
Cache all embeddings for R/K end-AA probing, run a cosine‐kNN probe,
then train a tiny MLP head manually, saving & evaluating the best model.
"""
import os
import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb

from src.parse_args import create_output_dirs, parse_args_and_config
import src.models.custom.encoder as encoders
import src.models.dc_models.dc_encoder as dc_encoders
from src.wrappers.pretrain_wrappers import (
    dIonPretrainWrapper,
)
import src.utils
from src.collate_functions import pad_peaks

# mapping names → classes
ENCODER_DICT = {**encoders.__dict__, **dc_encoders.__dict__}
WRAPPER_DICT = {
    "dion": dIonPretrainWrapper,
}


def train_manual(
    model: nn.Module,
    train_emb,
    train_lbl,
    val_emb,
    val_lbl,
    test_emb,
    test_lbl,
    lr,
    n_epochs,
    batch_size,
    device,
    logger=None,
):
    """
    A single tqdm loop over epochs, tracks best val_loss, reloads best model,
    and evaluates on test set.
    """
    tr_loader = DataLoader(
        TensorDataset(train_emb, train_lbl), batch_size=batch_size, shuffle=True
    )
    va_loader = DataLoader(
        TensorDataset(val_emb, val_lbl), batch_size=batch_size, shuffle=False
    )
    te_loader = (
        DataLoader(
            TensorDataset(test_emb, test_lbl), batch_size=batch_size, shuffle=False
        )
        if test_emb is not None
        else None
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1

    pbar = tqdm(range(1, n_epochs + 1), desc="Epoch")
    for epoch in pbar:
        # — train —
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for X, Y in tr_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = F.cross_entropy(logits, Y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * X.size(0)
            train_n += X.size(0)
        train_loss = train_loss_sum / train_n

        # — val —
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        val_corr = 0
        with torch.no_grad():
            for X, Y in va_loader:
                X, Y = X.to(device), Y.to(device)
                logits = model(X)
                loss = F.cross_entropy(logits, Y)
                val_loss_sum += loss.item() * X.size(0)
                val_n += X.size(0)
                val_corr += (logits.argmax(1) == Y).sum().item()
        val_loss = val_loss_sum / val_n
        val_acc = val_corr / val_n

        # — track best —
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        # — update tqdm & W&B —
        pbar.set_postfix(
            {
                "tr_loss": f"{train_loss:.4f}",
                "va_loss": f"{val_loss:.4f}",
                "va_acc": f"{val_acc:.4f}",
            }
        )
        if logger:
            logger.experiment.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch": epoch,
                }
            )

    # — after all epochs: log best_val_loss & epoch, reload best model —
    if best_state is not None:
        model.load_state_dict(best_state)
    if logger:
        logger.experiment.log(
            {
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
            }
        )

    # — final test using best model —
    if te_loader:
        model.eval()
        test_corr = test_n = 0
        with torch.no_grad():
            for X, Y in te_loader:
                X, Y = X.to(device), Y.to(device)
                pred = model(X).argmax(1)
                test_corr += (pred == Y).sum().item()
                test_n += X.size(0)
        test_acc = test_corr / test_n
        print(f"→ Best‐model test_acc = {test_acc:.4f}")
        if logger:
            logger.experiment.log({"test_acc": test_acc})

    return model


def main():
    # 0) k for the kNN probe
    K = 401

    # 1) parse args & set up
    args, pretrain_cfg, ds_cfg, _ = parse_args_and_config()
    create_output_dirs(args, is_main_process=utils.get_rank() == 0)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.matmul_precision:
        torch.set_float32_matmul_precision(args.matmul_precision)

    # 2) W&B logger
    logger = None
    if args.log_wandb and utils.get_rank() == 0:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={**vars(args), **{"downstream_config": ds_cfg}},
            dir=args.log_dir,
            user="allow",
        )
        logger = WandbLogger(experiment=run)

    # 3) load pretrained wrapper & extract embedder
    device = {"cpu": "cpu", "gpu": "cuda", "mps": "mps"}[args.accelerator]
    dev = torch.device(device)

    Enc = ENCODER_DICT[args.encoder_model]
    encoder = Enc(
        use_charge=args.use_charge,
        use_mass=args.use_mass,
        use_energy=args.use_energy,
        dropout=pretrain_cfg[args.pretraining_task].get("dropout", 0.0),
        cls_token=args.cls_token,
    )
    Wrap = WRAPPER_DICT[args.pretraining_task]
    pl_wrapper = Wrap.load_from_checkpoint(
        args.encoder_weights,
        map_location=dev,
        global_args=args,
        encoder=encoder,
        task_dict=pretrain_cfg[args.pretraining_task],
    ).eval()
    embedder = pl_wrapper.get_embedder().eval()

    # 4) load Parquet splits with pandas
    train_df = pd.read_parquet(args.downstream_train_path)
    val_df = pd.read_parquet(args.downstream_val_path)
    test_df = pd.read_parquet(args.downstream_test_path) if args.test_on_end else None

    # 5) build DataLoaders for raw caching (uses pad_peaks)
    from functools import partial

    collate_fn = partial(
        pad_peaks,
        max_peaks=ds_cfg["top_peaks"],
        precursor_mz_name="precursor_mz",
        precursor_mass_name=False,
        filter_method=args.peak_filter_method,
        intensity_scaling=args.intensity_scaling,
        min_mz=args.min_mz,
        max_mz=args.max_mz,
        min_intensity=args.min_intensity,
        remove_precursor_tol=args.remove_precursor_tol,
    )

    class DFDS(torch.utils.data.Dataset):
        def __init__(self, df):
            self.df = df.reset_index(drop=True)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            row = self.df.iloc[i]
            return {
                "mz_array": torch.tensor(row["mz_array"], dtype=torch.float32),
                "intensity_array": torch.tensor(
                    row["intensity_array"], dtype=torch.float32
                ),
                "precursor_mz": torch.tensor(row["precursor_mz"], dtype=torch.float32),
                "precursor_charge": torch.tensor(
                    row["precursor_charge"], dtype=torch.long
                ),
                "label": torch.tensor(row["label"], dtype=torch.long),
            }

    train_loader = DataLoader(
        DFDS(train_df), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        DFDS(val_df), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = (
        DataLoader(
            DFDS(test_df),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        if test_df is not None
        else None
    )

    # 6) cache embeddings + labels
    def cache(loader, split):
        Xs, Ys = [], []
        for batch in tqdm(loader, desc=f"Caching {split}", leave=False):
            mz = batch["mz_array"].to(dev)
            it = batch["intensity_array"].to(dev)
            specs = torch.stack([mz, it], dim=-1)
            B, L, _ = specs.shape
            lengths = batch["peak_lengths"].view(B).to(dev)
            pad_mask = (
                torch.arange(L, device=dev)
                .unsqueeze(0)
                .expand(B, L)
                .ge(lengths.unsqueeze(1))
            )
            mass = batch["precursor_mz"].to(dev) if args.use_mass else None
            charge = batch["precursor_charge"].to(dev) if args.use_charge else None
            with torch.no_grad():
                emb = embedder(specs, pad_mask, mass, charge)
            Xs.append(emb.cpu())
            Ys.append(batch["label"])
        return torch.cat(Xs, dim=0), torch.cat(Ys, dim=0)

    train_emb, train_lbl = cache(train_loader, "train")
    val_emb, val_lbl = cache(val_loader, "val")
    if test_loader:
        test_emb, test_lbl = cache(test_loader, "test")
    else:
        test_emb = test_lbl = None

    # free memory
    running_units = embedder.running_units
    del embedder, pl_wrapper, encoder, Wrap, Enc

    # 7) quick cosine-kNN probe
    print(f"→ Running cosine‐kNN probe (k={K})…")
    T = F.normalize(train_emb, dim=1)
    Q = F.normalize(val_emb, dim=1)
    topk = (Q @ T.t()).topk(K, dim=1).indices
    preds = torch.stack([torch.bincount(train_lbl[n]).argmax() for n in topk])
    knn_va = (preds == val_lbl).float().mean().item()
    print(f"→ kNN val_acc = {knn_va:.4f}")
    if logger:
        logger.experiment.log({"knn_val_acc": knn_va})

    if test_emb is not None:
        QT = F.normalize(test_emb, dim=1)
        topk_t = (QT @ T.t()).topk(K, dim=1).indices
        preds_t = torch.stack([torch.bincount(train_lbl[n]).argmax() for n in topk_t])
        knn_te = (preds_t == test_lbl).float().mean().item()
        print(f"→ kNN test_acc = {knn_te:.4f}")
        if logger:
            logger.experiment.log({"knn_test_acc": knn_te})

    # 8) train a tiny head manually, saving + evaluating best
    head = nn.Linear(running_units, int(train_lbl.unique().numel()))
    train_manual(
        head,
        train_emb,
        train_lbl,
        val_emb,
        val_lbl,
        test_emb,
        test_lbl,
        lr=ds_cfg[args.downstream_task]["blr"],
        n_epochs=ds_cfg[args.downstream_task]["epochs"],
        batch_size=ds_cfg[args.downstream_task]["batch_size"],
        device=dev,
        logger=logger,
    )

    if logger:
        wandb.finish()


if __name__ == "__main__":
    main()
