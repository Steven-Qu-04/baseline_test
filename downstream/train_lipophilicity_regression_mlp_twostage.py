#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from mol_gtn.config import PipelineConfig
from mol_gtn.features import feature_dimensions
from mol_gtn.lmdb_io import deserialize_data, open_lmdb
from mol_gtn.model import MolecularGTN


class RegressionLmdbDataset(Dataset):
    def __init__(self, lmdb_path: Path):
        self.lmdb_path = str(lmdb_path)
        self.env = None
        self.length = 0

    def _ensure_env(self) -> None:
        if self.env is None:
            self.env = open_lmdb(self.lmdb_path, readonly=True)
            with self.env.begin() as txn:
                self.length = int((txn.get(b"length") or b"0").decode())

    def __len__(self) -> int:
        self._ensure_env()
        return self.length

    def __getitem__(self, index: int) -> Data:
        self._ensure_env()
        with self.env.begin() as txn:
            blob = txn.get(f"{index:012d}".encode())
        if blob is None:
            raise IndexError(index)
        return deserialize_data(blob)


def collate_regression(batch: list[Data]) -> Batch:
    return Batch.from_data_list(batch)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(length: int, seed: int) -> tuple[list[int], list[int], list[int]]:
    if length < 10:
        raise ValueError(f"Dataset too small for 80/10/10 split: n={length}")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(length, generator=g).tolist()
    n_train = int(length * 0.8)
    n_val = int(length * 0.1)
    n_test = length - n_train - n_val
    if n_val == 0 or n_test == 0:
        raise ValueError(f"Invalid split sizes for n={length}: train={n_train} val={n_val} test={n_test}")
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx


def build_backbone_config(checkpoint_config: dict) -> PipelineConfig:
    cfg = PipelineConfig()
    for f in fields(PipelineConfig):
        if f.name in checkpoint_config:
            setattr(cfg, f.name, checkpoint_config[f.name])
    return cfg


def parse_pretrain_epoch(path: Path) -> int:
    m = re.search(r"(?:^|_)epoch_(\d+)\.pth$", path.name)
    if m:
        return int(m.group(1))
    m = re.search(r"epoch_(\d+)", path.name)
    return int(m.group(1)) if m else -1


class MolecularGTNRegressorMLP(nn.Module):
    def __init__(self, backbone_config: PipelineConfig):
        super().__init__()
        dims = feature_dimensions()
        self.backbone = MolecularGTN(dims["atom_dim"], dims["bond_dim"], backbone_config)
        hidden = backbone_config.hidden_dim
        hidden_mid = max(1, hidden // 2)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden, hidden_mid),
            nn.BatchNorm1d(hidden_mid),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_mid, 1),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        encoded = self.backbone.encode(batch, compute_bond_embeddings=False)
        return self.regression_head(encoded.mol_embeddings).squeeze(-1)


def load_pretrained_backbone(model: MolecularGTNRegressorMLP, checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", {})
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    return {
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "checkpoint_config": checkpoint.get("config", {}),
    }


def regression_metrics(pred_chunks: list[torch.Tensor], target_chunks: list[torch.Tensor], total_loss: float, steps: int) -> dict[str, float]:
    if not pred_chunks:
        return {"mse": math.nan, "rmse": math.nan, "mae": math.nan, "r2": math.nan}
    pred_t = torch.cat(pred_chunks).numpy()
    true_t = torch.cat(target_chunks).numpy()
    mse = float(np.mean((pred_t - true_t) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(pred_t - true_t)))
    ss_res = float(np.sum((true_t - pred_t) ** 2))
    ss_tot = float(np.sum((true_t - np.mean(true_t)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": float(r2),
        "avg_batch_mse": total_loss / max(1, steps),
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, desc: str) -> dict[str, float]:
    model.eval()
    preds = []
    targets = []
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = batch.to(device)
            y = batch.y.view(-1)
            pred = model(batch)
            loss = F.mse_loss(pred, y)
            total_loss += float(loss.item())
            steps += 1
            preds.append(pred.detach().cpu())
            targets.append(y.detach().cpu())
    return regression_metrics(preds, targets, total_loss, steps)


def set_backbone_trainable(model: MolecularGTNRegressorMLP, trainable: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def build_optimizer_for_stage(model: MolecularGTNRegressorMLP, stage: str, args: argparse.Namespace) -> AdamW:
    if stage == "warmup":
        return AdamW(model.regression_head.parameters(), lr=args.warmup_lr, weight_decay=args.weight_decay)
    if stage == "finetune":
        return AdamW(
            [
                {"params": model.backbone.parameters(), "lr": args.finetune_backbone_lr},
                {"params": model.regression_head.parameters(), "lr": args.finetune_head_lr},
            ],
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unknown stage: {stage}")


def plot_phase_loss_curves(history: list[dict], warmup_epochs: int, out_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    warmup = [h for h in history if h["stage"] == "warmup"]
    finetune = [h for h in history if h["stage"] == "finetune"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    if warmup:
        x = [h["epoch"] for h in warmup]
        y_train = [h["train_mse"] for h in warmup]
        y_val = [h["val_mse"] for h in warmup]
        sns.lineplot(x=x, y=y_train, marker="o", ax=ax1, label="train_mse")
        sns.lineplot(x=x, y=y_val, marker="o", ax=ax1, label="val_mse")
    ax1.set_title("Warmup Phase")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")

    if finetune:
        x = [h["epoch"] for h in finetune]
        y_train = [h["train_mse"] for h in finetune]
        y_val = [h["val_mse"] for h in finetune]
        sns.lineplot(x=x, y=y_train, marker="o", ax=ax2, label="train_mse")
        sns.lineplot(x=x, y=y_val, marker="o", ax=ax2, label="val_mse")
    ax2.set_title("Fine-tuning Phase")
    ax2.set_xlabel("Epoch")

    fig.suptitle("Loss Curves by Training Phase")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage MLP downstream training for Lipophilicity.")
    parser.add_argument("--lmdb", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-backbone-lr", type=float, default=1e-5)
    parser.add_argument("--finetune-head-lr", type=float, default=5e-4)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.warmup_epochs < 0 or args.warmup_epochs > args.epochs:
        raise ValueError("--warmup-epochs must be in [0, epochs]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu if needed.")
    device = torch.device(args.device)

    print("[STAGE 1/6] Loading dataset and split...")
    dataset = RegressionLmdbDataset(args.lmdb)
    n_total = len(dataset)
    train_idx, val_idx, test_idx = split_indices(n_total, args.seed)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, collate_fn=collate_regression)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_regression)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_regression)

    print("[STAGE 2/6] Building model and loading pretrained backbone...")
    checkpoint_obj = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone_cfg = build_backbone_config(checkpoint_obj.get("config", {}))
    model = MolecularGTNRegressorMLP(backbone_cfg).to(device)
    preload_info = load_pretrained_backbone(model, args.checkpoint)

    print("[STAGE 3/6] Two-stage training with early stopping...")
    history: list[dict] = []
    best_val_rmse = float("inf")
    best_epoch = -1
    best_state = None
    no_improve = 0

    current_stage = "warmup" if args.warmup_epochs > 0 else "finetune"
    set_backbone_trainable(model, trainable=(current_stage == "finetune"))
    optimizer = build_optimizer_for_stage(model, current_stage, args)

    for epoch in range(1, args.epochs + 1):
        next_stage = "warmup" if epoch <= args.warmup_epochs else "finetune"
        if next_stage != current_stage:
            current_stage = next_stage
            set_backbone_trainable(model, trainable=True)
            optimizer = build_optimizer_for_stage(model, current_stage, args)
            print(f"[INFO] Switched to stage={current_stage} at epoch={epoch}")

        model.train()
        total_train_loss = 0.0
        train_steps = 0
        for batch in tqdm(train_loader, desc=f"train-{current_stage}-ep{epoch}", leave=False):
            batch = batch.to(device)
            y = batch.y.view(-1)
            pred = model(batch)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.item())
            train_steps += 1

        train_mse = total_train_loss / max(1, train_steps)
        val_metrics = evaluate(model, val_loader, device, desc=f"val-ep{epoch}")

        history.append(
            {
                "epoch": epoch,
                "stage": current_stage,
                "train_mse": train_mse,
                "val_mse": val_metrics["mse"],
                "val_rmse": val_metrics["rmse"],
            }
        )

        print(
            f"[epoch {epoch:03d}][{current_stage}] train_mse={train_mse:.6f} "
            f"val_mse={val_metrics['mse']:.6f} val_rmse={val_metrics['rmse']:.6f}"
        )

        improvement = best_val_rmse - val_metrics["rmse"]
        if improvement >= args.early_stop_min_delta:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.early_stop_patience:
            print(
                f"[INFO] Early stopping at epoch={epoch}; "
                f"best_epoch={best_epoch}, best_val_rmse={best_val_rmse:.6f}"
            )
            break

    if best_state is None:
        raise RuntimeError("No best model state captured during training.")

    print("[STAGE 4/6] Restoring best model and evaluating...")
    model.load_state_dict(best_state, strict=True)
    val_best = evaluate(model, val_loader, device, desc="val-best")
    test_best = evaluate(model, test_loader, device, desc="test-best")

    print("[STAGE 5/6] Saving artifacts...")
    pretrain_epoch = parse_pretrain_epoch(args.checkpoint)

    torch.save(
        {
            "model_state": model.state_dict(),
            "checkpoint_source": str(args.checkpoint),
            "pretrain_epoch": pretrain_epoch,
            "best_epoch": best_epoch,
            "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
            "seed": args.seed,
        },
        args.output_dir / "best_model.pth",
    )

    (args.output_dir / "split_indices.json").write_text(
        json.dumps({"train": train_idx, "val": val_idx, "test": test_idx}, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "epoch_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    plot_phase_loss_curves(history, args.warmup_epochs, args.output_dir / "loss_curves_warmup_finetune.png")

    metrics = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_name": args.checkpoint.name,
        "pretrain_epoch": pretrain_epoch,
        "dataset_size": n_total,
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "training": {
            "epochs": args.epochs,
            "warmup_epochs": args.warmup_epochs,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "warmup_lr": args.warmup_lr,
            "finetune_backbone_lr": args.finetune_backbone_lr,
            "finetune_head_lr": args.finetune_head_lr,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "best_epoch": best_epoch,
            "best_val_rmse": best_val_rmse,
        },
        "val": val_best,
        "test": test_best,
        "preload": {
            "checkpoint_epoch": preload_info["checkpoint_epoch"],
            "missing_key_count": len(preload_info["missing_keys"]),
            "unexpected_key_count": len(preload_info["unexpected_keys"]),
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    print("[STAGE 6/6] Done")
    print("[OK] Two-stage MLP downstream evaluation complete")
    print(json.dumps({"pretrain_epoch": pretrain_epoch, "test": test_best}, indent=2))


if __name__ == "__main__":
    main()
