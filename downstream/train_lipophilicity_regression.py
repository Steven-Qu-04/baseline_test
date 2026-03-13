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
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
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


class MolecularGTNRegressor(nn.Module):
    def __init__(self, backbone_config: PipelineConfig, head_dropout: float):
        super().__init__()
        dims = feature_dimensions()
        self.backbone = MolecularGTN(dims["atom_dim"], dims["bond_dim"], backbone_config)
        hidden = backbone_config.hidden_dim
        self.regression_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        encoded = self.backbone.encode(batch, compute_bond_embeddings=False)
        pred = self.regression_head(encoded.mol_embeddings).squeeze(-1)
        return pred


def load_pretrained_backbone(model: MolecularGTNRegressor, checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", {})
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    return {
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "checkpoint_config": checkpoint.get("config", {}),
    }


def evaluate_batch_model(model: nn.Module, loader: DataLoader, device: torch.device, desc: str) -> dict[str, float]:
    model.eval()
    preds = []
    targets = []
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = batch.to(device)
            y = batch.y.view(-1).to(device)
            pred = model(batch)
            loss = F.mse_loss(pred, y)
            total_loss += float(loss.item())
            steps += 1
            preds.append(pred.detach().cpu())
            targets.append(y.detach().cpu())
    return regression_metrics(preds, targets, total_loss, steps)


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


def plot_losses(train_losses: list[float], val_losses: list[float], out_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=epochs, y=train_losses, marker="o", linewidth=1.6, label="train_mse", ax=ax)
    sns.lineplot(x=epochs, y=val_losses, marker="o", linewidth=1.6, label="val_mse", ax=ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Downstream Regression Loss Curves")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def extract_embeddings(
    backbone: MolecularGTN,
    dataset: Dataset,
    batch_size: int,
    infer_device: torch.device,
    split_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_regression)
    backbone.eval()
    x_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"embed-{split_name}", leave=False):
            batch = batch.to(infer_device)
            encoded = backbone.encode(batch, compute_bond_embeddings=False)
            x_parts.append(encoded.mol_embeddings.detach().cpu())
            y_parts.append(batch.y.view(-1).detach().cpu())
    return torch.cat(x_parts, dim=0), torch.cat(y_parts, dim=0)


def evaluate_head(head: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int, desc: str) -> dict[str, float]:
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    head.eval()
    pred_chunks = []
    target_chunks = []
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc=desc, leave=False):
            pred = head(xb).squeeze(-1)
            loss = F.mse_loss(pred, yb)
            total_loss += float(loss.item())
            steps += 1
            pred_chunks.append(pred.detach().cpu())
            target_chunks.append(yb.detach().cpu())
    return regression_metrics(pred_chunks, target_chunks, total_loss, steps)


def run_all_cpu(
    model: MolecularGTNRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    args: argparse.Namespace,
) -> tuple[list[float], list[float], dict[str, float], dict[str, float], dict]:
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history_train: list[float] = []
    history_val: list[float] = []
    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        steps = 0
        for batch in tqdm(train_loader, desc=f"train-epoch-{epoch}", leave=False):
            batch = batch.to(device)
            y = batch.y.view(-1).to(device)
            pred = model(batch)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            steps += 1

        train_mse = train_loss / max(1, steps)
        val_metrics = evaluate_batch_model(model, val_loader, device, desc=f"val-epoch-{epoch}")
        val_rmse = val_metrics["rmse"]
        history_train.append(train_mse)
        history_val.append(val_metrics["mse"])
        print(
            f"[epoch {epoch:03d}] train_mse={train_mse:.6f} "
            f"val_mse={val_metrics['mse']:.6f} val_rmse={val_rmse:.6f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No checkpoint state captured during training.")

    model.load_state_dict(best_state, strict=True)
    val_best = evaluate_batch_model(model, val_loader, device, desc="val-best")
    test_best = evaluate_batch_model(model, test_loader, device, desc="test-best")

    save_obj = {
        "mode": "all_cpu",
        "model_state": model.state_dict(),
    }
    return history_train, history_val, val_best, test_best, save_obj


def run_hybrid(
    model: MolecularGTNRegressor,
    train_set: Dataset,
    val_set: Dataset,
    test_set: Dataset,
    args: argparse.Namespace,
) -> tuple[list[float], list[float], dict[str, float], dict[str, float], dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("Hybrid mode requires CUDA for backbone inference, but CUDA is unavailable.")

    print("[STAGE] Extracting frozen backbone embeddings on GPU...")
    infer_device = torch.device("cuda")
    model.backbone = model.backbone.to(infer_device)
    model.backbone.eval()
    for p in model.backbone.parameters():
        p.requires_grad = False

    x_train, y_train = extract_embeddings(model.backbone, train_set, args.batch_size, infer_device, "train")
    x_val, y_val = extract_embeddings(model.backbone, val_set, args.batch_size, infer_device, "val")
    x_test, y_test = extract_embeddings(model.backbone, test_set, args.batch_size, infer_device, "test")

    print("[STAGE] Training regression head on CPU...")
    head = model.regression_head.to(torch.device("cpu"))
    optimizer = AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    history_train: list[float] = []
    history_val: list[float] = []
    best_val_rmse = float("inf")
    best_head_state = None

    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0.0
        steps = 0
        for xb, yb in tqdm(train_loader, desc=f"head-train-epoch-{epoch}", leave=False):
            pred = head(xb).squeeze(-1)
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            steps += 1

        train_mse = total / max(1, steps)
        val_metrics = evaluate_head(head, x_val, y_val, args.batch_size, desc=f"head-val-epoch-{epoch}")
        history_train.append(train_mse)
        history_val.append(val_metrics["mse"])

        print(
            f"[epoch {epoch:03d}] train_mse={train_mse:.6f} "
            f"val_mse={val_metrics['mse']:.6f} val_rmse={val_metrics['rmse']:.6f}"
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_head_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}

    if best_head_state is None:
        raise RuntimeError("No best head state captured during hybrid training.")

    head.load_state_dict(best_head_state, strict=True)
    val_best = evaluate_head(head, x_val, y_val, args.batch_size, desc="head-val-best")
    test_best = evaluate_head(head, x_test, y_test, args.batch_size, desc="head-test-best")

    save_obj = {
        "mode": "hybrid",
        "backbone_state": {k: v.detach().cpu() for k, v in model.backbone.state_dict().items()},
        "head_state": head.state_dict(),
    }
    return history_train, history_val, val_best, test_best, save_obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate Lipophilicity regression from a pretrained checkpoint.")
    parser.add_argument("--lmdb", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--head-dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execution-mode", choices=["all_cpu", "hybrid"], default="all_cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    print("[STAGE 1/5] Loading dataset and making 80/10/10 split...")
    dataset = RegressionLmdbDataset(args.lmdb)
    n_total = len(dataset)
    train_idx, val_idx, test_idx = split_indices(n_total, args.seed)

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_regression)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_regression)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_regression)

    print("[STAGE 2/5] Building model and loading pretrained backbone...")
    checkpoint_obj = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone_cfg = build_backbone_config(checkpoint_obj.get("config", {}))
    model = MolecularGTNRegressor(backbone_cfg, head_dropout=args.head_dropout)
    preload_info = load_pretrained_backbone(model, args.checkpoint)

    print(f"[INFO] execution_mode={args.execution_mode}")
    print(f"[INFO] split_sizes train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    print("[STAGE 3/5] Training...")
    if args.execution_mode == "all_cpu":
        history_train, history_val, val_best, test_best, save_obj = run_all_cpu(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args,
        )
    else:
        history_train, history_val, val_best, test_best, save_obj = run_hybrid(
            model=model,
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            args=args,
        )

    print("[STAGE 4/5] Saving model, metrics, and plots...")
    pretrain_epoch = parse_pretrain_epoch(args.checkpoint)

    torch.save(
        {
            **save_obj,
            "checkpoint_source": str(args.checkpoint),
            "pretrain_epoch": pretrain_epoch,
            "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
            "seed": args.seed,
        },
        args.output_dir / "best_model.pth",
    )

    plot_losses(history_train, history_val, args.output_dir / "loss_curves.png")

    metrics = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_name": args.checkpoint.name,
        "pretrain_epoch": pretrain_epoch,
        "dataset_size": n_total,
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "execution_mode": args.execution_mode,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "best_val_rmse": float(val_best["rmse"]),
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
    (args.output_dir / "split_indices.json").write_text(
        json.dumps({"train": train_idx, "val": val_idx, "test": test_idx}, indent=2), encoding="utf-8"
    )

    print("[STAGE 5/5] Done")
    print("[OK] Downstream evaluation complete")
    print(json.dumps({"pretrain_epoch": pretrain_epoch, "test": test_best}, indent=2))


if __name__ == "__main__":
    main()
