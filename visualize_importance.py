#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Batch, Data
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matplotlib import cm
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from mol_gtn.config import PipelineConfig
from mol_gtn.features import atom_features, bond_features_and_index, feature_dimensions, validate_feature_dimensions
from mol_gtn.lap_pe import compute_laplacian_positional_encoding
from mol_gtn.model import MolecularGTN


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_backbone_config(checkpoint_config: dict) -> PipelineConfig:
    cfg = PipelineConfig()
    for f in fields(PipelineConfig):
        if f.name in checkpoint_config:
            setattr(cfg, f.name, checkpoint_config[f.name])
    return cfg


def split_indices(length: int, seed: int) -> tuple[list[int], list[int], list[int]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(length, generator=g).tolist()
    n_train = int(length * 0.8)
    n_val = int(length * 0.1)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx


def make_data_from_smiles(smiles: str, target: float, lap_pe_dim: int) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    Chem.SanitizeMol(mol)
    x = atom_features(mol)
    edge_index, edge_attr = bond_features_and_index(mol)
    validate_feature_dimensions(x, edge_attr)
    lap_pe, lap_pe_valid_mask = compute_laplacian_positional_encoding(
        num_nodes=mol.GetNumAtoms(), edge_index=edge_index, k=lap_pe_dim
    )
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        lap_pe=lap_pe,
        lap_pe_valid_mask=lap_pe_valid_mask,
        padding_mask=torch.zeros(mol.GetNumAtoms(), dtype=torch.bool),
        y=torch.tensor([float(target)], dtype=torch.float32),
        smiles=smiles,
    )


def load_dataset(csv_path: Path, lap_pe_dim: int) -> list[Data]:
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        if "smiles" not in reader.fieldnames or "exp" not in reader.fieldnames:
            raise ValueError(f"Expected columns smiles/exp, got {reader.fieldnames}")
        for row in tqdm(reader, desc="featurize-csv"):
            try:
                smiles = str(row["smiles"]).strip()
                target = float(row["exp"])
                rows.append(make_data_from_smiles(smiles, target, lap_pe_dim))
            except Exception:
                continue
    if not rows:
        raise RuntimeError("No valid molecules loaded from CSV")
    return rows


def load_checkpoint(model: MolecularGTNRegressorMLP, checkpoint_path: Path) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", {})
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    return {
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "missing_backbone_keys": len(missing),
        "unexpected_backbone_keys": len(unexpected),
        "config": ckpt.get("config", {}),
    }


def extract_embeddings(
    backbone: MolecularGTN,
    data_list: list[Data],
    indices: list[int],
    batch_size: int,
    device: torch.device,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    subset = [data_list[i] for i in indices]
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=Batch.from_data_list)
    backbone.eval()
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            batch = batch.to(device)
            enc = backbone.encode(batch, compute_bond_embeddings=False).mol_embeddings
            xs.append(enc.detach().cpu())
            ys.append(batch.y.view(-1).detach().cpu())
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def fit_regression_head(
    model: MolecularGTNRegressorMLP,
    data_list: list[Data],
    train_idx: list[int],
    val_idx: list[int],
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> None:
    print("[INFO] Fitting downstream regression head on frozen backbone embeddings...")
    model.backbone.eval().to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False

    x_train, y_train = extract_embeddings(model.backbone, data_list, train_idx, batch_size, device, "embed-train")
    x_val, y_val = extract_embeddings(model.backbone, data_list, val_idx, batch_size, device, "embed-val")

    head = model.regression_head.to(torch.device("cpu"))
    optimizer = AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)

    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_rmse = float("inf")
    best_state = None
    patience = 8
    no_improve = 0

    for epoch in range(1, epochs + 1):
        head.train()
        total = 0.0
        steps = 0
        for xb, yb in tqdm(train_loader, desc=f"head-train-ep{epoch}", leave=False):
            pred = head(xb).squeeze(-1)
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            steps += 1

        head.eval()
        with torch.no_grad():
            pred_val = head(x_val).squeeze(-1)
            val_mse = F.mse_loss(pred_val, y_val).item()
            val_rmse = float(np.sqrt(val_mse))
        print(f"[head epoch {epoch:03d}] train_mse={total/max(1,steps):.6f} val_rmse={val_rmse:.6f}")

        if best_rmse - val_rmse >= 1e-4:
            best_rmse = val_rmse
            best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[INFO] Head early stopping at epoch {epoch}")
            break

    if best_state is not None:
        head.load_state_dict(best_state, strict=True)
    model.regression_head = head.to(device)
    model.eval()


def integrated_gradients_atom_scores(
    model: MolecularGTNRegressorMLP,
    data: Data,
    device: torch.device,
    steps: int,
) -> tuple[np.ndarray, float]:
    model.eval()
    batch = Batch.from_data_list([data]).to(device)
    x_input = batch.x.detach()
    baseline = torch.zeros_like(x_input)
    total_grad = torch.zeros_like(x_input)

    for alpha in torch.linspace(0.0, 1.0, steps, device=device):
        x_step = baseline + alpha * (x_input - baseline)
        x_step.requires_grad_(True)

        step_batch = batch.clone()
        step_batch.x = x_step

        pred = model(step_batch).sum()
        grad = torch.autograd.grad(pred, x_step, retain_graph=False, create_graph=False)[0]
        total_grad += grad

    avg_grad = total_grad / float(steps)
    attr = (x_input - baseline) * avg_grad
    atom_scores = attr.abs().sum(dim=1).detach().cpu().numpy()

    with torch.no_grad():
        pred_value = float(model(batch).item())

    return atom_scores, pred_value


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    smin = float(scores.min())
    smax = float(scores.max())
    if smax - smin < 1e-12:
        return np.zeros_like(scores)
    return (scores - smin) / (smax - smin)


def draw_saliency_png(smiles: str, atom_scores_norm: np.ndarray, out_png: Path, legend: str) -> None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for drawing: {smiles}")
    rdDepictor.Compute2DCoords(mol)

    cmap = cm.get_cmap("Reds")
    highlight_atoms = list(range(mol.GetNumAtoms()))
    atom_colors = {}
    atom_radii = {}
    for i in highlight_atoms:
        score = float(atom_scores_norm[i]) if i < len(atom_scores_norm) else 0.0
        rgba = cmap(score)
        atom_colors[i] = (float(rgba[0]), float(rgba[1]), float(rgba[2]))
        atom_radii[i] = 0.28 + 0.22 * score

    drawer = rdMolDraw2D.MolDraw2DCairo(700, 500)
    opts = drawer.drawOptions()
    opts.addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        legend=legend,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
    )
    drawer.FinishDrawing()
    out_png.write_bytes(drawer.GetDrawingText())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize atom-level importance using Integrated Gradients for Lipophilicity.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/hy-tmp/result_flash_1m/run_20260312_214955/20260312_214955_epoch_9.pth"),
        help="Backbone checkpoint path.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("/root/baseline_test/datasets/Lipophilicity/Lipophilicity.csv"),
        help="Lipophilicity CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/baseline_test/interpretability_results"),
        help="Output folder for PNGs and summary.",
    )
    parser.add_argument("--num-mols", type=int, default=10)
    parser.add_argument("--ig-steps", type=int, default=64)
    parser.add_argument("--fit-head-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu.")
    device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[STAGE 1/5] Loading checkpoint and dataset...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone_cfg = build_backbone_config(ckpt.get("config", {}))
    model = MolecularGTNRegressorMLP(backbone_cfg).to(device)
    ckpt_info = load_checkpoint(model, args.checkpoint)

    data_list = load_dataset(args.csv, lap_pe_dim=backbone_cfg.lap_pe_dim)
    train_idx, val_idx, test_idx = split_indices(len(data_list), seed=args.seed)

    print("[STAGE 2/5] Fitting regression head for LogP prediction...")
    fit_regression_head(
        model=model,
        data_list=data_list,
        train_idx=train_idx,
        val_idx=val_idx,
        device=device,
        epochs=args.fit_head_epochs,
        batch_size=args.batch_size,
    )

    print("[STAGE 3/5] Computing IG and drawing molecules...")
    selected = test_idx[: args.num_mols]
    summary_rows = []

    for local_i, ds_idx in enumerate(tqdm(selected, desc="visualize-test-mols")):
        data = data_list[ds_idx]
        smiles = str(data.smiles)
        true_y = float(data.y.item())

        atom_scores, pred_y = integrated_gradients_atom_scores(
            model=model,
            data=data,
            device=device,
            steps=args.ig_steps,
        )
        atom_scores_norm = normalize_scores(atom_scores)
        top_atom_idx = int(np.argmax(atom_scores_norm)) if atom_scores_norm.size > 0 else -1

        out_png = args.output_dir / f"mol_{local_i}_saliency.png"
        legend = f"idx={ds_idx} pred={pred_y:.3f} true={true_y:.3f} top_atom={top_atom_idx}"
        draw_saliency_png(smiles, atom_scores_norm, out_png, legend)

        summary_rows.append(
            {
                "mol_index": local_i,
                "dataset_index": ds_idx,
                "smiles": smiles,
                "pred_logp": pred_y,
                "true_logp": true_y,
                "most_important_atom_index": top_atom_idx,
            }
        )

    print("[STAGE 4/5] Writing summary...")
    summary_csv = args.output_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mol_index",
                "dataset_index",
                "smiles",
                "pred_logp",
                "true_logp",
                "most_important_atom_index",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = args.output_dir / "run_metadata.json"
    summary_json.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "csv": str(args.csv),
                "device": args.device,
                "ig_steps": args.ig_steps,
                "fit_head_epochs": args.fit_head_epochs,
                "selected_test_count": len(selected),
                "checkpoint_info": ckpt_info,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[STAGE 5/5] Done")
    print(f"[OK] Wrote saliency images and summary to: {args.output_dir}")


if __name__ == "__main__":
    main()
