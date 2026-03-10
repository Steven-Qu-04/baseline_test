from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import PipelineConfig
from .dataset import LmdbGraphDataset, contrastive_collate
from .features import feature_dimensions
from .losses import nt_xent_loss
from .model import MolecularGTN
from .utils.logging import configure_logging
from .utils.runtime import seed_everything


def run_training(config: PipelineConfig) -> None:
    seed_everything(config.seed)
    logger = configure_logging(config.log_path)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    dataset = LmdbGraphDataset(config.active_lmdb_path(), config)
    if len(dataset) == 0:
        raise RuntimeError(f"No serialized graphs found in {config.active_lmdb_path()}")
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: contrastive_collate(batch, config),
    )
    dims = feature_dimensions()
    model = MolecularGTN(dims["atom_dim"], dims["bond_dim"], config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")
    best_loss = float("inf")
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress = tqdm(loader, desc=f"train-epoch-{epoch + 1}")
        for step, batch in enumerate(progress, start=1):
            anchors = batch["anchors"].to(device)
            positives = batch["positives"].to(device)
            with autocast(enabled=device.type == "cuda"):
                _, anchor_z = model(anchors)
                _, positive_z = model(positives)
                loss = nt_xent_loss(anchor_z, positive_z, config.temperature)
                loss = loss / config.grad_accum_steps
            scaler.scale(loss).backward()
            if step % config.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * config.grad_accum_steps
            progress.set_postfix(loss=f"{running_loss / step:.4f}", eff_pairs=config.batch_size * config.active_num_masked_views() * config.grad_accum_steps)
        if len(loader) % config.grad_accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        epoch_loss = running_loss / max(1, len(loader))
        logger.info("Epoch %s loss %.6f", epoch + 1, epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state": model.state_dict(), "config": config.to_dict(), "best_loss": best_loss},
                config.model_path,
            )
            logger.info("Saved best checkpoint to %s", config.model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Molecular GTN with NT-Xent.")
    parser.add_argument("--lmdb", dest="lmdb_path", default="/hy-tmp/result/pretraining.lmdb")
    parser.add_argument("--smoke-lmdb", dest="smoke_lmdb_path", default="/hy-tmp/result/smoke_pretraining.lmdb")
    parser.add_argument("--output-dir", default="/hy-tmp/result")
    parser.add_argument("--model-path", default="/hy-tmp/result/model_final.pth")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--mask-ratio", type=float, default=0.1)
    parser.add_argument("--num-masked-views", type=int, default=6)
    parser.add_argument("--smoke-num-masked-views", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--lap-pe-dim", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(PipelineConfig(), **vars(args))
    run_training(config)


if __name__ == "__main__":
    main()
