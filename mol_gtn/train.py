from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .config import PipelineConfig
from .dataset import LmdbGraphDataset, contrastive_collate
from .features import feature_dimensions
from .losses import distributed_nt_xent_loss, nt_xent_loss
from .model import MolecularGTN
from .utils.logging import configure_logging
from .utils.runtime import seed_everything


def ddp_is_enabled(config: PipelineConfig) -> bool:
    return config.ddp or int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed(config: PipelineConfig) -> tuple[bool, int, int, int, torch.device]:
    enabled = ddp_is_enabled(config)
    if enabled:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend=config.ddp_backend)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return True, rank, local_rank, world_size, device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    return False, 0, 0, 1, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def rank_zero_only(rank: int) -> bool:
    return rank == 0


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= world_size
    return value


def gather_min_value(value: float, device: torch.device, world_size: int) -> float:
    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return float(tensor.item())


def build_loader(dataset: LmdbGraphDataset, config: PipelineConfig, distributed: bool) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        drop_last=distributed,
        collate_fn=lambda batch: contrastive_collate(batch, config),
    )
    return loader, sampler


def build_model(config: PipelineConfig, device: torch.device, distributed: bool) -> torch.nn.Module:
    dims = feature_dimensions()
    model = MolecularGTN(dims["atom_dim"], dims["bond_dim"], config).to(device)
    if distributed:
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)
    return model


def save_json(path: str, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))


def run_training_loop(
    config: PipelineConfig,
    model: torch.nn.Module,
    loader: DataLoader,
    sampler: DistributedSampler | None,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict:
    logger = configure_logging(config.log_path)
    best_loss = float("inf")
    total_steps = 0
    effective_lr = config.effective_learning_rate()
    for group in optimizer.param_groups:
        group["lr"] = effective_lr

    for epoch in range(config.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        local_steps = 0
        progress = tqdm(loader, desc=f"train-epoch-{epoch + 1}", disable=not rank_zero_only(rank))
        for step, batch in enumerate(progress, start=1):
            anchors = batch["anchors"].to(device)
            positives = batch["positives"].to(device)
            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                anchor_z = model(anchors)
                positive_z = model(positives)
                raw_loss = distributed_nt_xent_loss(anchor_z, positive_z, config.temperature) if world_size > 1 else nt_xent_loss(anchor_z, positive_z, config.temperature)
                loss = raw_loss / config.grad_accum_steps
            scaler.scale(loss).backward()
            if step % config.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(raw_loss.detach().item())
            local_steps += 1
            total_steps += 1
            if rank_zero_only(rank):
                progress.set_postfix(
                    loss=f"{running_loss / local_steps:.4f}",
                    batch_per_gpu=config.batch_size,
                    grad_accum=config.grad_accum_steps,
                )
            if config.max_steps > 0 and total_steps >= config.max_steps:
                break
        if local_steps % config.grad_accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        epoch_loss = torch.tensor(running_loss / max(1, local_steps), device=device)
        epoch_loss = reduce_mean(epoch_loss, world_size)
        if rank_zero_only(rank):
            logger.info(
                "Epoch %s loss %.6f batch_per_gpu=%s grad_accum_steps=%s lr=%.8f",
                epoch + 1,
                epoch_loss.item(),
                config.batch_size,
                config.grad_accum_steps,
                effective_lr,
            )
        if rank_zero_only(rank) and not config.skip_save and epoch_loss.item() < best_loss:
            best_loss = epoch_loss.item()
            state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": state_dict, "config": config.to_dict(), "best_loss": best_loss}, config.model_path)
            logger.info("Saved best checkpoint to %s", config.model_path)
        if config.max_steps > 0 and total_steps >= config.max_steps:
            break
    return {"best_loss": best_loss, "total_steps": total_steps}


def run_autotune(config: PipelineConfig, device: torch.device, rank: int, world_size: int, distributed: bool) -> None:
    logger = configure_logging(config.log_path)
    dataset = LmdbGraphDataset(config.active_lmdb_path(), config)
    if len(dataset) == 0:
        raise RuntimeError(f"No serialized graphs found in {config.active_lmdb_path()}")

    candidates = []
    batch = max(1, config.autotune_batch_min)
    while batch <= config.autotune_batch_max:
        candidates.append(batch)
        batch *= 2
    if candidates[-1] != config.autotune_batch_max:
        candidates.append(config.autotune_batch_max)

    best_trial: dict | None = None
    baseline_global_batch = config.baseline_global_batch()
    logger.info("Starting autotune on %s candidates=%s baseline_global_batch=%s", config.active_lmdb_path(), candidates, baseline_global_batch)

    for candidate in candidates:
        trial_config = replace(config, batch_size=candidate, grad_accum_steps=1, max_steps=config.autotune_steps, skip_save=True)
        loader, sampler = build_loader(dataset, trial_config, distributed)
        model = build_model(trial_config, device, distributed)
        optimizer = optim.AdamW(model.parameters(), lr=trial_config.learning_rate, weight_decay=trial_config.weight_decay)
        scaler = GradScaler("cuda", enabled=device.type == "cuda")
        torch.cuda.empty_cache()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        success = True
        elapsed = 0.0
        measured_steps = 0
        try:
            if sampler is not None:
                sampler.set_epoch(0)
            optimizer.zero_grad(set_to_none=True)
            start_time = None
            for step, batch in enumerate(loader, start=1):
                anchors = batch["anchors"].to(device)
                positives = batch["positives"].to(device)
                with autocast(device_type="cuda", enabled=device.type == "cuda"):
                    anchor_z = model(anchors)
                    positive_z = model(positives)
                    raw_loss = distributed_nt_xent_loss(anchor_z, positive_z, trial_config.temperature) if world_size > 1 else nt_xent_loss(anchor_z, positive_z, trial_config.temperature)
                scaler.scale(raw_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if step == config.autotune_warmup_steps:
                    start_time = time.time()
                if step > config.autotune_warmup_steps and start_time is not None:
                    measured_steps += 1
                if step >= trial_config.max_steps:
                    break
            if start_time is not None:
                elapsed = time.time() - start_time
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                success = False
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                logger.warning("Autotune candidate batch=%s hit OOM on rank=%s", candidate, rank)
            else:
                raise

        success_tensor = torch.tensor(1 if success else 0, device=device, dtype=torch.int32)
        if distributed:
            dist.all_reduce(success_tensor, op=dist.ReduceOp.MIN)
        success = bool(success_tensor.item())
        peak_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024**3) if device.type == "cuda" else 0.0
        total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3) if device.type == "cuda" else 0.0
        free_vram_gb = total_vram_gb - peak_reserved_gb
        min_free_vram_gb = gather_min_value(free_vram_gb, device, world_size)
        if success and measured_steps > 0 and elapsed > 0:
            trial = {
                "batch_per_gpu": candidate,
                "grad_accum_steps": 1,
                "steps_per_second": measured_steps / elapsed,
                "peak_reserved_gb": peak_reserved_gb,
                "min_free_vram_gb": min_free_vram_gb,
                "global_batch": candidate * world_size,
            }
            if best_trial is None or candidate > best_trial["batch_per_gpu"]:
                best_trial = trial
        if distributed:
            dist.barrier()

    if best_trial is None:
        raise RuntimeError("Autotune failed to find any stable batch size")

    chosen_grad_accum = 1
    rationale = "selected largest stable per-GPU batch with grad_accum_steps=1"
    if best_trial["global_batch"] < baseline_global_batch:
        chosen_grad_accum = min(config.grad_accum_steps, max(1, math.ceil(baseline_global_batch / best_trial["global_batch"])))
        rationale = "raised grad_accum_steps minimally to recover baseline global batch"
    elif best_trial["min_free_vram_gb"] > config.min_free_vram_gb:
        chosen_grad_accum = 1
        rationale = "kept grad_accum_steps=1 because global batch was already sufficient while substantial VRAM remained free"

    scaled_lr = config.learning_rate * ((best_trial["batch_per_gpu"] * world_size * chosen_grad_accum) / baseline_global_batch)
    result = {
        "batch_per_gpu": best_trial["batch_per_gpu"],
        "grad_accum_steps": chosen_grad_accum,
        "global_batch": best_trial["batch_per_gpu"] * world_size * chosen_grad_accum,
        "scaled_learning_rate": scaled_lr,
        "steps_per_second": best_trial["steps_per_second"],
        "peak_reserved_gb": best_trial["peak_reserved_gb"],
        "min_free_vram_gb": best_trial["min_free_vram_gb"],
        "baseline_global_batch": baseline_global_batch,
        "rationale": rationale,
    }
    if rank_zero_only(rank):
        save_json(config.autotune_result_path, result)
        logger.info(
            "Autotune result batch_per_gpu=%s grad_accum_steps=%s scaled_lr=%.8f rationale=%s",
            result["batch_per_gpu"],
            result["grad_accum_steps"],
            result["scaled_learning_rate"],
            rationale,
        )


def run_training(config: PipelineConfig) -> None:
    distributed, rank, local_rank, world_size, device = setup_distributed(config)
    try:
        seed_everything(config.seed + rank)
        logger = configure_logging(config.log_path)
        logger.info(
            "Training start ddp=%s rank=%s local_rank=%s world_size=%s lmdb=%s",
            distributed,
            rank,
            local_rank,
            world_size,
            config.active_lmdb_path(),
        )
        if config.autotune_enabled:
            run_autotune(config, device, rank, world_size, distributed)
            return

        dataset = LmdbGraphDataset(config.active_lmdb_path(), config)
        if len(dataset) == 0:
            raise RuntimeError(f"No serialized graphs found in {config.active_lmdb_path()}")
        loader, sampler = build_loader(dataset, config, distributed)
        model = build_model(config, device, distributed)
        optimizer = optim.AdamW(model.parameters(), lr=config.effective_learning_rate(), weight_decay=config.weight_decay)
        scaler = GradScaler("cuda", enabled=device.type == "cuda")
        run_training_loop(config, model, loader, sampler, optimizer, scaler, device, rank, world_size)
    finally:
        cleanup_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Molecular GTN with NT-Xent.")
    parser.add_argument("--lmdb", dest="lmdb_path", default="/hy-tmp/result/pretraining.lmdb")
    parser.add_argument("--smoke-lmdb", dest="smoke_lmdb_path", default="/hy-tmp/result/smoke_pretraining.lmdb")
    parser.add_argument("--medium-lmdb", dest="medium_lmdb_path", default="/hy-tmp/result/ddp_medium_50k.lmdb")
    parser.add_argument("--output-dir", default="/hy-tmp/result")
    parser.add_argument("--model-path", default="/hy-tmp/result/model_final.pth")
    parser.add_argument("--autotune-result-path", default="/hy-tmp/result/autotune_result.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-per-gpu", dest="batch_size", type=int)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--scaled-learning-rate", type=float, default=0.0)
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
    parser.add_argument("--medium-test", action="store_true")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--backend", dest="ddp_backend", default="nccl")
    parser.add_argument("--stress-steps", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--skip-save", action="store_true")
    parser.add_argument("--autotune", dest="autotune_enabled", action="store_true")
    parser.add_argument("--autotune-batch-min", type=int, default=2)
    parser.add_argument("--autotune-batch-max", type=int, default=64)
    parser.add_argument("--autotune-steps", type=int, default=24)
    parser.add_argument("--autotune-warmup-steps", type=int, default=4)
    parser.add_argument("--baseline-batch-size", type=int, default=8)
    parser.add_argument("--baseline-grad-accum-steps", type=int, default=4)
    parser.add_argument("--min-free-vram-gb", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(PipelineConfig(), **vars(args))
    run_training(config)


if __name__ == "__main__":
    main()
