from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace
from datetime import datetime
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
from .data_utils import DynamicBatchSampler
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


def gather_min_int(value: int, device: torch.device, world_size: int) -> int:
    tensor = torch.tensor(value, dtype=torch.int64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return int(tensor.item())


def build_loader(
    dataset: LmdbGraphDataset,
    config: PipelineConfig,
    distributed: bool,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DynamicBatchSampler | DistributedSampler | None]:
    sampler = None
    if config.max_nodes_per_batch > 0:
        sampler = DynamicBatchSampler(
            node_counts=dataset.node_counts(),
            max_nodes=config.max_nodes_per_batch,
            shuffle=True,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            seed=config.seed,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=lambda batch: contrastive_collate(batch, config, dataset),
        )
        return loader, sampler

    fallback_sampler = None
    if distributed:
        fallback_sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=fallback_sampler is None,
        sampler=fallback_sampler,
        num_workers=0,
        drop_last=distributed,
        collate_fn=lambda batch: contrastive_collate(batch, config, dataset),
    )
    return loader, fallback_sampler


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


class BufferedLossLogger:
    def __init__(self, path: str, collect_every: int, flush_every: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.collect_every = max(1, int(collect_every))
        self.flush_every = max(1, int(flush_every))
        self.buffer: list[dict] = []

    def maybe_collect(self, epoch: int, step: int, loss: float, learning_rate: float) -> None:
        if step % self.collect_every != 0:
            return
        self.buffer.append(
            {
                "epoch": int(epoch),
                "step": int(step),
                "loss": float(loss),
                "learning_rate": float(learning_rate),
            }
        )
        if step % self.flush_every == 0:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            for row in self.buffer:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        self.buffer.clear()


def _resolve_run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _latest_checkpoint(root_output_dir: str) -> Path | None:
    root = Path(root_output_dir)
    if not root.exists():
        return None
    if root.name.startswith("run_"):
        epoch_pattern = "**/*epoch_*.pth"
        best_pattern = "**/*model_best*.pth"
    else:
        epoch_pattern = "run_*/**/*epoch_*.pth"
        best_pattern = "run_*/**/*model_best*.pth"
    candidates = sorted(root.glob(epoch_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    best_candidates = sorted(root.glob(best_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return best_candidates[0] if best_candidates else None


def run_training_loop(
    config: PipelineConfig,
    model: torch.nn.Module,
    loader: DataLoader,
    sampler: DynamicBatchSampler | DistributedSampler | None,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    rank: int,
    world_size: int,
    run_timestamp: str,
    start_epoch: int = 0,
    start_global_step: int = 0,
    start_best_loss: float = float("inf"),
    loss_logger: BufferedLossLogger | None = None,
) -> dict:
    logger = configure_logging(config.log_path)
    best_loss = float(start_best_loss)
    total_steps = int(start_global_step)
    effective_lr = config.effective_learning_rate()
    for group in optimizer.param_groups:
        group["lr"] = effective_lr

    for epoch in range(start_epoch, config.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        if hasattr(loader.dataset, "set_epoch"):
            loader.dataset.set_epoch(epoch)
        max_epoch_steps = len(loader)
        if world_size > 1:
            max_epoch_steps = gather_min_int(max_epoch_steps, device, world_size)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        local_steps = 0
        progress = tqdm(total=max_epoch_steps, desc=f"train-epoch-{epoch + 1}", disable=not rank_zero_only(rank))
        data_iter = iter(loader)
        step = 0
        while True:
            try:
                batch = next(data_iter)
                has_batch = 1
            except StopIteration:
                batch = None
                has_batch = 0
            if world_size > 1:
                has_batch_tensor = torch.tensor(has_batch, device=device, dtype=torch.int32)
                dist.all_reduce(has_batch_tensor, op=dist.ReduceOp.MIN)
                if int(has_batch_tensor.item()) == 0:
                    break
            elif has_batch == 0:
                break
            if batch is None:
                break
            step += 1
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
                progress.update(1)
                progress.set_postfix(
                    loss=f"{running_loss / local_steps:.4f}",
                    graphs=batch.get("num_graphs", config.batch_size),
                    total_nodes=batch.get("total_nodes", 0),
                    max_nodes=batch.get("max_nodes", 0),
                    grad_accum=config.grad_accum_steps,
                )
                if loss_logger is not None:
                    loss_logger.maybe_collect(
                        epoch=epoch + 1,
                        step=total_steps,
                        loss=float(raw_loss.detach().item()),
                        learning_rate=effective_lr,
                    )
            if config.max_steps > 0 and total_steps >= config.max_steps:
                break
            if step >= max_epoch_steps:
                break
        progress.close()
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
            if loss_logger is not None:
                loss_logger.flush()
        if rank_zero_only(rank) and not config.skip_save:
            state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            checkpoint = {
                "model_state": state_dict,
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "config": config.to_dict(),
                "run_timestamp": run_timestamp,
                "epoch": epoch + 1,
                "global_step": total_steps,
                "epoch_loss": epoch_loss.item(),
                "best_loss": best_loss,
            }
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
            epoch_checkpoint_path = Path(config.output_dir) / f"{run_timestamp}_epoch_{epoch + 1}.pth"
            torch.save(checkpoint, str(epoch_checkpoint_path))
            logger.info("Saved epoch checkpoint to %s", epoch_checkpoint_path)
            if epoch_loss.item() < best_loss:
                best_loss = epoch_loss.item()
                checkpoint["best_loss"] = best_loss
                torch.save(checkpoint, config.model_path)
                logger.info("Saved best checkpoint to %s", config.model_path)
        if config.max_steps > 0 and total_steps >= config.max_steps:
            break
    if rank_zero_only(rank) and loss_logger is not None:
        loss_logger.flush()
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
        trial_kwargs = {"grad_accum_steps": 1, "max_steps": config.autotune_steps, "skip_save": True}
        if config.max_nodes_per_batch > 0:
            trial_kwargs["max_nodes_per_batch"] = candidate
        else:
            trial_kwargs["batch_size"] = candidate
        trial_config = replace(config, **trial_kwargs)
        loader, sampler = build_loader(dataset, trial_config, distributed, rank, world_size)
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
                "batch_per_gpu": candidate if config.max_nodes_per_batch <= 0 else None,
                "max_nodes_per_batch": candidate if config.max_nodes_per_batch > 0 else trial_config.max_nodes_per_batch,
                "grad_accum_steps": 1,
                "steps_per_second": measured_steps / elapsed,
                "peak_reserved_gb": peak_reserved_gb,
                "min_free_vram_gb": min_free_vram_gb,
                "global_batch": candidate * world_size if config.max_nodes_per_batch <= 0 else None,
            }
            if best_trial is None or candidate > (
                best_trial["max_nodes_per_batch"] if config.max_nodes_per_batch > 0 else best_trial["batch_per_gpu"]
            ):
                best_trial = trial
        if distributed:
            dist.barrier()

    if best_trial is None:
        raise RuntimeError("Autotune failed to find any stable batch size")

    chosen_grad_accum = 1
    rationale = "selected largest stable per-rank packing budget with grad_accum_steps=1"
    if config.max_nodes_per_batch <= 0 and best_trial["global_batch"] < baseline_global_batch:
        chosen_grad_accum = min(config.grad_accum_steps, max(1, math.ceil(baseline_global_batch / best_trial["global_batch"])))
        rationale = "raised grad_accum_steps minimally to recover baseline global batch"
    elif best_trial["min_free_vram_gb"] > config.min_free_vram_gb:
        chosen_grad_accum = 1
        rationale = "kept grad_accum_steps=1 because global batch was already sufficient while substantial VRAM remained free"

    scaled_lr = config.learning_rate
    if config.max_nodes_per_batch <= 0:
        scaled_lr = config.learning_rate * ((best_trial["batch_per_gpu"] * world_size * chosen_grad_accum) / baseline_global_batch)
    result = {
        "batch_per_gpu": best_trial["batch_per_gpu"],
        "max_nodes_per_batch": best_trial["max_nodes_per_batch"],
        "grad_accum_steps": chosen_grad_accum,
        "global_batch": best_trial["batch_per_gpu"] * world_size * chosen_grad_accum if best_trial["batch_per_gpu"] is not None else None,
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
            "Autotune result batch_per_gpu=%s max_nodes_per_batch=%s grad_accum_steps=%s scaled_lr=%.8f rationale=%s",
            result["batch_per_gpu"],
            result["max_nodes_per_batch"],
            result["grad_accum_steps"],
            result["scaled_learning_rate"],
            rationale,
        )


def run_training(config: PipelineConfig) -> None:
    distributed, rank, local_rank, world_size, device = setup_distributed(config)
    try:
        seed_everything(config.seed + rank)
        root_output_dir = Path(config.output_dir)
        resume_from = config.resume_from.strip() if hasattr(config, "resume_from") else ""
        run_timestamp = config.run_timestamp.strip() if hasattr(config, "run_timestamp") else ""
        resume_checkpoint: str | None = None
        if rank_zero_only(rank):
            if resume_from:
                resolved = Path(resume_from)
                if not resolved.exists():
                    raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
                resume_checkpoint = str(resolved.resolve())
                run_timestamp = resolved.parent.name.replace("run_", "")
            elif run_timestamp:
                latest = _latest_checkpoint(str(root_output_dir / f"run_{run_timestamp}"))
                if latest is not None:
                    resume_checkpoint = str(latest.resolve())
            else:
                latest = _latest_checkpoint(str(root_output_dir))
                if latest is not None:
                    resume_checkpoint = str(latest.resolve())
                    run_timestamp = latest.parent.name.replace("run_", "")
            if not run_timestamp:
                run_timestamp = _resolve_run_timestamp()
        if distributed:
            payload = [run_timestamp, resume_checkpoint]
            dist.broadcast_object_list(payload, src=0)
            run_timestamp, resume_checkpoint = payload[0], payload[1]

        run_dir = root_output_dir / f"run_{run_timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = replace(
            config,
            output_dir=str(run_dir),
            model_path=str(run_dir / f"{run_timestamp}_model_best.pth"),
            log_path=str(run_dir / f"{run_timestamp}_project.log"),
            autotune_result_path=str(run_dir / f"{run_timestamp}_autotune_result.json"),
            rank=rank,
            world_size=world_size,
        )

        logger = configure_logging(config.log_path)
        logger.info(
            "Training start ddp=%s rank=%s local_rank=%s world_size=%s lmdb=%s run_dir=%s resume=%s",
            distributed,
            rank,
            local_rank,
            world_size,
            config.active_lmdb_path(),
            run_dir,
            resume_checkpoint if resume_checkpoint else "none",
        )
        if config.autotune_enabled:
            run_autotune(config, device, rank, world_size, distributed)
            return

        dataset = LmdbGraphDataset(config.active_lmdb_path(), config)
        if len(dataset) == 0:
            raise RuntimeError(f"No serialized graphs found in {config.active_lmdb_path()}")
        loader, sampler = build_loader(dataset, config, distributed, rank, world_size)
        model = build_model(config, device, distributed)
        optimizer = optim.AdamW(model.parameters(), lr=config.effective_learning_rate(), weight_decay=config.weight_decay)
        scaler = GradScaler("cuda", enabled=device.type == "cuda")
        start_epoch = 0
        start_global_step = 0
        best_loss = float("inf")
        if resume_checkpoint:
            checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
            target = model.module if isinstance(model, DDP) else model
            target.load_state_dict(checkpoint["model_state"])
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            if "scaler_state" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state"])
            start_epoch = int(checkpoint.get("epoch", 0))
            start_global_step = int(checkpoint.get("global_step", 0))
            best_loss = float(checkpoint.get("best_loss", float("inf")))
            logger.info(
                "Resumed from %s start_epoch=%s global_step=%s best_loss=%.6f",
                resume_checkpoint,
                start_epoch,
                start_global_step,
                best_loss,
            )
        loss_logger = BufferedLossLogger(
            str(run_dir / f"{run_timestamp}_loss_log.jsonl"),
            collect_every=config.loss_log_every_steps,
            flush_every=config.loss_flush_every_steps,
        ) if rank_zero_only(rank) else None
        if rank_zero_only(rank):
            save_json(
                str(run_dir / f"{run_timestamp}_run_manifest.json"),
                {
                    "run_timestamp": run_timestamp,
                    "resume_checkpoint": resume_checkpoint,
                    "config": config.to_dict(),
                },
            )
        if rank_zero_only(rank):
            logger.info(
                "Loader mode=%s max_nodes_per_batch=%s fallback_batch_size=%s dataset_size=%s",
                "dynamic" if config.max_nodes_per_batch > 0 else "fixed",
                config.max_nodes_per_batch,
                config.batch_size,
                len(dataset),
            )
        run_training_loop(
            config,
            model,
            loader,
            sampler,
            optimizer,
            scaler,
            device,
            rank,
            world_size,
            run_timestamp=run_timestamp,
            start_epoch=start_epoch,
            start_global_step=start_global_step,
            start_best_loss=best_loss,
            loss_logger=loss_logger,
        )
    finally:
        cleanup_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Molecular GTN with NT-Xent.")
    parser.add_argument("--lmdb", dest="lmdb_path", default="/hy-tmp/result/pretraining.lmdb")
    parser.add_argument("--smoke-lmdb", dest="smoke_lmdb_path", default="/hy-tmp/result/smoke_pretraining.lmdb")
    parser.add_argument("--medium-lmdb", dest="medium_lmdb_path", default="/hy-tmp/result/ddp_medium_50k.lmdb")
    parser.add_argument("--output-dir", default="/hy-tmp/result")
    parser.add_argument("--model-path", default="/hy-tmp/result/model_final.pth")
    parser.add_argument("--log-path", default="/hy-tmp/result/project.log")
    parser.add_argument("--autotune-result-path", default="/hy-tmp/result/autotune_result.json")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--run-timestamp", default="")
    parser.add_argument("--data-mode", choices=["auto", "offline", "online"], default="auto")
    parser.add_argument("--loss-log-every-steps", type=int, default=10)
    parser.add_argument("--loss-flush-every-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-per-gpu", dest="batch_size", type=int)
    parser.add_argument("--max-nodes-per-batch", type=int, default=2048)
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
