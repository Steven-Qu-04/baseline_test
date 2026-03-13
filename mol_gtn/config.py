from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PipelineConfig:
    csv_path: str = "pretraining.csv"
    output_dir: str = "/hy-tmp/result"
    lmdb_path: str = "/hy-tmp/result/pretraining.lmdb"
    smoke_lmdb_path: str = "/hy-tmp/result/smoke_pretraining.lmdb"
    medium_lmdb_path: str = "/hy-tmp/result/ddp_medium_50k.lmdb"
    model_path: str = "/hy-tmp/result/model_final.pth"
    embeddings_path: str = "/hy-tmp/result/top10_embeddings.csv"
    log_path: str = "/hy-tmp/result/project.log"
    autotune_result_path: str = "/hy-tmp/result/autotune_result.json"
    platform_summary_path: str = "/hy-tmp/result/platform_summary.json"
    smoke_test: bool = False
    medium_test: bool = False
    smoke_rows: int = 100
    medium_rows: int = 50000
    hidden_dim: int = 256
    num_layers: int = 6
    lap_pe_dim: int = 8
    num_heads: int = 8
    dropout: float = 0.1
    projection_dim: int = 128
    temperature: float = 0.07
    batch_size: int = 8
    max_nodes_per_batch: int = 0
    grad_accum_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 3
    mask_ratio: float = 0.1
    num_masked_views: int = 6
    smoke_num_masked_views: int = 2
    num_infer: int = 10
    seed: int = 42
    device: str = "cuda"
    log_interval: int = 10
    cpu_reserve_threads: int = 4
    preprocess_worker_count: int = 0
    task_queue_maxsize: int = 0
    result_queue_maxsize: int = 0
    writer_batch_size: int = 64
    ddp: bool = False
    ddp_backend: str = "nccl"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    stress_steps: int = 1000
    max_steps: int = 0
    skip_save: bool = False
    resume_from: str = ""
    run_timestamp: str = ""
    data_mode: str = "auto"
    loss_log_every_steps: int = 10
    loss_flush_every_steps: int = 100
    autotune_enabled: bool = False
    autotune_batch_min: int = 2
    autotune_batch_max: int = 64
    autotune_steps: int = 24
    autotune_warmup_steps: int = 4
    baseline_batch_size: int = 8
    baseline_grad_accum_steps: int = 4
    scaled_learning_rate: float = 0.0
    min_free_vram_gb: float = 20.0
    master_port_min: int = 20000
    master_port_max: int = 65000

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir)

    def active_lmdb_path(self) -> str:
        if self.smoke_test:
            return self.smoke_lmdb_path
        if self.medium_test:
            return self.medium_lmdb_path
        return self.lmdb_path

    def active_row_limit(self) -> Optional[int]:
        if self.smoke_test:
            return self.smoke_rows
        if self.medium_test:
            return self.medium_rows
        return None

    def active_num_masked_views(self) -> int:
        return self.smoke_num_masked_views if self.smoke_test else self.num_masked_views

    def detected_preprocess_worker_count(self) -> int:
        if self.preprocess_worker_count > 0:
            return self.preprocess_worker_count
        logical_cpus = os.cpu_count() or 1
        return max(1, logical_cpus - self.cpu_reserve_threads)

    def resolved_task_queue_maxsize(self) -> int:
        return self.task_queue_maxsize if self.task_queue_maxsize > 0 else self.detected_preprocess_worker_count() * 2

    def resolved_result_queue_maxsize(self) -> int:
        return self.result_queue_maxsize if self.result_queue_maxsize > 0 else self.detected_preprocess_worker_count() * 2

    def baseline_global_batch(self) -> int:
        return self.baseline_batch_size * self.baseline_grad_accum_steps

    def effective_learning_rate(self) -> float:
        return self.scaled_learning_rate if self.scaled_learning_rate > 0 else self.learning_rate

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
