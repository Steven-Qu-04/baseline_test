from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class PipelineConfig:
    csv_path: str = "pretraining.csv"
    output_dir: str = "/hy-tmp/result"
    lmdb_path: str = "/hy-tmp/result/pretraining.lmdb"
    smoke_lmdb_path: str = "/hy-tmp/result/smoke_pretraining.lmdb"
    model_path: str = "/hy-tmp/result/model_final.pth"
    embeddings_path: str = "/hy-tmp/result/top10_embeddings.csv"
    log_path: str = "/hy-tmp/result/project.log"
    smoke_test: bool = False
    smoke_rows: int = 100
    hidden_dim: int = 256
    num_layers: int = 6
    lap_pe_dim: int = 8
    num_heads: int = 8
    dropout: float = 0.1
    projection_dim: int = 128
    temperature: float = 0.07
    batch_size: int = 8
    grad_accum_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 3
    mask_ratio: float = 0.1
    num_masked_views: int = 6
    smoke_num_masked_views: int = 2
    num_workers: int = 8
    queue_size: int = 256
    writer_batch_size: int = 64
    num_infer: int = 10
    seed: int = 42
    device: str = "cuda"
    log_interval: int = 10

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir)

    def active_lmdb_path(self) -> str:
        return self.smoke_lmdb_path if self.smoke_test else self.lmdb_path

    def active_num_masked_views(self) -> int:
        return self.smoke_num_masked_views if self.smoke_test else self.num_masked_views

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
