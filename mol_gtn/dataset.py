from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from .augment import build_masked_views
from .config import PipelineConfig
from .lmdb_io import deserialize_data, open_lmdb


class LmdbGraphDataset(Dataset):
    def __init__(self, lmdb_path: str, config: PipelineConfig):
        self.lmdb_path = lmdb_path
        self.config = config
        self.env = None
        self.length = None

    def _ensure_env(self) -> None:
        if self.env is None:
            self.env = open_lmdb(self.lmdb_path, readonly=True)
            with self.env.begin() as txn:
                self.length = int((txn.get(b"length") or b"0").decode())

    def __len__(self) -> int:
        self._ensure_env()
        return int(self.length or 0)

    def __getitem__(self, index: int) -> Data:
        self._ensure_env()
        with self.env.begin() as txn:
            blob = txn.get(f"{index:012d}".encode())
        if blob is None:
            raise IndexError(index)
        return deserialize_data(blob)


def contrastive_collate(samples: list[Data], config: PipelineConfig) -> dict[str, Any]:
    anchors = []
    positives = []
    molecule_ids = []
    for sample in samples:
        masked_views = build_masked_views(sample, config.mask_ratio, config.active_num_masked_views())
        for masked in masked_views:
            anchors.append(sample.clone())
            positives.append(masked)
            molecule_ids.append(sample.molecule_id)
    return {
        "anchors": Batch.from_data_list(anchors),
        "positives": Batch.from_data_list(positives),
        "molecule_ids": molecule_ids,
    }
