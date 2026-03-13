from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from .augment import build_anchor_view, build_diverse_views
from .config import PipelineConfig
from .lmdb_io import deserialize_data, open_lmdb


class LmdbGraphDataset(Dataset):
    def __init__(self, lmdb_path: str, config: PipelineConfig):
        self.lmdb_path = lmdb_path
        self.config = config
        self.env = None
        self.length = None
        self._node_counts: list[int] | None = None
        self._epoch: int = 0

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

    def node_counts(self) -> list[int]:
        self._ensure_env()
        if self._node_counts is not None:
            return self._node_counts

        cache_path = Path(f"{self.lmdb_path}.node_counts.pt")
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            cached_counts = payload.get("node_counts")
            cached_length = int(payload.get("length", -1))
            if isinstance(cached_counts, list) and cached_length == len(self):
                self._node_counts = [int(value) for value in cached_counts]
                return self._node_counts

        counts: list[int] = []
        with self.env.begin() as txn:
            progress = tqdm(
                range(len(self)),
                desc="scan-node-counts",
                leave=False,
            )
            for index in progress:
                blob = txn.get(f"{index:012d}".encode())
                if blob is None:
                    raise IndexError(index)
                data = deserialize_data(blob)
                counts.append(int(data.x.size(0)))

        self._node_counts = counts
        try:
            torch.save({"length": len(counts), "node_counts": counts}, cache_path)
        except OSError:
            pass
        return self._node_counts

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)


def _stable_token(value: str) -> int:
    token = 0
    for char in value:
        token = (token * 131 + ord(char)) % 2_147_483_647
    return token


def _build_generator(config: PipelineConfig, dataset: Any, sample: Data, view_id: int) -> torch.Generator:
    molecule_id = str(getattr(sample, "molecule_id", "unknown"))
    base = _stable_token(molecule_id)
    seed = (
        int(config.seed)
        + int(getattr(config, "rank", 0)) * 1_000_003
        + int(getattr(dataset, "_epoch", 0)) * 97_003
        + base
        + int(view_id) * 19_097
    ) % (2**31 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def contrastive_collate(samples: list[Data], config: PipelineConfig, dataset: LmdbGraphDataset | None = None) -> dict[str, Any]:
    anchors = []
    positives = []
    molecule_ids = []
    node_counts = []
    edge_counts = []
    for sample in samples:
        node_counts.append(int(sample.x.size(0)))
        edge_counts.append(int(sample.edge_index.size(1)))
        epoch_holder = dataset if dataset is not None else SimpleNamespace(_epoch=0)
        anchor_generator = _build_generator(config, epoch_holder, sample, view_id=0)
        use_offline = config.data_mode == "offline" or (
            config.data_mode == "auto" and hasattr(sample, "offline_anchor") and hasattr(sample, "offline_positives")
        )
        if use_offline and hasattr(sample, "offline_anchor") and hasattr(sample, "offline_positives"):
            anchor = sample.offline_anchor.clone()
            masked_views = [view.clone() for view in list(sample.offline_positives)[: config.active_num_masked_views()]]
        else:
            view_generators = [
                _build_generator(config, epoch_holder, sample, view_id=view_index + 1)
                for view_index in range(config.active_num_masked_views())
            ]
            anchor = build_anchor_view(sample, mask_ratio=0.05, generator=anchor_generator)
            masked_views = build_diverse_views(
                sample,
                num_views=config.active_num_masked_views(),
                total_mask_ratio=0.2,
                generators=view_generators,
            )
        for masked in masked_views:
            anchors.append(anchor.clone())
            positives.append(masked.clone())
            molecule_ids.append(str(getattr(sample, "molecule_id", "unknown")))
    return {
        "anchors": Batch.from_data_list(anchors),
        "positives": Batch.from_data_list(positives),
        "molecule_ids": molecule_ids,
        "num_graphs": len(samples),
        "total_nodes": sum(node_counts),
        "max_nodes": max(node_counts) if node_counts else 0,
        "total_edges": sum(edge_counts),
    }
