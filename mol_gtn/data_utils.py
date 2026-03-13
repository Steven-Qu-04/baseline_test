from __future__ import annotations

import math
from typing import Iterator

import torch
from torch.utils.data import Sampler


class DynamicBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        node_counts: list[int],
        max_nodes: int,
        shuffle: bool = True,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ) -> None:
        if max_nodes <= 0:
            raise ValueError("max_nodes must be positive for DynamicBatchSampler")
        self.node_counts = node_counts
        self.max_nodes = max_nodes
        self.shuffle = shuffle
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _rank_indices(self) -> list[int]:
        total = len(self.node_counts)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            order = torch.randperm(total, generator=generator).tolist()
        else:
            order = list(range(total))

        if self.distributed and self.world_size > 1:
            return order[self.rank :: self.world_size]
        return order

    def __iter__(self) -> Iterator[list[int]]:
        current_batch: list[int] = []
        current_nodes = 0
        for index in self._rank_indices():
            node_count = max(1, int(self.node_counts[index]))
            if node_count > self.max_nodes:
                if current_batch:
                    yield current_batch
                    current_batch = []
                    current_nodes = 0
                yield [index]
                continue

            if current_batch and current_nodes + node_count > self.max_nodes:
                yield current_batch
                current_batch = [index]
                current_nodes = node_count
            else:
                current_batch.append(index)
                current_nodes += node_count

        if current_batch:
            yield current_batch

    def __len__(self) -> int:
        indices = self._rank_indices()
        if not indices:
            return 0

        batch_count = 0
        current_nodes = 0
        for index in indices:
            node_count = max(1, int(self.node_counts[index]))
            if node_count > self.max_nodes:
                if current_nodes > 0:
                    batch_count += 1
                    current_nodes = 0
                batch_count += 1
                continue
            if current_nodes > 0 and current_nodes + node_count > self.max_nodes:
                batch_count += 1
                current_nodes = node_count
            else:
                current_nodes += node_count
        if current_nodes > 0:
            batch_count += 1
        return batch_count
