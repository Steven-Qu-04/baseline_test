from __future__ import annotations

import math
from typing import List

import torch
from torch_geometric.data import Data


def _resolve_generator(generator: torch.Generator | None, device: torch.device) -> torch.Generator:
    if generator is not None:
        return generator
    resolved = torch.Generator(device=device.type if device.type in {"cpu", "cuda"} else "cpu")
    resolved.manual_seed(torch.seed())
    return resolved


def _choose_indices(num_rows: int, ratio: float, generator: torch.Generator | None, device: torch.device) -> torch.Tensor:
    if num_rows <= 0:
        return torch.empty(0, dtype=torch.long)
    num_mask = min(num_rows, max(1, math.floor(num_rows * ratio)))
    if num_mask <= 0:
        return torch.empty(0, dtype=torch.long)
    resolved = _resolve_generator(generator, device)
    return torch.randperm(num_rows, generator=resolved, device=device)[:num_mask].cpu()


def _mask_tensor_rows(tensor: torch.Tensor, ratio: float, generator: torch.Generator | None = None) -> torch.Tensor:
    if tensor.numel() == 0 or tensor.size(0) == 0:
        return tensor
    clone = tensor.clone()
    indices = _choose_indices(clone.size(0), ratio, generator, clone.device)
    clone[indices] = 0
    return clone


def _drop_edges(data: Data, ratio: float, generator: torch.Generator | None = None) -> Data:
    if data.edge_index.numel() == 0 or data.edge_index.size(1) == 0:
        return data
    clone = data.clone()
    edge_count = clone.edge_index.size(1)
    drop_indices = _choose_indices(edge_count, ratio, generator, clone.edge_index.device)
    keep_mask = torch.ones(edge_count, dtype=torch.bool, device=clone.edge_index.device)
    keep_mask[drop_indices] = False
    clone.edge_index = clone.edge_index[:, keep_mask]
    if clone.edge_attr is not None and clone.edge_attr.size(0) == edge_count:
        clone.edge_attr = clone.edge_attr[keep_mask]
    return clone


def build_anchor_view(data: Data, mask_ratio: float = 0.05, generator: torch.Generator | None = None) -> Data:
    anchor = data.clone()
    anchor.x = _mask_tensor_rows(anchor.x, mask_ratio, generator=generator)
    return anchor


def _mixed_view(
    data: Data,
    atom_ratio: float,
    bond_ratio: float,
    generator: torch.Generator | None = None,
) -> Data:
    mixed = data.clone()
    mixed.x = _mask_tensor_rows(mixed.x, atom_ratio, generator=generator)
    mixed = _drop_edges(mixed, bond_ratio, generator=generator)
    return mixed


def build_diverse_views(
    data: Data,
    num_views: int = 6,
    total_mask_ratio: float = 0.2,
    generator: torch.Generator | None = None,
    generators: list[torch.Generator | None] | None = None,
) -> List[Data]:
    templates = [
        (total_mask_ratio, 0.0),
        (total_mask_ratio, 0.0),
        (0.0, total_mask_ratio),
        (0.0, total_mask_ratio),
        (total_mask_ratio * 0.7, total_mask_ratio * 0.3),
        (total_mask_ratio * 0.5, total_mask_ratio * 0.5),
    ]
    resolved_count = max(1, min(num_views, len(templates)))
    views: list[Data] = []
    for idx in range(resolved_count):
        view_generator = generators[idx] if generators is not None and idx < len(generators) else generator
        views.append(
            _mixed_view(
                data,
                atom_ratio=templates[idx][0],
                bond_ratio=templates[idx][1],
                generator=view_generator,
            )
        )
    return views
