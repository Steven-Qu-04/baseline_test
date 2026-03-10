from __future__ import annotations

import math
import random
from typing import List

import torch
from torch_geometric.data import Data


def _mask_tensor_rows(tensor: torch.Tensor, ratio: float) -> torch.Tensor:
    if tensor.numel() == 0 or tensor.size(0) == 0:
        return tensor
    clone = tensor.clone()
    num_rows = clone.size(0)
    num_mask = max(1, math.floor(num_rows * ratio))
    indices = random.sample(range(num_rows), min(num_rows, num_mask))
    clone[indices] = 0
    return clone


def build_masked_views(data: Data, mask_ratio: float, num_views: int) -> List[Data]:
    views = []
    for _ in range(num_views):
        masked = data.clone()
        masked.x = _mask_tensor_rows(masked.x, mask_ratio)
        masked.edge_attr = _mask_tensor_rows(masked.edge_attr, mask_ratio)
        views.append(masked)
    return views
