from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(anchor_z: torch.Tensor, positive_z: torch.Tensor, temperature: float) -> torch.Tensor:
    representations = torch.cat([anchor_z, positive_z], dim=0)
    representations = F.normalize(representations, dim=-1)
    similarity = torch.matmul(representations, representations.T) / temperature
    batch_size = anchor_z.size(0)
    mask = torch.eye(2 * batch_size, device=similarity.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, float("-inf"))
    targets = torch.arange(batch_size, 2 * batch_size, device=similarity.device)
    targets = torch.cat([targets, torch.arange(0, batch_size, device=similarity.device)])
    return F.cross_entropy(similarity, targets)
