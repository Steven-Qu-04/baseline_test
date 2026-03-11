from __future__ import annotations

import torch
import torch.distributed as dist
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


def _gather_with_local_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.detach())
    gathered[rank] = tensor
    return torch.cat(gathered, dim=0)


def distributed_nt_xent_loss(anchor_z: torch.Tensor, positive_z: torch.Tensor, temperature: float) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return nt_xent_loss(anchor_z, positive_z, temperature)

    anchor_z = F.normalize(anchor_z, dim=-1)
    positive_z = F.normalize(positive_z, dim=-1)
    global_anchor = _gather_with_local_grad(anchor_z)
    global_positive = _gather_with_local_grad(positive_z)
    global_repr = torch.cat([global_anchor, global_positive], dim=0)

    local_batch_size = anchor_z.size(0)
    global_batch_size = global_anchor.size(0)
    rank = dist.get_rank()
    start = rank * local_batch_size
    local_indices = torch.arange(local_batch_size, device=anchor_z.device)

    anchor_logits = torch.matmul(anchor_z, global_repr.T) / temperature
    positive_logits = torch.matmul(positive_z, global_repr.T) / temperature
    anchor_logits[local_indices, start + local_indices] = float("-inf")
    positive_logits[local_indices, global_batch_size + start + local_indices] = float("-inf")

    anchor_targets = global_batch_size + start + local_indices
    positive_targets = start + local_indices
    anchor_loss = F.cross_entropy(anchor_logits, anchor_targets)
    positive_loss = F.cross_entropy(positive_logits, positive_targets)
    return 0.5 * (anchor_loss + positive_loss)
