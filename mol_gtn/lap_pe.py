from __future__ import annotations

import numpy as np
import torch


def compute_laplacian_positional_encoding(num_nodes: int, edge_index: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    if num_nodes == 0:
        raise ValueError("Graph has no nodes")
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for src, dst in edge_index.t().tolist():
        adjacency[src, dst] = 1.0
    degrees = adjacency.sum(axis=1)
    laplacian = np.diag(degrees) - adjacency
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    order = np.argsort(eigvals)
    eigvecs = eigvecs[:, order]
    usable = eigvecs[:, 1 : min(num_nodes, k + 1)] if num_nodes > 1 else np.zeros((num_nodes, 0), dtype=np.float32)
    pe = np.zeros((num_nodes, k), dtype=np.float32)
    valid_dims = usable.shape[1]
    if valid_dims > 0:
        pe[:, :valid_dims] = usable[:, :valid_dims]
    lap_pe = torch.tensor(pe, dtype=torch.float32)
    lap_pe_valid_mask = torch.zeros(k, dtype=torch.bool)
    if valid_dims > 0:
        lap_pe_valid_mask[:valid_dims] = True
    return lap_pe, lap_pe_valid_mask
