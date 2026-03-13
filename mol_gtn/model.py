from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_adj, to_dense_batch

from .config import PipelineConfig


@dataclass
class EncoderOutput:
    node_embeddings: torch.Tensor
    bond_embeddings: torch.Tensor
    mol_embeddings: torch.Tensor
    node_mask: torch.Tensor


class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, node_mask: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        key_padding_mask = torch.zeros_like(node_mask, dtype=x.dtype)
        key_padding_mask = key_padding_mask.masked_fill(~node_mask, float("-inf"))
        attn_out, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_bias,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = x.masked_fill(~node_mask.unsqueeze(-1), 0.0)
        return x


class GlobalAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        gate = self.gate(x).squeeze(-1)
        gate = gate.masked_fill(~node_mask, float("-inf"))
        weights = torch.softmax(gate, dim=-1).unsqueeze(-1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = weights.masked_fill(~node_mask.unsqueeze(-1), 0.0)
        x = x.masked_fill(~node_mask.unsqueeze(-1), 0.0)
        return (weights * x).sum(dim=1)


class MolecularGTN(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int, config: PipelineConfig):
        super().__init__()
        self.node_proj = nn.Linear(atom_dim + config.lap_pe_dim, config.hidden_dim)
        self.edge_bias_proj = nn.Linear(bond_dim, config.num_heads)
        self.layers = nn.ModuleList(
            GraphTransformerLayer(config.hidden_dim, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        )
        self.pool = GlobalAttentionPooling(config.hidden_dim)
        self.bond_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2 + bond_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.projection_dim),
        )
        self.config = config

    def _build_attention_bias(self, batch: Batch, node_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes = node_mask.shape
        if batch.edge_attr.size(0) == 0:
            edge_bias = torch.zeros(
                batch_size,
                self.config.num_heads,
                num_nodes,
                num_nodes,
                device=node_mask.device,
                dtype=self.node_proj.weight.dtype,
            )
        else:
            dense_edges = to_dense_adj(batch.edge_index, batch=batch.batch, edge_attr=batch.edge_attr)
            if dense_edges.dim() == 3:
                dense_edges = dense_edges.unsqueeze(-1)
            edge_bias = self.edge_bias_proj(dense_edges)
            edge_bias = edge_bias.permute(0, 3, 1, 2).contiguous()
        _, num_heads, _, _ = edge_bias.shape
        invalid_mask = (~node_mask).unsqueeze(1).expand(batch_size, num_heads, num_nodes)
        attn_bias = edge_bias.masked_fill(invalid_mask.unsqueeze(-1), float("-inf"))
        attn_bias = attn_bias.masked_fill(invalid_mask.unsqueeze(-2), float("-inf"))
        attn_bias = attn_bias.reshape(batch_size * num_heads, num_nodes, num_nodes)
        return attn_bias

    def encode(self, batch: Batch, compute_bond_embeddings: bool = True) -> EncoderOutput:
        x = torch.cat([batch.x, batch.lap_pe], dim=-1)
        dense_x, node_mask = to_dense_batch(x, batch.batch)
        projected = self.node_proj(dense_x)
        attn_bias = self._build_attention_bias(batch, node_mask)
        hidden = projected
        for layer in self.layers:
            hidden = layer(hidden, node_mask, attn_bias)
        mol_embeddings = self.pool(hidden, node_mask)
        flat_hidden = hidden[node_mask]
        edge_src = batch.edge_index[0]
        edge_dst = batch.edge_index[1]
        if not compute_bond_embeddings:
            bond_embeddings = batch.edge_attr.new_zeros((0, self.config.hidden_dim))
        elif batch.edge_attr.size(0) == 0:
            bond_embeddings = batch.edge_attr.new_zeros((0, self.config.hidden_dim))
        else:
            bond_inputs = torch.cat([flat_hidden[edge_src], flat_hidden[edge_dst], batch.edge_attr], dim=-1)
            bond_embeddings = self.bond_head(bond_inputs)
        return EncoderOutput(
            node_embeddings=hidden,
            bond_embeddings=bond_embeddings,
            mol_embeddings=mol_embeddings,
            node_mask=node_mask,
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        encoded = self.encode(batch, compute_bond_embeddings=False)
        return self.projector(encoded.mol_embeddings)
