from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FrequencyGatedMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        num_leaves: int,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
    ):
        super().__init__()
        if num_leaves <= 0:
            raise ValueError("num_leaves must be > 0")

        trunk_layers: list[nn.Module] = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            trunk_layers.append(nn.Linear(prev_dim, h_dim))
            trunk_layers.append(nn.BatchNorm1d(h_dim))
            trunk_layers.append(nn.ReLU())
            trunk_layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.trunk = nn.Sequential(*trunk_layers)
        self.heads = nn.ModuleList([nn.Linear(prev_dim, num_classes) for _ in range(num_leaves)])
        self.num_classes = int(num_classes)
        self.num_leaves = int(num_leaves)

    def forward(self, x: torch.Tensor, leaf_id: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        leaf_id = leaf_id.to(dtype=torch.long)
        out = torch.empty((x.shape[0], self.num_classes), device=x.device, dtype=h.dtype)
        for lid in torch.unique(leaf_id).tolist():
            m = leaf_id == int(lid)
            if int(m.sum().item()) == 0:
                continue
            out[m] = self.heads[int(lid)](h[m])
        return out

