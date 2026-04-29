from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn


class GraphConvolution(nn.Module):
    """Graph convolution used by the Colab training script: A @ X @ W."""

    def __init__(self, in_feats: int, out_feats: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_feats, out_feats))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.einsum("ij,bjf->bif", adj, x)
        return support @ self.weight


class ColabSTGCN(nn.Module):
    """ST-GCN + LSTM architecture from welcome_to_colab.py."""

    def __init__(self, num_nodes: int, in_feats: int, gcn_hidden: int, lstm_hidden: int):
        super().__init__()
        self.gcn = GraphConvolution(in_feats, gcn_hidden)
        self.lstm = nn.LSTM(
            input_size=num_nodes * gcn_hidden,
            hidden_size=lstm_hidden,
            batch_first=True,
        )
        self.fc = nn.Linear(lstm_hidden, num_nodes)

    def forward(self, x_seq: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch, seq_len, num_nodes, _ = x_seq.shape
        gcn_outputs = []

        for step in range(seq_len):
            h = self.gcn(x_seq[:, step, :, :], adj)
            h = torch.relu(h)
            gcn_outputs.append(h.reshape(batch, num_nodes * h.size(2)))

        seq_tensor = torch.stack(gcn_outputs, dim=1)
        lstm_out, _ = self.lstm(seq_tensor)
        return self.fc(lstm_out)


class CompactTrafficGNNLSTM(nn.Module):
    """Checkpoint-compatible GNN + LSTM forecaster.

    The supplied checkpoint contains a compact per-node graph projection,
    an LSTM, a fusion layer, and two output heads named fc_mean/fc_var.
    This module keeps those exact parameter names so the trained state_dict
    loads strictly.
    """

    def __init__(self, gcn_hidden: int = 64, lstm_hidden: int = 64, aux_features: int = 3):
        super().__init__()
        self.aux_features = aux_features
        self.gcn_weight = nn.Parameter(torch.randn(1, gcn_hidden))
        self.lstm = nn.LSTM(input_size=gcn_hidden, hidden_size=lstm_hidden, batch_first=True)
        self.fc_fusion = nn.Linear(lstm_hidden + aux_features, lstm_hidden)
        self.fc_mean = nn.Linear(lstm_hidden, 1)
        self.fc_var = nn.Linear(lstm_hidden, 1)

    @staticmethod
    def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        denom = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return adj / denom

    def _auxiliary_features(
        self,
        speed: torch.Tensor,
        time_of_day: torch.Tensor,
        degree: torch.Tensor,
    ) -> torch.Tensor:
        candidates = [speed, time_of_day, degree]
        chosen = candidates[: self.aux_features]

        if self.aux_features > len(chosen):
            pad_shape = list(speed.shape)
            for _ in range(self.aux_features - len(chosen)):
                chosen.append(torch.zeros(pad_shape, dtype=speed.dtype, device=speed.device))

        return torch.cat(chosen, dim=-1) if chosen else speed[..., :0]

    def forward(self, x_seq: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        speed = x_seq[..., 0:1]
        time_of_day = x_seq[..., 1:2]
        batch, seq_len, num_nodes, _ = speed.shape

        adj_norm = self._normalize_adj(adj)
        spatial = torch.einsum("ij,btjf->btif", adj_norm, speed)
        gcn_out = torch.relu(spatial @ self.gcn_weight)

        per_node_series = gcn_out.permute(0, 2, 1, 3).reshape(
            batch * num_nodes,
            seq_len,
            gcn_out.size(-1),
        )
        lstm_out, _ = self.lstm(per_node_series)
        lstm_out = lstm_out.reshape(batch, num_nodes, seq_len, -1).permute(0, 2, 1, 3)

        degree = adj_norm.sum(dim=1).view(1, 1, num_nodes, 1).expand(batch, seq_len, num_nodes, 1)
        fused_input = torch.cat(
            [lstm_out, self._auxiliary_features(speed, time_of_day, degree)],
            dim=-1,
        )
        fused = torch.relu(self.fc_fusion(fused_input))
        mean = self.fc_mean(fused).squeeze(-1)
        log_variance = self.fc_var(fused).squeeze(-1)
        return mean, log_variance


@dataclass(frozen=True)
class LoadedModel:
    model: nn.Module
    family: str
    state_dict_keys: list[str]


def _clean_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                checkpoint = value
                break

    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint is not a PyTorch state_dict or checkpoint mapping.")

    state_dict = {}
    for key, value in checkpoint.items():
        cleaned = key.removeprefix("module.")
        state_dict[cleaned] = value
    return state_dict


def _load_checkpoint(path: Path, device: torch.device) -> Mapping[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    return _clean_state_dict(checkpoint)


def build_model_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    num_nodes: int,
) -> LoadedModel:
    keys = set(state_dict.keys())

    if {"gcn_weight", "fc_fusion.weight", "fc_mean.weight", "fc_var.weight"}.issubset(keys):
        gcn_hidden = int(state_dict["gcn_weight"].shape[1])
        lstm_hidden = int(state_dict["lstm.weight_hh_l0"].shape[1])
        fusion_in = int(state_dict["fc_fusion.weight"].shape[1])
        aux_features = fusion_in - lstm_hidden
        model = CompactTrafficGNNLSTM(
            gcn_hidden=gcn_hidden,
            lstm_hidden=lstm_hidden,
            aux_features=aux_features,
        )
        model.load_state_dict(state_dict, strict=True)
        return LoadedModel(model=model, family="compact_gnn_lstm", state_dict_keys=sorted(keys))

    if {"gcn.weight", "lstm.weight_ih_l0", "fc.weight", "fc.bias"}.issubset(keys):
        in_feats = int(state_dict["gcn.weight"].shape[0])
        gcn_hidden = int(state_dict["gcn.weight"].shape[1])
        lstm_hidden = int(state_dict["lstm.weight_hh_l0"].shape[1])
        checkpoint_nodes = int(state_dict["fc.bias"].shape[0])
        model = ColabSTGCN(
            num_nodes=checkpoint_nodes or num_nodes,
            in_feats=in_feats,
            gcn_hidden=gcn_hidden,
            lstm_hidden=lstm_hidden,
        )
        model.load_state_dict(state_dict, strict=True)
        return LoadedModel(model=model, family="colab_stgcn", state_dict_keys=sorted(keys))

    sample_keys = ", ".join(sorted(keys)[:8])
    raise ValueError(f"Unsupported traffic model checkpoint. First keys: {sample_keys}")


def load_trained_model(path: Path, device: torch.device, num_nodes: int) -> LoadedModel:
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    state_dict = _load_checkpoint(path, device)
    loaded = build_model_from_state_dict(state_dict, num_nodes=num_nodes)
    loaded.model.to(device)
    loaded.model.eval()
    return loaded

