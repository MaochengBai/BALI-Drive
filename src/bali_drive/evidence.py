from collections.abc import Mapping
import torch
from torch import nn

from .config import ModelConfig
from .positional import position_3d

Bank = Mapping[tuple[int, int], torch.Tensor]


def validate_bank(bank: Bank, cfg: ModelConfig):
    if set(bank) != set(cfg.source_keys):
        raise ValueError(f"Bank sources must be exactly {cfg.source_keys}; got {tuple(bank)}")
    first = bank[cfg.source_keys[0]]
    if first.ndim != 3 or first.shape[0] == 0:
        raise ValueError("Expected raw bank tensors [batch, video_tokens, video_hidden_dim]")
    for key in cfg.source_keys:
        value = bank[key]
        if value.shape != (first.shape[0], cfg.video_tokens, cfg.video_hidden_dim):
            raise ValueError(f"Incorrect bank shape at {key}: {tuple(value.shape)}")
        if not value.is_floating_point() or value.device != first.device or value.dtype != first.dtype:
            raise ValueError("All bank tensors must share floating dtype and device")


class EvidenceProjector(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.projection = nn.Linear(cfg.video_hidden_dim, cfg.state_dim, bias=False)
        self.step_embedding = nn.Embedding(len(cfg.video_progress), cfg.state_dim)
        self.layer_embedding = nn.Embedding(cfg.video_layers, cfg.state_dim)
        self.norm = nn.LayerNorm(cfg.state_dim)
        self.register_buffer("positions", position_3d(cfg.latent_grid, cfg.state_dim), persistent=False)

    def forward(self, bank: Bank):
        validate_bank(bank, self.cfg)
        output = []
        for step, layer in self.cfg.source_keys:
            # The world model is frozen; the projection remains differentiable.
            x = self.projection(bank[step, layer].detach())
            x = x + self.positions.to(x.dtype)
            x = x + self.step_embedding.weight[step] + self.layer_embedding.weight[layer - 1]
            output.append(self.norm(x))
        return torch.stack(output, dim=1)  # [B, G, N, d_s]
