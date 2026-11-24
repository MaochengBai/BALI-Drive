import torch
from torch import nn

from .config import ModelConfig


class ActionConditionedFeedback(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.state_dim
        self.cfg = cfg
        self.action_embedding = nn.Linear(cfg.action_dim, d)
        self.waypoint_embedding = nn.Embedding(cfg.horizon, d)
        self.ego_embedding = nn.Linear(cfg.ego_dim, d)
        self.command_embedding = nn.Linear(cfg.command_dim, d)
        self.condition = nn.Linear(cfg.ego_dim + cfg.command_dim, d)
        self.action_to_bank = nn.MultiheadAttention(d, cfg.state_heads, batch_first=True, dropout=0.0)
        self.state_to_action = nn.MultiheadAttention(d, cfg.state_heads, batch_first=True, dropout=0.0)
        self.delta = nn.Sequential(nn.Linear(3 * d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.gate = nn.Linear(3 * d, 1)
        self.norm = nn.LayerNorm(d)

    def forward(self, candidate, state, bank, ego, command):
        if candidate.shape[1:] != (self.cfg.horizon, self.cfg.action_dim):
            raise ValueError("Candidate must match the configured action horizon")
        condition = self.condition(torch.cat((ego, command), dim=-1))[:, None]
        u = self.action_embedding(candidate) + self.waypoint_embedding.weight[None]
        u = u + self.ego_embedding(ego)[:, None] + self.command_embedding(command)[:, None]
        evidence, _ = self.action_to_bank(u, bank, bank, need_weights=False)
        aligned, _ = self.state_to_action(state + condition, evidence, evidence, need_weights=False)
        features = torch.cat((state, aligned, condition.expand_as(state)), dim=-1)
        delta = self.delta(features)
        gate = self.gate(features).sigmoid()
        return self.norm(state + gate * delta), gate
