import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class StateTokenizer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.state_dim
        self.ego_embedding = nn.Linear(cfg.ego_dim, d)
        self.command_embedding = nn.Linear(cfg.command_dim, d)
        self.condition = nn.Linear(cfg.ego_dim + cfg.command_dim, d)
        self.routing = nn.Sequential(nn.Linear(3 * d, d), nn.SiLU(), nn.Linear(d, 1, bias=False))
        self.queries = nn.Parameter(torch.randn(cfg.state_tokens, d) / d ** 0.5)
        self.to_q = nn.Linear(d, d, bias=False)
        self.to_k = nn.Linear(d, d, bias=False)
        self.to_v = nn.Linear(d, d, bias=False)
        block = nn.TransformerEncoderLayer(d, cfg.state_heads, 4 * d, dropout=0,
                                          activation="gelu", batch_first=True, norm_first=True)
        self.state_transformer = nn.TransformerEncoder(block, cfg.state_layers, enable_nested_tensor=False)
        self.state_projection = nn.Linear(d, d, bias=False)
        self.state_norm = nn.LayerNorm(d)

    def forward(self, bank, ego, command):
        b, groups, tokens, d = bank.shape
        ee = self.ego_embedding(ego)[:, None].expand(-1, groups, -1)
        eg = self.command_embedding(command)[:, None].expand(-1, groups, -1)
        logits = self.routing(torch.cat((bank.mean(dim=2), ee, eg), dim=-1)).squeeze(-1)
        prior = logits.float().softmax(dim=-1)
        bias = (prior + self.cfg.route_epsilon).log().repeat_interleave(tokens, dim=-1)
        condition = self.condition(torch.cat((ego, command), dim=-1))[:, None]
        flat = bank.reshape(b, groups * tokens, d)
        q = self.to_q(self.queries[None] + condition)
        # One d_s-dimensional attention matches the paper's sqrt(d_s) scaling.
        z = F.scaled_dot_product_attention(q[:, None], self.to_k(flat)[:, None],
                                          self.to_v(flat)[:, None],
                                          attn_mask=bias[:, None, None].to(q.dtype), dropout_p=0.0).squeeze(1)
        state = self.state_transformer(z + condition)
        state = self.state_norm(self.state_projection(state))
        return state, prior
