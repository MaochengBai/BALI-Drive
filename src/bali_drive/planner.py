from torch import nn
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


class ActionDiT(nn.Module):
    """Full action expert, with compact state tokens as its only cross-attention context.

    Reuses the released action blocks rather than a smaller substitute. Only
    action modules are retained: the video backbone is not registered here.
    """

    def __init__(self, expert, cfg: ModelConfig, preprocess):
        super().__init__()
        self.cfg = cfg
        if any("lora" in name.lower() for name, _ in expert.named_parameters()):
            raise ValueError("Expected an unadapted action expert; adapter checkpoints are unsupported")
        self.expert = nn.Module()
        for name, module in expert.named_children():
            if name.startswith("action_"):
                self.expert.add_module(name, module)
        for name, param in expert.named_parameters(recurse=False):
            if name.startswith("action_"):
                self.expert.register_parameter(name, param)
        for name, buffer in expert.named_buffers(recurse=False):
            if name.startswith("action_"):
                self.expert.register_buffer(name, buffer)
        for name in ("action_expert", "action_inner_dim", "action_final_embeddings", "learnable_action_state"):
            setattr(self.expert, name, getattr(expert, name))
        if self.expert.learnable_action_state:
            raise ValueError("A learned constant action would discard the noisy trajectory input")
        if len(self.expert.action_blocks) != cfg.action_layers:
            raise ValueError("Action DiT layer count differs from config")
        for block in self.expert.action_blocks:
            if block.attn1.heads != cfg.action_heads or block.attn2.to_k.in_features != cfg.action_context_dim:
                raise ValueError("Action DiT heads/context width differ from config")
        if self.expert.action_inner_dim != cfg.action_heads * cfg.action_head_dim:
            raise ValueError("Action hidden width differs from config")
        self.state_to_context = nn.Linear(cfg.state_dim, cfg.action_context_dim)
        self.expert.requires_grad_(True)
        self.preprocess = preprocess
        self.gradient_checkpointing = False

    def forward(self, action, tau, state, ego, command):
        cfg = self.cfg
        batch = action.shape[0]
        if action.shape != (batch, cfg.horizon, cfg.action_dim) or tau.shape != (batch,):
            raise ValueError("Expected actions [B,H,3] and continuous times [B]")
        if state.shape != (batch, cfg.state_tokens, cfg.state_dim):
            raise ValueError("Expected compact planning state, not the full video bank")
        context = self.state_to_context(state)
        # Sinusoidal time embeddings accept floats: do not round continuous flow time.
        timesteps = (tau.float() * 1000.0)[:, None].expand(-1, cfg.horizon)
        temb, time_embedding, rotary, hidden = self.preprocess(
            self.expert, action_states=action, action_timestep=timesteps,
            context_dict={"vel": ego[:, :2], "acc": ego[:, 2:], "cmd_onehot": command})
        for block in self.expert.action_blocks:
            kwargs = dict(hidden_states=hidden, encoder_hidden_states=context, temb=temb, rotary_emb=rotary)
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(block, use_reentrant=False, **kwargs)
            else:
                hidden = block(**kwargs)
        hidden = self.expert.action_norm_out(hidden)
        if self.expert.action_final_embeddings:
            shift_scale = self.expert.action_scale_shift_table[None, None] + time_embedding[:, :, None]
            hidden = hidden * (1 + shift_scale[:, :, 1]) + shift_scale[:, :, 0]
        else:
            hidden = self.expert.action_proj_extra(hidden)
        return self.expert.action_proj_out(hidden)
