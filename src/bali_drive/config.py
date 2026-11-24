from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    video_hidden_dim: int = 2048
    video_layers: int = 28
    video_progress: tuple[float, ...] = (0.0, 0.25, 0.5)
    source_layers: tuple[int, ...] = (7, 14, 21)  # Paper uses one-based layers.
    latent_grid: tuple[int, int, int] = (6, 24, 42)
    state_dim: int = 256
    state_tokens: int = 128
    state_layers: int = 2
    state_heads: int = 8
    ego_dim: int = 4
    command_dim: int = 4
    action_layers: int = 28
    action_heads: int = 16
    action_head_dim: int = 32
    action_context_dim: int = 2048
    horizon: int = 8
    action_dim: int = 3
    initial_steps: int = 5
    refine_steps: int = 3
    refine_time: float = 0.3
    refine_sigma: float = 0.1
    cycle_weight: float = 0.2
    state_weight: float = 0.05
    route_epsilon: float = 1e-6
    # Same affine action coordinates as the released Action DiT. No clipping.
    action_offset: tuple[float, ...] = (1.57, 19.68, 1.67)
    action_span: tuple[float, ...] = (66.74, 42.0, 3.53)

    def __post_init__(self):
        for name in ("video_progress", "source_layers", "latent_grid", "action_offset", "action_span"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        positive = (self.video_hidden_dim, self.video_layers, self.state_tokens, self.state_layers,
                    self.state_heads, self.action_layers, self.action_heads, self.action_head_dim,
                    self.action_context_dim, self.horizon, self.initial_steps, self.refine_steps)
        if any(x <= 0 for x in positive):
            raise ValueError("Dimensions and step counts must be positive")
        if self.state_dim < 6 or self.state_dim % 2 or self.state_dim % self.state_heads:
            raise ValueError("state_dim must be even and divisible by state_heads")
        if len(self.latent_grid) != 3 or min(self.latent_grid) <= 0:
            raise ValueError("latent_grid must be a positive (time, height, width) triple")
        if not self.video_progress or tuple(sorted(set(self.video_progress))) != self.video_progress:
            raise ValueError("video_progress must be unique and increasing")
        if self.video_progress[0] < 0 or self.video_progress[-1] >= 1:
            raise ValueError("video_progress must be in [0, 1)")
        if not self.source_layers or tuple(sorted(set(self.source_layers))) != self.source_layers:
            raise ValueError("source_layers must be unique and increasing")
        if min(self.source_layers) < 1 or max(self.source_layers) > self.video_layers:
            raise ValueError("source_layers are one-based Video DiT indices")
        if (self.ego_dim, self.command_dim, self.action_dim) != (4, 4, 3):
            raise ValueError("Released Action DiT expects ego=4, command=4, action=3")
        if not 0 < self.refine_time < 1 or self.refine_sigma < 0 or self.route_epsilon <= 0:
            raise ValueError("Invalid refinement or routing constants")
        if min(self.cycle_weight, self.state_weight) < 0:
            raise ValueError("Loss weights must be nonnegative")
        if len(self.action_offset) != 3 or len(self.action_span) != 3 or min(self.action_span) <= 0:
            raise ValueError("Invalid action normalization")

    @property
    def source_keys(self):
        return tuple((step, layer) for step in range(len(self.video_progress)) for layer in self.source_layers)

    @property
    def video_tokens(self):
        t, h, w = self.latent_grid
        return t * h * w

    def to_dict(self):
        return asdict(self)

