import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .evidence import EvidenceProjector
from .tokenizer import StateTokenizer
from .feedback import ActionConditionedFeedback
from .flow import initial_bridge, training_candidate, refinement_bridge, euler_integrate


class BaliDrive(nn.Module):
    def __init__(self, cfg: ModelConfig, planner: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.projector = EvidenceProjector(cfg)
        self.tokenizer = StateTokenizer(cfg)
        self.feedback = ActionConditionedFeedback(cfg)
        self.planner = planner

    def encode(self, raw_bank, ego, command):
        batch = next(iter(raw_bank.values())).shape[0]
        if ego.shape != (batch, self.cfg.ego_dim) or command.shape != (batch, self.cfg.command_dim):
            raise ValueError("Condition shapes must match the bank batch")
        bank = self.projector(raw_bank)
        state, prior = self.tokenizer(bank, ego, command)
        return bank.flatten(1, 2), state, prior

    def forward(self, raw_bank, ego, command, expert, stage="joint", generator=None):
        """Training losses. expert is in the action expert's normalized coordinates."""
        if stage not in ("initial", "joint"):
            raise ValueError("stage must be initial or joint")
        if expert.shape != (ego.shape[0], self.cfg.horizon, self.cfg.action_dim):
            raise ValueError("Expert trajectory shape differs from config")
        bank, s0, prior = self.encode(raw_bank, ego, command)
        tau = torch.rand(expert.shape[0], device=expert.device, generator=generator)
        epsilon = torch.randn(expert.shape, device=expert.device, dtype=expert.dtype, generator=generator)
        noisy, target0 = initial_bridge(expert, epsilon, tau)
        velocity0 = self.planner(noisy, tau, s0, ego, command)
        loss0 = F.mse_loss(velocity0.float(), target0.float())
        if stage == "initial":
            return {"loss": loss0, "flow_initial": loss0.detach()}
        candidate = training_candidate(noisy, velocity0, tau)
        s1, gate = self.feedback(candidate, s0, bank, ego, command)
        nu = self.cfg.refine_time * torch.rand(expert.shape[0], device=expert.device, generator=generator)
        noise = torch.randn(expert.shape, device=expert.device, dtype=expert.dtype, generator=generator)
        noisy1, target1, _ = refinement_bridge(expert, candidate, noise, nu,
                                              self.cfg.refine_time, self.cfg.refine_sigma)
        velocity1 = self.planner(noisy1, nu, s1, ego, command)
        loss1 = F.mse_loss(velocity1.float(), target1.float())
        refined = noisy1 - nu[:, None, None] * velocity1
        cycle = F.l1_loss(refined.float(), expert.float())
        state_penalty = F.mse_loss(s1.float(), s0.float())
        total = loss0 + loss1 + self.cfg.cycle_weight * (cycle + self.cfg.state_weight * state_penalty)
        return {"loss": total, "flow_initial": loss0.detach(), "flow_refine": loss1.detach(),
                "cycle": cycle.detach(), "state_penalty": state_penalty.detach(),
                "gate_mean": gate.detach().mean()}

    @torch.no_grad()
    def sample(self, raw_bank, ego, command, generator=None, initial_noise=None, refine_noise=None):
        """Algorithm 1: one bank, one feedback update, one shared 5+3-step planner."""
        if self.training:
            raise RuntimeError("Call model.eval() before sampling")
        bank, s0, prior = self.encode(raw_bank, ego, command)
        shape = (ego.shape[0], self.cfg.horizon, self.cfg.action_dim)
        def noise_or(value):
            if value is None:
                return torch.randn(shape, device=ego.device, dtype=ego.dtype, generator=generator)
            if value.shape != shape or value.device != ego.device:
                raise ValueError("Supplied noise has the wrong shape/device")
            return value.to(ego.dtype)
        initial = euler_integrate(self.planner, noise_or(initial_noise), s0, ego, command,
                                  1.0, self.cfg.initial_steps)
        s1, gate = self.feedback(initial, s0, bank, ego, command)
        anchor = initial + self.cfg.refine_time * self.cfg.refine_sigma * noise_or(refine_noise)
        final = euler_integrate(self.planner, anchor, s1, ego, command,
                               self.cfg.refine_time, self.cfg.refine_steps)
        return {"initial": initial, "refined": final, "route_prior": prior, "gate": gate}
