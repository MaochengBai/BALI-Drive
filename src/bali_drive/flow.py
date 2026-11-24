import torch


def initial_bridge(expert, epsilon, tau):
    time = tau[:, None, None]
    return (1 - time) * expert + time * epsilon, epsilon - expert


def training_candidate(noisy, velocity, tau):
    return (noisy - tau[:, None, None] * velocity).detach()


def refinement_bridge(expert, candidate, noise, nu, start_time, sigma):
    anchor = (candidate.detach() + start_time * sigma * noise).detach()
    ratio = nu[:, None, None] / start_time
    return (1 - ratio) * expert + ratio * anchor, (anchor - expert) / start_time, anchor


def euler_integrate(planner, action, state, ego, command, start_time, steps):
    if not 0 < start_time <= 1 or steps < 1:
        raise ValueError("Invalid Euler interval or step count")
    dt = start_time / steps
    for index in range(steps):
        tau = torch.full((action.shape[0],), start_time - index * dt,
                         device=action.device, dtype=torch.float32)
        action = action - dt * planner(action, tau, state, ego, command)
    return action


def normalize_actions(action, cfg):
    offset = action.new_tensor(cfg.action_offset)
    span = action.new_tensor(cfg.action_span)
    return 2 * (action + offset) / span - 1


def denormalize_actions(action, cfg):
    return (action + 1) * action.new_tensor(cfg.action_span) / 2 - action.new_tensor(cfg.action_offset)
