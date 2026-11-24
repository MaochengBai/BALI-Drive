# Usage Guide

[Back to README](../README.md)

## Backbone Integration

`ActionDiT(action_expert, cfg, preprocess)` accepts a compatible initialized
action expert and its `preprocessing_action_states` callable. It retains the
expert's action modules and projects compact state tokens into the action
cross-attention context. Both planning passes use this same adapter instance.

The default adapter expects 28 action blocks, 16 attention heads, a hidden width
of 512, and a context width of 2048. These dimensions are checked at construction.
The backbone must consume the noisy trajectory rather than replace it with a
learned constant action. The upstream interface is described in
[DriveLaW-Act (2026)](https://github.com/xiaomi-research/drivelaw).

Constructing `BaliDrive` initializes new evidence, tokenizer, feedback, and
state-conditioning parameters. It does not load trained BALI-Drive weights.
The backbone, BALI-Drive parameters, and configuration must be compatible;
changing the horizon or feature dimensions alone does not adapt a checkpoint.

## Inputs

| Argument | Shape / type | Convention |
|:---|:---|:---|
| `raw_bank` | Dictionary of `[B, N_v, 2048]` tensors | Frozen video hidden states keyed by `(slot, layer)` |
| `ego` | `[B, 4]` | `[vx, vy, ax, ay]` in the ego frame |
| `command` | `[B, 4]` | Navigation-command encoding in the upstream category order |
| `expert` | `[B, H, 3]` | Normalized expert trajectory, required by the loss interface |

`B` denotes batch size, `N_v` the number of video tokens per source, and `H` the
trajectory horizon. Use floating-point tensors on devices and with dtypes
compatible with the model.

### Evidence Bank

Use exactly the keys in `cfg.source_keys`. Under the defaults, each of slots
`0`, `1`, and `2` is paired with layers `7`, `14`, and `21`, giving nine sources.
Slots index `cfg.video_progress = (0.0, 0.25, 0.5)`; they are not floating-point
progress values. Layer indices are **one-based**.

All sources must describe the same scenes in the same batch order. Flatten
video tokens in latent-time, row, then column order, with column varying fastest.
The default latent grid is `(6, 24, 42)`, so `N_v = 6048`.
The grid and feature width must match the extracted hidden states. Raw features
are detached before the learnable projection; the evidence is reused during
feedback rather than extracted again.

### Action Coordinates

Physical trajectories contain ego-frame longitudinal displacement, lateral
displacement, and heading, in metres, metres, and radians. The model consumes
and returns normalized action coordinates. Convert at the interface using
`normalize_actions(action, cfg)` and `denormalize_actions(action, cfg)` from
[`flow.py`](../src/bali_drive/flow.py):

```text
normalized = 2 * (physical + offset) / span - 1
physical   = (normalized + 1) * span / 2 - offset

offset = (1.57, 19.68, 1.67)
span   = (66.74, 42.0, 3.53)
```

These are affine transforms without clipping. Do not normalize an expert
trajectory twice or interpret normalized predictions as metres.

## Model API

| Entry point | Output |
|:---|:---|
| `model.encode(raw_bank, ego, command)` | `(bank, state, prior)` |
| `model(raw_bank, ego, command, expert, stage="initial")` | `loss`, `flow_initial` |
| `model(raw_bank, ego, command, expert, stage="joint")` | `loss`, `flow_initial`, `flow_refine`, `cycle`, `state_penalty`, `gate_mean` |
| `model.sample(raw_bank, ego, command)` | `initial`, `refined`, `route_prior`, `gate` |

Under the default configuration:

- `encode` returns projected evidence `[B, 9 * N_v, 256]`, initial state
  `[B, 128, 256]`, and source prior `[B, 9]`.
- `sample` returns normalized `initial` and `refined` trajectories `[B, 8, 3]`,
  `route_prior` `[B, 9]`, and a token-wise `gate` `[B, 128, 1]`.
- `loss` is differentiable. Other loss-dictionary values are detached diagnostics.

Call `model.eval()` before `sample`; sampling disables gradient recording.
For controlled randomness, pass a device-compatible `generator`, or provide
`initial_noise` and `refine_noise`, each of shape `[B, H, 3]`, on the input device.
These control the noise, not platform-wide numerical determinism.

## Configuration

Defaults are defined in [`ModelConfig`](../src/bali_drive/config.py).

| Component | Default |
|:---|:---|
| Video source progress | `(0.0, 0.25, 0.5)` |
| Video source layers | `(7, 14, 21)`, one-based |
| Raw features | Width 2048; latent grid `(6, 24, 42)` |
| Planning state | 128 tokens of width 256 |
| State Transformer | 2 layers, 8 heads |
| Action DiT | 28 blocks, 16 heads, 32 channels per head |
| Action cross-attention context | Width 2048 |
| Trajectory | 8 waypoints, 3 coordinates each |
| Initial flow integration | 5 reverse Euler steps, from time 1 to 0 |
| Local refinement | 3 reverse Euler steps, from time 0.3 to 0 |
| Refinement noise parameter | `refine_sigma = 0.1` |
| Joint-objective weights | `cycle_weight = 0.2`, `state_weight = 0.05` |

Video denoising progress and action flow time are distinct quantities. The
refinement anchor is `initial + refine_time * refine_sigma * noise`; the default
perturbation is therefore `0.03 * noise`, not `0.1 * noise`.
The joint objective is
`flow_initial + flow_refine + cycle_weight * (cycle + state_weight * state_penalty)`.

## Source Map

| File | Responsibility |
|:---|:---|
| [`model.py`](../src/bali_drive/model.py) | Module composition, losses, and the bidirectional planning loop |
| [`evidence.py`](../src/bali_drive/evidence.py) | Bank validation, projection, and source embeddings |
| [`positional.py`](../src/bali_drive/positional.py) | Factorized spatiotemporal positions |
| [`tokenizer.py`](../src/bali_drive/tokenizer.py) | Source routing, resampling, and state Transformer |
| [`planner.py`](../src/bali_drive/planner.py) | Shared action expert and compact-state conditioning |
| [`feedback.py`](../src/bali_drive/feedback.py) | Candidate-conditioned retrieval and gated state update |
| [`flow.py`](../src/bali_drive/flow.py) | Flow bridges, integration, and coordinate transforms |
| [`config.py`](../src/bali_drive/config.py) | Architecture and method parameters |
