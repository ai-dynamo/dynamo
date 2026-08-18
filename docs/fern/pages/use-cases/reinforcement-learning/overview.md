---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Reinforcement Learning
subtitle: Use Dynamo as the rollout-serving plane for RL training systems
---

**Experimental.** Dynamo can serve rollout generation for reinforcement learning systems that need more than a static inference endpoint. RL frameworks remain responsible for the training loop, reward pipeline, policy update logic, and checkpoint production; Dynamo provides the serving plane around rollout workers.

Use Dynamo when your RL system needs low-latency generation, backend-aware routing, worker discovery, rollout metadata, weight refreshes, fault tolerance, and autoscaling as part of the training loop. The goal is to let RL engineers operate the rollout path with production serving primitives while still integrating with the framework that owns training.

## Where Dynamo Fits

A typical RL setup has three planes:

| Plane | Owned by | Dynamo role |
|---|---|---|
| Training | RL framework | Produces updated policy weights and decides when rollout workers should refresh. |
| Rollout serving | Dynamo | Routes generation requests, exposes token and log probability data, discovers live workers, and provides engine control surfaces. |
| Operations | Platform stack | Scales capacity, observes health, handles failures, and manages deployment lifecycle. |

Dynamo sits between the RL orchestrator and inference backends such as vLLM, SGLang, and TensorRT-LLM. For SGLang rollouts, use Dynamo's SGLang-compatible `POST /generate` or `PUT /generate` API. This API routes token-input requests through Dynamo. It preserves SGLang's native streaming response objects. The OpenAI-compatible frontend remains available for cross-backend integrations. Use backend-specific control surfaces to manage workers.

## What Dynamo Adds

| Capability | Why it matters for RL rollouts |
|---|---|
| Advanced routing | Steer rollout traffic across workers based on cache locality, backend metadata, load, or deployment topology instead of treating all workers as identical endpoints. |
| Weight synchronization | Use Model Express and engine control routes to move updated checkpoints into serving without rebuilding the whole rollout stack. |
| Fault tolerance | Keep rollout generation available when requests, engines, or workers fail, and recover without forcing the RL job to restart its serving plane. |
| Autoscaling | Match rollout-serving capacity to changing training demand, including bursty generation phases and idle windows between policy updates. |
| Token and metadata surfaces | Return token IDs, prompt log probabilities, completion log probabilities, routed expert data, and backend metadata needed by RL pipelines. |

## Integration Pattern

1. Deploy Dynamo with the inference backend you want to use for rollouts.
2. For SGLang, enable the SGLang-compatible `/generate` API.
3. Send native token-input requests through the Dynamo frontend.
4. For cross-backend clients, use the OpenAI-compatible completion or chat routes.
5. When you use the OpenAI-compatible routes, request token and log probability fields through NVIDIA request extensions.
6. Discover live rollout workers when the orchestrator needs direct worker administration.
7. Pause selected workers, refresh weights, validate the update, and resume generation.
8. Use Dynamo's routing, autoscaling, and fault-tolerance features to keep rollout serving aligned with training demand.

For the concrete API shapes, environment variables, and command examples, see the [RL Implementation Guide](implementation-guide.md).

## Framework Integrations

Use Dynamo as the rollout-serving plane behind an RL framework. The framework remains responsible for the training loop and policy updates; Dynamo serves rollout generation and provides production serving capabilities around the rollout workers.

| Framework or example | Dynamo integration path | Status |
|---|---|---|
| [verl Dynamo rollout backend recipe](https://github.com/verl-project/verl-recipe/blob/main/dynamo/README.md) | Run Dynamo as an async rollout backend with KV-aware routing, rollout token data, and weight-update control. | Available recipe |
| [prime-rl Dynamo training recipes](https://github.com/PrimeIntellect-ai/prime-rl/pull/3180) | Train against an external Dynamo/vLLM rollout-serving stack using Dynamo worker discovery and weight-update control. The PR includes Dynamo example configs for Qwen3 0.6B Math, Qwen3 30B Thinking, and GLM-5.2 FP8 R2E. | Open PR |
| [Slime external rollout endpoint](https://github.com/Aphoh/slime/pull/1) | Point Slime's external SGLang-compatible engine path at Dynamo's `/generate` API. Use direct per-worker routes for engine controls. | Open PR |

## Backend Support Snapshot

| Capability | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| Engine-native token-in/token-out API | `/inference/v1/generate` | `/generate` | Not supported |
| Token input through `prompt` token arrays | Supported | Supported | Supported |
| `nvext.token_data` tokenizer bypass | Supported | Supported | Supported |
| Completion token IDs | Supported | Supported | Supported |
| Prompt log probabilities | Supported | Supported | Not supported |
| RL worker discovery | Supported with `--enable-rl` | Not supported | Not supported |
| Direct RL administration routes | Supported with `--enable-rl` | Backend-specific routes | Not supported |
| SGLang metadata upload | Not applicable | Supported with `--enable-rl` | Not applicable |

## Start Here

Use the [RL Implementation Guide](implementation-guide.md) when you are ready to wire an orchestrator to Dynamo. It covers:

- The vLLM happy path for token-in rollouts, worker discovery, and weight updates.
- The recommended SGLang `/generate` path for native token-in/token-out rollouts.
- NVIDIA request extensions for token IDs, log probabilities, routed expert data, and SGLang metadata uploads.
- The `/v1/rl/workers` discovery API and direct `/engine/` administration routes.
- How to register custom engine routes for framework-specific rollout control.
