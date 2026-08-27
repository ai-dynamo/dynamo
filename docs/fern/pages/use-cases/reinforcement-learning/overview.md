---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Reinforcement Learning
subtitle: Use Dynamo as the rollout-serving layer for RL workloads
---

NVIDIA Dynamo provides routing, worker management, weight-update controls, and serving telemetry for reinforcement learning (RL) rollouts. Your RL framework continues to own training, rewards, environments, trajectory semantics, checkpoints, and sample acceptance.

Use Dynamo when rollout serving has become a distributed-systems problem: many workers serve bursty traffic, samples share large prompt prefixes, policies must refresh without rebuilding the serving stack, or you need to diagnose and replay the serving workload independently of the trainer.

## Decide Whether to Add Dynamo

| Current setup | Recommended path |
|---|---|
| One rollout worker with no routing or live-update problem | Keep the framework's direct backend path. |
| Multiple workers serving repeated or bursty prompts | Evaluate Dynamo routing against the current direct-backend baseline. |
| A colocated framework already owns policy transfer | Use Dynamo for generation and routing while keeping the proven framework update path. |
| An external rollout fleet needs discovery and direct control | Integrate Dynamo's request, discovery, and worker-administration surfaces explicitly. |

Before adopting Dynamo, name the bottleneck and the metric that would justify the change. Endpoint connectivity alone does not justify another serving layer.

## Understand the Ownership Boundary

| Responsibility | RL framework | Dynamo | Inference backend |
|---|---|---|---|
| Training, rewards, checkpoints, and sample acceptance | Owns | — | — |
| Rollout requests, retries, and policy freshness | Owns | Transports and reports failures | Executes accepted requests |
| Shared frontend, routing, and request-plane health | Integrates | Owns | Publishes worker state |
| Token IDs, log probabilities, and backend metadata | Verifies | Transports where supported | Produces |
| Policy refresh and fleet-wide update barrier | Orchestrates | Exposes control surfaces | Applies the update |

> [!IMPORTANT]
> Dynamo does not decide whether a trajectory is on-policy, accepted, or fresh enough for training. The framework must gate requests around synchronous updates or enforce its own bounded-staleness policy.

## Choose a Framework

| Framework | Status | Start here |
|---|---|---|
| verl | Experimental | [Integrate with verl](verl.md) for the public colocated Dynamo/vLLM recipe. |
| NeMo RL | Experimental | [Integrate with NeMo RL](nemo-rl.md) for the managed Slurm/Ray Dynamo backend. |
| SLIME | Integration in progress | Review the current boundary in [Framework Compatibility](integration-reference.md#framework-compatibility). |
| Prime-RL | Integration in progress | Review the current boundary in [Framework Compatibility](integration-reference.md#framework-compatibility). |

Experimental guides have runnable upstream artifacts but do not make a general compatibility promise. Integrations in progress remain in the compatibility table until a maintained path lands.

## Choose Your Task

| Goal | Guide |
|---|---|
| Implement or review a framework adapter | [RL Integration Reference](integration-reference.md) |
| Improve cache reuse or worker balance | [Route RL Rollouts](routing.md) |
| Refresh rollout workers after training | [Update Rollout Weights](weight-updates.md) |
| Diagnose a live run or reproduce its serving workload | [Observe and Simulate RL Rollouts](operations-and-simulation.md) |

> [!NOTE]
> Kubernetes is optional for these RL integrations. Use it only when the selected framework or deployment environment requires it.

## Next Step

Start with the guide for your framework. If you are building a new integration, begin with the [RL Integration Reference](integration-reference.md) and preserve the framework's existing token, retry, and policy-update contracts before changing its serving path.
