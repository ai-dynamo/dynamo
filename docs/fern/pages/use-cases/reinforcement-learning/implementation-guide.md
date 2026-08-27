---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: RL Implementation Guide
subtitle: Find the task-specific RL integration, optimization, and operations guides
---

**Experimental.** This URL is retained so existing links continue to work. The original implementation guide combined generation contracts, backend setup, worker discovery, engine administration, weight updates, and custom routes. Those topics now live in focused pages with explicit maturity and evidence boundaries.

## Choose the Page for Your Task

| Task | Go to |
|---|---|
| Implement an RL framework adapter or check an exact framework/backend combination | [RL integration and compatibility reference](integration-reference.md) |
| Run the public verl recipe | [Integrate with verl](verl.md) |
| Run the managed NeMo RL backend | [Integrate with NeMo RL](nemo-rl.md) |
| Track or review the SLIME integration | [SLIME integration status](slime.md) |
| Track or review the Prime-RL integration | [Prime-RL integration status](prime-rl.md) |
| Select and tune rollout routing | [Route RL rollouts](routing.md) |
| Pause workers, refresh policy weights, verify versions, and recover | [Update rollout weights](weight-updates.md) |
| Correlate a rollout with traces and metrics, troubleshoot it, or replay/simulate its request plane | [Observe, debug, replay, and simulate RL rollouts](operations-and-simulation.md) |

Start at the [RL overview](overview.md) if you need the system boundary, maturity legend, or framework chooser.
