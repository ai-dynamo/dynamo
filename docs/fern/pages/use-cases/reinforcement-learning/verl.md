---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: verl Integration
subtitle: Use the verl-recipe Dynamo rollout backend
---

**Experimental.** The [verl-recipe Dynamo rollout backend](https://github.com/verl-project/verl-recipe/tree/main/dynamo) runs verl training with vLLM or SGLang rollout workers behind a shared Dynamo frontend and KV-aware router. The recipe owns its tested versions, install steps, launchers, and results. Follow the upstream README for those; this page is a quick reference.

## Integration Shape

| Concern | Current path |
|---|---|
| Training and rollout orchestration | verl (`verl.trainer.main_ppo`) with the verl-recipe backend registered |
| Generation | Dynamo vLLM or SGLang workers behind one shared frontend per rollout pool |
| Routing | Dynamo KV router (`router_mode=kv`) when ThunderAgent is disabled; ThunderAgent when `thunderagent.enabled=true` |
| GPU placement | Shared training and rollout GPUs, or a separate rollout pool |
| Policy update | Recipe-owned node-local CUDA IPC; a separate pool first receives weights through verl's checkpoint engine |
| Service discovery | Recipe-managed etcd and NATS, started on the first node of the pool |
| Deployment | Ray on one or more nodes; Kubernetes is not required |

The Dynamo KV router and ThunderAgent are different scheduling paths. Choose one before you build the environment, and do not compare their results as if only a router flag changed.

## Quick Reference

| Task | Upstream |
|---|---|
| Check the tested verl, Dynamo, vLLM, and SGLang versions | [Required versions](https://github.com/verl-project/verl-recipe/tree/main/dynamo#required-versions) |
| Build and install Dynamo from the pinned commit | [Installation guide](https://github.com/verl-project/verl-recipe/blob/main/dynamo/INSTALL.md) |
| Launch training from a preset | [Launch](https://github.com/verl-project/verl-recipe/tree/main/dynamo#launch) and [`config/`](https://github.com/verl-project/verl-recipe/tree/main/dynamo/config) |
| Run a short training smoke | [`tests/smoke/`](https://github.com/verl-project/verl-recipe/tree/main/dynamo/tests/smoke) |
| Set Dynamo options | [Configuration](https://github.com/verl-project/verl-recipe/tree/main/dynamo#configuration) and [`dynamo_base.yaml`](https://github.com/verl-project/verl-recipe/blob/main/dynamo/config/dynamo_base.yaml) |
| Enable ThunderAgent | [Enable ThunderAgent](https://github.com/verl-project/verl-recipe/tree/main/dynamo#enable-thunderagent) and the [vLLM 0.24 ThunderAgent guide](https://github.com/verl-project/verl-recipe/blob/main/dynamo/THUNDERAGENT_VLLM024.md) |
| Compare against native verl rollouts | [Results](https://github.com/verl-project/verl-recipe/tree/main/dynamo#results) |

The recipe pins a Dynamo source commit and builds Dynamo from it. A published Dynamo release that differs from that commit is not a tested pair.

Dynamo options live under `actor_rollout_ref.rollout.engine_kwargs.dynamo`. Keep `actor_rollout_ref.rollout.name=dynamo` for both engines, and select the engine with `engine=vllm` or `engine=sglang`. To score the exact tokens that the engine generated, set `request_completion_token_ids=true`.

## Related Pages

- [RL Integration Reference](integration-reference.md#preserve-token-authority) for the shared token and response checks.
- [KV-Aware Load Balancing for RL Rollouts](routing.md) for router experiments.
- [Distribute and Update Rollout Weights](weight-updates.md#verl-update) for the policy-update boundary.
- [Profile and Simulate RL Rollouts](operations-and-simulation.md) for request tracing.
