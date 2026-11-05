---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Slime Integration
subtitle: Run Slime external rollouts through Dynamo on Kubernetes
---

**Experimental.** [Slime](https://github.com/THUDM/slime) is an RL post-training framework that trains with Megatron-LM and generates rollouts with SGLang. In this integration, a DynamoGraphDeployment (DGD) runs a fixed set of SGLang engines in place of the engines that Slime would otherwise launch itself.

Slime sends rollout generation through the Dynamo frontend. It sends control and weight-update requests to each SGLang engine.

## Integration Shape

| Concern | Current path |
|---|---|
| Training and rollout orchestration | Slime, running on a Ray cluster |
| Generation | Dynamo frontend with SGLang workers |
| Routing | Dynamo routes rollout requests to the fixed worker set |
| Policy update | Slime calls the native SGLang APIs for each worker |
| Service discovery | Fixed Kubernetes Services |
| Deployment | DGD for the engines; a KubeRay `RayCluster` for the trainer |

Slime starts an SGLang router to track the external engines. A custom generate function sends rollout requests to the Dynamo frontend instead of that router.

## How a Training Iteration Runs

1. At startup, Slime reads `/server_info` from each engine address passed in `--rollout-external-engine-addrs` and registers the engines with its own SGLang router.
2. For each rollout batch, the custom generate function sends streaming `/generate` requests to the Dynamo frontend. The frontend routes them to the SGLang workers.
3. Megatron-LM trains the policy on the completed rollouts.
4. Slime calls `/pause_generation` on every engine, transfers the new weights, and calls `/continue_generation`.

The custom generate function is a short Python module that Slime loads with `--custom-generate-function-path`. It copies Slime's arguments, replaces the router address with the Dynamo frontend address, and calls Slime's streaming generator. The [tutorial](tutorial.mdx) creates it as a ConfigMap. The Dynamo example keeps a copy in [`examples/rl/slime/dynamo_generate.py`](https://github.com/ai-dynamo/dynamo/blob/main/examples/rl/slime/dynamo_generate.py).

| Source | Files |
|---|---|
| Slime | External-engine and streaming rollout support from [THUDM/slime#2272](https://github.com/THUDM/slime/pull/2272) in [`slime/rollout/`](https://github.com/THUDM/slime/tree/main/slime/rollout) |
| Dynamo | Fixed worker set example in [`examples/rl/slime/`](https://github.com/ai-dynamo/dynamo/tree/main/examples/rl/slime) |

## Choose a Weight-Update Transport

Slime transfers new weights to external engines in one of three ways. Pick one with `--update-weight-transport` and `--update-weight-mode`.

| Transport | Slime flags | Requirement | Status with Dynamo SGLang images |
|---|---|---|---|
| Full checkpoint from disk | `--update-weight-transport disk --update-weight-disk-dir <dir>` | The trainer and every engine mount the same shared path | Works with SGLang 0.5.21. The tutorial uses this path. |
| NCCL | `--update-weight-transport nccl` (default) | The trainer and the engines form one NCCL group with compatible NCCL versions | Fails with SGLang 0.5.21 engines. See [Known Issues](../../../reference/general/releases/known-issues.mdx). |
| Delta from disk | `--update-weight-mode delta --update-weight-transport disk` | Every engine serves the `/pull_weights` route from Slime's SGLang patch | Not available. Dynamo images run unpatched SGLang. |

SGLang 0.5.21 accepts `update_weights_from_distributed` only inside a `begin_weight_update` and `end_weight_update` session. Slime does not open that session, so the first NCCL update returns HTTP 400. `update_weights_from_disk` does not require a session.

## Get Started

- [Installation](installation.mdx) lists the operators, storage, and command-line tools to set up first.
- [Tutorial](tutorial.mdx) runs a short GRPO training loop on Qwen3-0.6B with two Dynamo-managed SGLang engines.

## Make Sure That the Integration Works

Complete these steps before you increase the workload size:

1. Run one rollout with the selected model and weight transport.
2. Apply one policy update to every engine.
3. Run one rollout after the update.
4. Stop one worker during a test run.
5. Make sure that Slime stops or reports the worker failure.
6. Restart the Slime job after Kubernetes replaces the worker Pod.

The integration does not recover a replaced external engine during a Slime job.

## Current Limitations

- The example uses a fixed worker set.
- The example does not support dynamic endpoint registration.
- Slime must restart after Kubernetes replaces an engine Pod.
- The native engine Services require network isolation because they expose administrative and weight-update APIs.
- The example does not provide a fleet-wide policy-update transaction or automatic rollback.
- External engines run on their own GPUs, so `--colocate` does not apply. Plan separate GPUs for the trainer and the engines.
- Slime does not support score centering (`--use-score-centering`) with its streaming rollout generator, which the custom generate function wraps.

## Upstream Resources

- [Dynamo Slime example](https://github.com/ai-dynamo/dynamo/tree/main/examples/rl/slime)
- [Slime streaming external-rollout change](https://github.com/THUDM/slime/pull/2272)
- [Slime external rollout engine guide](https://github.com/THUDM/slime/blob/main/docs/en/advanced/external-rollout-engines.md)
- [Shared RL integration reference](../integration-reference.md)
