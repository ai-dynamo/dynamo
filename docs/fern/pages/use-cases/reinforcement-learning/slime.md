---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Slime Integration
subtitle: Run Slime external rollouts through Dynamo on Kubernetes
---

**Experimental.** This integration connects Slime to a fixed set of SGLang engines that a DynamoGraphDeployment manages.

Slime sends rollout generation through the Dynamo frontend. It sends control and weight-update requests to each SGLang engine.

## Integration Shape

| Concern | Current path |
|---|---|
| Training and rollout orchestration | Slime |
| Generation | Dynamo frontend with SGLang workers |
| Routing | Dynamo routes rollout requests to the fixed worker set |
| Policy update | Slime calls the native SGLang APIs for each worker |
| Service discovery | Fixed Kubernetes Services |
| Deployment | DynamoGraphDeployment on Kubernetes |

Slime starts an SGLang router to track the external engines. The custom generator sends rollout requests to the Dynamo frontend instead.

The Slime external-engine support lives in the upstream Slime repository. The Dynamo-specific deployment files live in the Dynamo example.

| Source | Files |
|---|---|
| Slime | [`slime/rollout/`](https://github.com/THUDM/slime/tree/4c1ab40203952b3dcc8582b653f3a83f2c6e8128/slime/rollout) |
| Dynamo | [`examples/rl/slime/`](https://github.com/ai-dynamo/dynamo/blob/main/examples/README.md#integration-examples) |

## Prerequisites

- Install the [Dynamo Kubernetes Platform](../../kubernetes/installation/install-dynamo.md) on a GPU cluster.
- Install the NVIDIA device plugin for the `nvidia.com/gpu` resource.
- Install `kubectl` and `envsubst` on the deployment host.
- Select matching Dynamo frontend and SGLang runtime image versions.
- Allocate two GPUs for the SGLang workers.
- Allocate the additional resources that the Slime training job requires.
- Use a Slime revision that includes [THUDM/slime#2272](https://github.com/THUDM/slime/pull/2272).

## Prepare the Source

Clone the Slime merge commit that added streaming external rollouts:

```bash
git clone https://github.com/THUDM/slime.git
git -C slime checkout 4c1ab40203952b3dcc8582b653f3a83f2c6e8128
test -f slime/slime/rollout/sglang_streaming_rollout.py
```

Install Slime with the [upstream instructions](https://github.com/THUDM/slime/blob/4c1ab40203952b3dcc8582b653f3a83f2c6e8128/README.md). Keep the checkout path for `SLIME_HOME`.

Clone Dynamo to get the deployment example:

```bash
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo
```

## Deploy the Worker Set

Set the Dynamo images to the same version. The deployment uses `Qwen/Qwen3-0.6B` by default.

```bash
export KUBE_CONTEXT=<cluster-context>
export NAMESPACE=<namespace>
export DYNAMO_FRONTEND_IMAGE=nvcr.io/nvidia/ai-dynamo/dynamo-frontend:<version>
export SGLANG_RUNTIME_IMAGE=nvcr.io/nvidia/ai-dynamo/sglang-runtime:<version>
export DYNAMO_RUNTIME_VERSION=<version>
export MODEL_PATH=Qwen/Qwen3-0.6B
examples/rl/slime/deploy-dynamo.sh
```

Each worker Pod runs SGLang and loads `dynamo.sglang.sidecar` through the SGLang `--sidecar` option.

The manifest creates these Services:

- `slime-sglang-rollout:8000` exposes the Dynamo frontend.
- `slime-sglang-engine-0:30000` exposes the first native SGLang API.
- `slime-sglang-engine-1:30000` exposes the second native SGLang API.

Each engine Service selects one worker component. Keep each worker component at one replica.

Restrict access to the engine Services. These Services expose administrative and weight-update APIs.

## Make Sure That the Engines Are Ready

Run these commands from the Slime environment or another Pod in the deployment namespace:

```bash
curl --fail-with-body http://slime-sglang-engine-0:30000/health_generate
curl --fail-with-body http://slime-sglang-engine-1:30000/health_generate
curl --fail-with-body http://slime-sglang-rollout:8000/health
```

The first two commands show that the SGLang engines are ready. The third command shows that the Dynamo frontend is ready.

## Run Slime

Set the fixed engine addresses and the Dynamo frontend address. Then add the arguments that your training workload requires.

```bash
export SLIME_HOME=<path-to-slime>
export DYNAMO_ENGINE_ADDRS="slime-sglang-engine-0:30000 slime-sglang-engine-1:30000"
export DYNAMO_ROLLOUT_URL="http://slime-sglang-rollout:8000"
examples/rl/slime/launch-slime.sh \
  --hf-checkpoint Qwen/Qwen3-0.6B \
  --prompt-data <prompt-data> \
  --input-key prompt \
  --rm-type random \
  --num-rollout 1 \
  --rollout-batch-size 2 \
  --n-samples-per-prompt 2 \
  --rollout-max-response-len 128 \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 1
```

The launcher passes the fixed worker list to Slime. It also enables incremental streaming output.

The `dynamo_generate.generate_streaming` adapter sends generation to `DYNAMO_ROLLOUT_URL`. Slime calls each fixed engine for control operations and weight updates.

## Make Sure That the Integration Works

Complete these steps before you increase the workload size:

1. Run one rollout with the selected model and weight transport.
2. Apply one policy update to both engines.
3. Run one rollout after the update.
4. Stop one worker during a test run.
5. Make sure that Slime stops or reports the worker failure.
6. Restart the Slime job after Kubernetes replaces the worker Pod.

The integration does not recover a replaced external engine during a Slime job.

## Current Limitations

- The example uses a fixed worker set.
- The example does not support dynamic endpoint registration.
- Slime must restart after Kubernetes replaces an engine Pod.
- The native engine Services require network isolation.
- The example does not provide a fleet-wide policy-update transaction or automatic rollback.

## Upstream Resources

- [Dynamo Slime example](https://github.com/ai-dynamo/dynamo/blob/main/examples/README.md#integration-examples)
- [Slime streaming external-rollout change](https://github.com/THUDM/slime/pull/2272)
- [Slime external rollout engine guide](https://github.com/THUDM/slime/blob/4c1ab40203952b3dcc8582b653f3a83f2c6e8128/docs/en/advanced/external-rollout-engines.md)
- [Shared RL integration reference](integration-reference.md)
