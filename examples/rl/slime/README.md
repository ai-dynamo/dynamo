<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Slime External Rollouts with Dynamo

**Experimental.** This example runs Slime with two SGLang engines in a fixed
worker set. A DynamoGraphDeployment manages the engines. Each worker Pod runs
SGLang and the Dynamo sidecar in one runtime container. Slime connects through
stable Kubernetes Services. Slime uses the Dynamo frontend for incremental
streaming responses.

This example uses the external-engine and streaming support from
[THUDM/slime#2272](https://github.com/THUDM/slime/pull/2272). Slime queries
`/server_info` on each engine. Slime registers the fixed addresses with its
SGLang router. Slime calls the native SGLang control and weight-update endpoints.
The included custom generator sends rollout requests to the Dynamo frontend.

> [!IMPORTANT]
> Slime starts an SGLang router to track the external engines. The custom
> generator does not send rollout requests to that router.
>
> Keep each worker component at one replica. If you change the worker set,
> restart Slime.

## Prerequisites

- Install the Dynamo Kubernetes Platform on a GPU-capable Kubernetes cluster.
- Install the NVIDIA device plugin for the `nvidia.com/gpu` resource.
- Use a Dynamo SGLang runtime image that contains `dynamo.sglang.sidecar`.
- For a gated model, create an `hf-token-secret` secret that contains
  `HF_TOKEN` in the deployment namespace. The public default model does not
  require this secret.
- Use a Slime revision that contains
  [THUDM/slime#2272](https://github.com/THUDM/slime/pull/2272).
- Run Slime in a location that can resolve and reach the worker Services. The
  same Kubernetes namespace meets this requirement.
- Install `envsubst` on the machine that deploys the manifest.

## Deploy the fixed worker set

Set the Dynamo frontend image and the SGLang runtime image to the same Dynamo
version. Set `DYNAMO_RUNTIME_VERSION` to that version.

The script uses `MODEL_PATH=Qwen/Qwen3-0.6B` by default.

```bash
export KUBE_CONTEXT=<cluster-context>
export NAMESPACE=<namespace>
export DYNAMO_FRONTEND_IMAGE=nvcr.io/nvidia/ai-dynamo/dynamo-frontend:<version>
export SGLANG_RUNTIME_IMAGE=nvcr.io/nvidia/ai-dynamo/sglang-runtime:<version>
export DYNAMO_RUNTIME_VERSION=<version>
export MODEL_PATH=Qwen/Qwen3-0.6B
examples/rl/slime/deploy-dynamo.sh
```

If you use an unreleased version, pin nightly tags from the same date.

SGLang loads `dynamo.sglang.sidecar` through its `--sidecar` option. It passes
the local gRPC endpoint to the module through `--sidecar-args`. This path does
not require a separate Dynamo sidecar image.

Each worker Pod requests one `nvidia.com/gpu` resource. The worker Pods also
tolerate the standard `nvidia.com/gpu=true:NoSchedule` GPU-node taint.

Add only the node selectors and tolerations that your cluster administrator
assigns to your workload.

The manifest creates three Services:

- `slime-sglang-rollout:8000` exposes the Dynamo frontend for rollout requests.
- `slime-sglang-engine-0:30000` exposes the first native SGLang control API.
- `slime-sglang-engine-1:30000` exposes the second native SGLang control API.

The `engine-*` names do not conflict with the operator-owned worker Services.
The operator uses those worker Services for Dynamo discovery on port 9090.

Each engine Service selects one worker component with one replica. The Service
name stays stable after Kubernetes replaces its Pod. Slime does not recover an
external engine automatically.

If Kubernetes restarts or replaces an engine Pod, restart the Slime job.

The native engine Services expose administrative APIs and weight-update APIs.

Restrict access to these Services to the training network.

## Engine Readiness

Run these commands from the Slime environment or another Pod in the deployment
namespace:

```bash
curl --fail-with-body http://slime-sglang-engine-0:30000/health_generate
curl --fail-with-body http://slime-sglang-engine-1:30000/health_generate
curl --fail-with-body http://slime-sglang-rollout:8000/health
```

The first two commands make sure that both fixed engines are reachable. The
third command makes sure that the Dynamo frontend is ready.

## Start Slime

`launch-slime.sh` supplies the fixed external-engine list. It uses the streaming
generator from THUDM/slime#2272.

Add the model, dataset, trainer, weight-update, and resource arguments that your
workload requires.

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

The launcher passes these integration arguments:

```text
--rollout-external-engine-addrs <worker-0> <worker-1>
--rollout-function-path slime.rollout.sglang_rollout.generate_rollout
--custom-generate-function-path dynamo_generate.generate_streaming
--sglang-incremental-streaming-output
```

The `dynamo_generate.generate_streaming` adapter copies the Slime arguments.
Then the adapter sets the generation address to `DYNAMO_ROLLOUT_URL`. The
adapter calls the streaming generator from THUDM/slime#2272. Slime calls each
fixed engine for control operations and weight updates.

The `--sglang-incremental-streaming-output` flag must match the
`--incremental-streaming-output` configuration in `dynamo.yaml`.

Read the Slime
[external rollout engine guide](https://github.com/THUDM/slime/blob/main/docs/en/advanced/external-rollout-engines.md)
for NCCL updates, full-checkpoint disk updates, and delta disk updates.

## Current Boundary

This example uses a fixed worker set. It does not support elastic discovery,
dynamic endpoint registration, or external-engine fault recovery.

Before you use the example for training, complete these tests:

1. Run one rollout with the selected model and weight transport.
2. Apply one policy update.
3. Run one post-update rollout.
4. Simulate one worker failure.
5. Make sure that the system recovers.
