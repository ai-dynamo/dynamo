<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Slime External Rollouts with Dynamo

**Experimental.** Run Slime against a fixed pair of SGLang engines managed by a
DynamoGraphDeployment. Each worker Pod runs SGLang and the Dynamo sidecar in one
runtime container. Slime connects through stable Kubernetes Services and
consumes incremental streaming responses.

This example uses the external-engine and streaming support from
[THUDM/slime#2272](https://github.com/THUDM/slime/pull/2272). Slime queries
`/server_info` on each engine. It registers the fixed addresses with its own
SGLang router. It calls the native SGLang control and weight-update endpoints.
The same engines register with Dynamo and remain available through the Dynamo
frontend.

> [!IMPORTANT]
> Slime rollout requests use the Slime router and the native engine Services in
> this version of the example. They do not pass through the Dynamo frontend.
> Dynamic registration between the Dynamo worker set and Slime is not yet
> available. Keep the two worker components at one replica each, and restart
> Slime after changing the worker set.

## Prerequisites

- Install the Dynamo Kubernetes Platform on a GPU-capable Kubernetes cluster.
- Install the NVIDIA device plugin. The worker Pods request the
  `nvidia.com/gpu` resource.
- Use a Dynamo SGLang runtime image that contains `dynamo.sglang.sidecar`.
- For a gated model, create an `hf-token-secret` secret that contains
  `HF_TOKEN` in the deployment namespace. The public default model does not
  require this secret.
- Use a Slime revision that contains
  [THUDM/slime#2272](https://github.com/THUDM/slime/pull/2272).
- Run Slime where it can resolve and reach the worker Services, normally in the
  same Kubernetes namespace.
- Install `envsubst` on the machine that deploys the manifest.

## Deploy the Fixed Worker Set

Set matching Dynamo frontend and SGLang runtime images. Set
`DYNAMO_RUNTIME_VERSION` to the matching release version. The script defaults
to `MODEL_PATH=Qwen/Qwen3-0.6B`.

```bash
export KUBE_CONTEXT=<cluster-context>
export NAMESPACE=<namespace>
export DYNAMO_FRONTEND_IMAGE=nvcr.io/nvidia/ai-dynamo/dynamo-frontend:<version>
export SGLANG_RUNTIME_IMAGE=nvcr.io/nvidia/ai-dynamo/sglang-runtime:<version>
export DYNAMO_RUNTIME_VERSION=<version>
export MODEL_PATH=Qwen/Qwen3-0.6B
examples/rl/slime/deploy-dynamo.sh
```

If you test an unreleased version, pin matching dated nightly tags for
reproducible runs.

SGLang loads `dynamo.sglang.sidecar` through its `--sidecar` option. It passes
the local gRPC endpoint to the module through `--sidecar-args`. This path does
not require a separate Dynamo sidecar image.

Each worker Pod requests one `nvidia.com/gpu` resource. The worker Pods also
tolerate the standard `nvidia.com/gpu=true:NoSchedule` GPU-node taint. Add only
the site-specific node selectors and tolerations that your cluster
administrator assigns to your workload.

The manifest creates three Services:

- `slime-sglang-rollout:8000` exposes the Dynamo frontend for independent
  Dynamo requests and smoke tests.
- `slime-sglang-engine-0:30000` exposes the first native SGLang engine to
  Slime.
- `slime-sglang-engine-1:30000` exposes the second native SGLang engine to
  Slime.

The `engine-*` names do not conflict with the operator-owned worker Services.
The operator uses those worker Services for Dynamo discovery on port 9090.

Each engine Service selects one worker component with one replica. The Service
name stays stable after Kubernetes replaces its Pod. Slime does not recover an
external engine automatically. After an engine restart or replacement, restart
the Slime job.

The native engine Services expose generation and administrative APIs. Restrict
them to the training network.

## Validate the Engines

Run these validation commands from the Slime environment or another Pod in the
deployment namespace:

```bash
curl --fail-with-body http://slime-sglang-engine-0:30000/health_generate
curl --fail-with-body http://slime-sglang-engine-1:30000/health_generate
curl --fail-with-body http://slime-sglang-rollout:8000/health
```

The first two commands validate access to both fixed engines. The third command
validates that the Dynamo frontend is ready.

## Start Slime

`launch-slime.sh` supplies the fixed external-engine list and enables the
streaming generator merged in THUDM/slime#2272. Append the model, dataset,
trainer, weight-update, and resource arguments required by your workload.

```bash
export SLIME_HOME=<path-to-slime>
export DYNAMO_ENGINE_ADDRS="slime-sglang-engine-0:30000 slime-sglang-engine-1:30000"
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
--custom-generate-function-path slime.rollout.sglang_streaming_rollout.generate_streaming
--sglang-incremental-streaming-output
```

The `--sglang-incremental-streaming-output` flag must match the
`--incremental-streaming-output` setting in `dynamo.yaml`. See the Slime
[external rollout engine guide](https://github.com/THUDM/slime/blob/main/docs/en/advanced/external-rollout-engines.md)
for NCCL, full-checkpoint disk, and delta disk weight-update options.

## Current Boundary

This example demonstrates a fixed external fleet. It does not provide elastic
discovery, dynamic router registration, or external-engine fault recovery.
Before you use the example for training, validate one complete rollout, policy
update, post-update rollout, and worker-failure path. Use the selected model and
weight transport for this validation.
