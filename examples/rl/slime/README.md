<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Slime External Rollouts with Dynamo

**Experimental.** Run Slime's external SGLang rollout mode against a Dynamo SGLang deployment. Dynamo receives all rollout generation through one frontend and selects a worker for each request. Slime receives only ordinary SGLang control URLs for the current ready workers.

The example uses a headless control Service to resolve worker Pod IPs. `dynamo_discovery.discover_engine_control_urls` checks each Dynamo system server's `/health` response and returns ready control bases such as `http://10.0.0.42:9090/engine`. Slime calls the function synchronously at initialization and before every weight update, then appends its standard SGLang paths such as `/server_info`, `/pause_generation`, and `/update_weights_from_distributed`.

## Prerequisites

- Install the Dynamo Kubernetes Platform, with a GPU-capable Kubernetes cluster.
- Use a Dynamo image containing the SGLang-native `POST /generate` frontend facade.
- Use a Slime revision that includes `--rollout-external-rollout-url` and `--rollout-external-dynamic-discovery-path`.
- Run the Slime client in the same Kubernetes namespace as the deployed graph, or configure DNS and network policy so it can reach the frontend and worker system ports.
- Make `examples/rl/slime` available in the Slime client environment, for example by mounting the Dynamo source tree or copying this directory into the client image.
- Install `envsubst` on the machine that deploys the manifest.

## Deploy Dynamo

Set the image that contains the SGLang `/generate` implementation. The default model is the public `Qwen/Qwen3-0.6B` checkpoint; set `MODEL_PATH` to use a model available to the worker Pods.

```bash
export KUBE_CONTEXT=<cluster-context>
export NAMESPACE=<namespace>
export DYNAMO_IMAGE=<registry>/sglang-runtime:<tag>
export MODEL_PATH=Qwen/Qwen3-0.6B
examples/rl/slime/deploy-dynamo.sh
```

The manifest creates two Services:

- `slime-sglang-rollout:8000` is the shared Dynamo data-plane frontend. It exposes the native SGLang `POST /generate` route.
- `slime-sglang-control:9090` is a headless Service that resolves current worker Pod IPs. It is only for direct worker administration.

The SGLang worker allowlist maps generic engine methods to Dynamo's `/engine` compatibility facade:

```text
server_info=get_server_info pause_generation:tm flush_cache init_weights_update_group update_weights_from_distributed:tm destroy_weights_update_group continue_generation:tm
```

Do not expose `slime-sglang-control` outside the training network. `/engine` methods are administrative APIs.

## Verify the Data and Control Planes

Run these commands from a Pod in the same namespace as the Dynamo graph. The health query must return one direct `/engine` base for each ready worker.

```bash
curl --fail-with-body http://slime-sglang-rollout:8000/health

PYTHONPATH=examples/rl/slime python3 -c \
  'from dynamo_discovery import discover_engine_control_urls; print(discover_engine_control_urls(None))'
```

The second command returns values shaped like:

```text
['http://10.0.0.42:9090/engine', 'http://10.0.0.43:9090/engine']
```

## Start Slime

`launch-slime.sh` adds this directory to `PYTHONPATH` and provides only the external-rollout flags. Append the model, dataset, trainer, and resource arguments required by your Slime workload.

Slime takes the frontend origin and sends rollout requests to its `/generate` path. It does not receive Dynamo routing state or make Dynamo-specific engine calls.

```bash
export SLIME_HOME=<path-to-slime>
export DYNAMO_ROLLOUT_URL=http://slime-sglang-rollout:8000
export DYNAMO_CONTROL_SERVICE=slime-sglang-control
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

Set `DYNAMO_CONTROL_PORT` when the worker system port differs from `9090`. Set `DYNAMO_CONTROL_HEALTH_TIMEOUT_SECONDS` to change the per-worker readiness timeout.

## Lifecycle

1. Slime calls the discovery function before creating its external engine controllers.
2. The function resolves the headless Service and returns only workers whose Dynamo `/health` payload reports `status: ready`.
3. Slime sends every rollout request to the shared frontend's `/generate` route; Dynamo routes that request to a selected healthy worker.
4. Before a weight update, Slime calls discovery again. It then uses each returned `/engine` base for the normal SGLang pause, flush, update, and continue methods.

The health filter prevents newly created or draining Pods from joining a Slime weight-update group before their Dynamo serving endpoint is ready. The frontend remains the only rollout data-plane endpoint, so scaling or replacement does not require Slime to select generation workers.
