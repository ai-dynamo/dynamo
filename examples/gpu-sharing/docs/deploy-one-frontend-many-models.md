<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Serve Many Models Behind One Frontend

_Run many models as GPU fractions in a single DynamoGraphDeployment, with one frontend that lists them all and routes each request to the right worker_

The other experiments in [GPU Sharing](../../../docs/fern/pages/use-cases/gpu-sharing/overview.mdx) give every model its own `DynamoGraphDeployment` (DGD) and its own frontend, so every model has its own endpoint. This guide deploys the same kind of workers in one DGD behind a single frontend. Clients use one endpoint and select a model by name.

| | One DGD per model (the other guides) | One DGD, one frontend (this page) |
|---|---|---|
| DGDs for N models | N | 1 |
| Frontends and endpoints | N | 1 |
| How a client picks a model | By endpoint | By the `model` field of the request |
| Worker pods | One per model, a GPU fraction each | The same |

All commands run from `examples/gpu-sharing` in a checkout of the [Dynamo repository](https://github.com/ai-dynamo/dynamo). The files this guide uses are in [`examples/gpu-sharing/one-frontend-many-models/`](https://github.com/ai-dynamo/dynamo/tree/main/examples/gpu-sharing/one-frontend-many-models).

> [!NOTE]
> This guide is experimental. It deploys two models on one GPU, which is the configuration that was tested; see [What Was Tested](#what-was-tested).

## Requirements

> [!IMPORTANT]
> Set up GPU fractions first. Follow [Build the KAI-Scheduler and GPU Fractioning Forks](build-gpu-fractioning-forks.md) from start to finish, including its two-pod smoke test, before you start this guide. Nothing below works until that smoke test passes.

The workers are GPU fractions, so the cluster needs the same scheduler stack as the [KAI + GPU fractions experiment](deploy-kai-gpu-fractions.md): KAI-Scheduler and kai-gpu-fractioning built from the forks, on a node with driver r615 or newer. The build guide covers the node requirements, the driver, both installs and the smoke test; none of that is repeated here.

This guide differs from the other experiments in one requirement:

| Requirement | This guide | Why |
|---|---|---|
| Dynamo platform (operator) | 1.6.2 (also tested with 1.5.0) | The other guides pin 1.4.2 |
| Runtime image | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.2` (also tested with 1.5.0) | The other guides pin `vllm-runtime:1.3.0` |

Pick the GPU node and export its name; every later step uses it:

```bash
kubectl get nodes -L nvidia.com/gpu.product,nvidia.com/gpu.count
export GPU_NODE=<node-name>
```

## Step 1: Install the Dynamo Platform

```bash
cd examples/gpu-sharing
PLATFORM_VERSION=1.6.2 common/install-dynamo-platform.sh
```

The script installs the `dynamo-platform` Helm chart into the `dynamo-system` namespace and waits for its pods to be Ready.

## Step 2: Download the Model to the GPU Node

```bash
NODE_NAME="$GPU_NODE" RUNTIME_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.2 common/download-model.sh
```

This downloads `Qwen/Qwen3-4B` at the pinned revision into the `hostPath` cache `/opt/hf-cache` on the GPU node. Every worker mounts that cache read-only and serves the same weights.

## Step 3: Deploy One DGD With Many Models

[`one-frontend-many-models/gen-dgd.sh`](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/one-frontend-many-models/gen-dgd.sh) renders a single DGD with one frontend and `NUM_MODELS` workers. Each worker serves the weights under its own name, `qwen3-4b-01`, `qwen3-4b-02`, and so on, set with `--served-model-name`.

Each worker takes `GPU_COMPUTE` of a GPU's streaming multiprocessors (SMs) and `GPU_MEMORY_MI` of its memory. With the defaults that is half of an A100 40GB, and the generator renders two workers, which fill one GPU:

```bash
NODE_NAME="$GPU_NODE" one-frontend-many-models/gen-dgd.sh | kubectl apply -f -
kubectl get dgd -n default      # wait until qwen3-4b-shared is Ready
```

To use more GPUs, set `NUM_MODELS` to two per A100 40GB.

| Variable | Default | Purpose |
|---|---|---|
| `NUM_MODELS` | `2` | Number of workers, one model each |
| `DGD_NAME` | `qwen3-4b-shared` | Name of the DGD. The frontend Service is `<DGD_NAME>-frontend` |
| `MODEL_NAME_PREFIX` | `qwen3-4b` | Models are served as `<prefix>-01` to `<prefix>-NN` |
| `GPU_COMPUTE` | `0.5` | Fraction of a GPU's SMs per worker |
| `GPU_MEMORY_MI` | `19968` | GPU memory cap per worker, in MiB |
| `GPU_MEMORY_UTILIZATION` | `0.80` | vLLM `--gpu-memory-utilization`, as a fraction of the cap |
| `WORKER_CPU_REQUEST`, `WORKER_MEMORY_REQUEST` | `2`, `12Gi` | Host CPU and memory request per worker |
| `RUNTIME_IMAGE` | `vllm-runtime:1.6.2` | Image for the frontend and the workers |
| `NAMESPACE`, `HF_CACHE_DIR`, `NODE_NAME` | `default`, `/opt/hf-cache`, unset | As in the other generators |

The generator takes an optional model snapshot path as its first argument, to serve different weights. The checked-in [`one-frontend-many-models/dgd-2-models.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/one-frontend-many-models/dgd-2-models.yaml) is the default output.

Each worker sets its share of the GPU with per-container annotations instead of `gpu-fraction`:

```yaml
annotations:
  nvidia.com/container.main.gpu-compute.request: "0.5"
  nvidia.com/container.main.gpu-memory.request: 19968Mi
  nvidia.com/container.main.gpu-memory.limit: 19968Mi
  nvidia.com/container.main.gpu-compute.mode: sm-sharing
```

With an explicit memory limit, vLLM sees the limit as the GPU's total memory. `--gpu-memory-utilization` is therefore a fraction of the cap, unlike the `0.40` of the physical card that the [KAI + GPU fractions experiment](deploy-kai-gpu-fractions.md) uses.

> [!WARNING]
> The warning in [Deploy the KAI + GPU Fractions Experiment](deploy-kai-gpu-fractions.md#step-4-deploy-the-16-dgds) about editing a running DGD on a fully packed node applies here too. To change a worker spec, run `kubectl delete dgd qwen3-4b-shared` and apply again.

## Step 4: Verify Models and Routing

All pods of the DGD should be Running: one frontend and `NUM_MODELS` workers.

```bash
kubectl get pods -n default -l experiment.nvidia.com/dgd=qwen3-4b-shared
```

Then run [`one-frontend-many-models/verify-routing.sh`](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/one-frontend-many-models/verify-routing.sh):

```bash
one-frontend-many-models/verify-routing.sh
```

The script lists the models on the frontend, sends a different number of chat requests to each model in parallel, and compares that with the counter `vllm:request_success_total` on each worker before and after. A request for a model that does not exist must return 404. It runs inside the frontend pod, so it needs only `kubectl` on your workstation. With two models it prints:

```text
frontend lists 2 models; 2 workers deployed; missing: none
sent 7 requests, 7 returned 200; unknown model returned 404
  qwen3-4b-01: worker finished +3, sent 3  PASS
  qwen3-4b-02: worker finished +4, sent 4  PASS
RESULT: PASS
```

`RESULT: PASS` means the frontend lists every deployed model, every request returned 200, and each worker finished exactly the requests sent to its model. If `missing` is not `none`, the frontend does not list a model that has a running worker, and requests for that model return 404.

## Step 5: Call a Model

The frontend Service is a ClusterIP. Forward it and select a model by name:

```bash
kubectl port-forward -n default svc/qwen3-4b-shared-frontend 8000:8000 &
curl -s localhost:8000/v1/models | python3 -c 'import json,sys; print([m["id"] for m in json.load(sys.stdin)["data"]])'
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-4b-02","messages":[{"role":"user","content":"Say hello."}],"max_tokens":32}'
```

## How the Frontend Finds Its Workers

The operator sets `DYN_NAMESPACE_PREFIX` on the frontend to `<k8s-namespace>-<dgd-name>`, here `default-qwen3-4b-shared`. It registers the workers of the DGD in the Dynamo namespace `default-qwen3-4b-shared-<8 hex>`, the DGD's worker generation, which is why the generated workers set no `DYN_NAMESPACE`. The frontend discovers that namespace and lists the models of its workers. See `DYN_NAMESPACE_PREFIX` in the [Kubernetes API reference](../../../docs/fern/pages/reference/kubernetes-api/additional-resources/api-reference-k8s.md).

What was observed around that:

- **A frontend lists only the models of its own DGD.** With two DGDs that each have a frontend, each frontend lists its own models and returns 404 for the other's (checked on 1.5.0).
- **A DGD can declare more than one frontend component.** Each frontend gets its own Service and serves all the models of the DGD; the frontends do not divide the models between them (checked on 1.5.0 and 1.6.2).

## What Was Tested

The steps on this page were run as written, with the files in this directory, on one A100 80GB with driver 615.71.09 and the fork builds of KAI-Scheduler and kai-gpu-fractioning. With Dynamo platform and runtime 1.6.2, the two workers and the frontend were Running after about two minutes, the frontend listed both models, and `verify-routing.sh` passed: 7 of 7 requests returned 200 and were counted by the worker of the model they named. The same deployment passed with Dynamo 1.5.0.

The generator was also run with more workers on that GPU, using smaller shares per worker: four Qwen3-4B models, and a larger number of Qwen3-0.6B models. The frontend listed every model and routed every request correctly in those runs too. Layouts that span several GPUs were not tested.

## Known Issues

- **A frontend serves only the models of its own DGD.** To put a model behind this frontend, add it as a worker in the same DGD. A model deployed in a separate DGD is not served by it; from Dynamo 1.6.0 this is by design.

## Clean Up

```bash
kubectl delete dgd qwen3-4b-shared -n default
kubectl delete job download-qwen3-4b -n default --ignore-not-found
```

To remove the Dynamo platform, run `helm uninstall dynamo-platform -n dynamo-system`. To remove the forks, follow the [uninstall steps](build-gpu-fractioning-forks.md#uninstall).

## Next Steps

- [GPU Sharing](../../../docs/fern/pages/use-cases/gpu-sharing/overview.mdx) compares dedicated GPUs, HAMi and GPU fractions for many-model serving.
- [Deploy the KAI + GPU Fractions Experiment](deploy-kai-gpu-fractions.md) runs the one-DGD-per-model layout with the same workers.
- [Per-GPU Compute and Memory Limits with MPS](mps-per-gpu-limits.md) explains the mechanism behind the per-worker caps.
