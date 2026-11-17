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

All commands run from `examples/gpu-sharing` in a checkout of the [Dynamo repository](https://github.com/ai-dynamo/dynamo). The files this guide uses are in [`examples/gpu-sharing/one-frontend-many-models/`](https://github.com/ai-dynamo/dynamo/tree/main/examples/gpu-sharing/one-frontend-many-models).

> [!NOTE]
> This guide is experimental. It deploys two models on one GPU, which is the configuration that was tested; see [What Was Tested](#what-was-tested). The way the frontend finds the workers of its DGD was observed on Dynamo 1.5.0 and 1.6.2, not taken from a documented contract, and may change in a later release; see [How the Frontend Finds Its Workers](#how-the-frontend-finds-its-workers).

## Requirements

> [!IMPORTANT]
> Set up GPU fractions first. Follow [Build the KAI-Scheduler and GPU Fractioning Forks](build-gpu-fractioning-forks.md) from start to finish, including its two-pod smoke test, before you start this guide. Nothing below works until that smoke test passes.

The workers are GPU fractions, so the cluster needs the same scheduler stack as the [KAI + GPU fractions experiment](deploy-kai-gpu-fractions.md): KAI-Scheduler and kai-gpu-fractioning built from the forks, on a node with driver r615 or newer. The build guide covers the node requirements, the driver, both installs and the smoke test; none of that is repeated here.

This guide differs from the other experiments in one requirement:

| Requirement | This guide | Why |
|---|---|---|
| Dynamo platform (operator) | 1.6.2 | The other guides pin 1.4.2 |
| Runtime image | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.2` | The other guides pin `vllm-runtime:1.3.0` |

> [!WARNING]
> The shared scripts default to the older versions: `common/install-dynamo-platform.sh` installs platform 1.4.2 and `common/download-model.sh` pulls `vllm-runtime:1.3.0` unless told otherwise. The commands below set `PLATFORM_VERSION` and `RUNTIME_IMAGE` explicitly; if you drop those variables you silently get the old versions. [What Was Tested](#what-was-tested) lists the versions this guide was run with.

Pick the GPU node and export its name; every later step uses it:

```bash
kubectl get nodes -L nvidia.com/gpu.product,nvidia.com/gpu.count
export GPU_NODE=<node-name>
```

## Step 1: Install the Dynamo Platform

```bash
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

Each worker takes `GPU_COMPUTE` of a GPU's streaming multiprocessors (SMs) and `GPU_MEMORY_MI` of its memory. The defaults are half the SMs and a 19,968 MiB memory cap per worker, so the two workers the generator renders use all the SMs of one GPU. That is the whole memory of an A100 40GB, and about half of the A100 80GB this guide was tested on:

| Card | Two workers at the defaults | Note |
|---|---|---|
| A100 80GB (tested) | All SMs, about half the memory | To give each worker more KV cache, raise `GPU_MEMORY_MI`, for example to `39936`. Not measured |
| A100 40GB | All SMs, all the memory | Same caps as the [KAI + GPU fractions experiment](deploy-kai-gpu-fractions.md), which ran 16 of these workers two per GPU on 8x A100 40GB. This generator was not run on a 40GB card |

```bash
NODE_NAME="$GPU_NODE" one-frontend-many-models/gen-dgd.sh | kubectl apply -f -
kubectl wait --for=condition=Ready pod -n default -l experiment.nvidia.com/dgd=qwen3-4b-shared --timeout=600s
```

The frontend and both workers were Ready about two minutes after apply in the tested run; the first start also loads the weights from the hostPath cache. `kubectl get dgd -n default` shows the DGD itself.

To use more GPUs, raise `NUM_MODELS` by two per GPU. KAI places each worker pod on a GPU with room, as it does for the one-DGD-per-model layout, which the [KAI + GPU fractions experiment](deploy-kai-gpu-fractions.md) ran across 8 GPUs. This generator, however, was only run on one GPU; a layout that spans several GPUs is expected to work but was not tested.

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

`RESULT: PASS` means the frontend lists every deployed model, every request returned 200, and each worker finished exactly the requests sent to its model. The per-worker counts come from each worker's vLLM metrics endpoint on port 9090, which the script reads from inside the frontend pod. The script exits 0 on `PASS` and 1 otherwise.

If it does not pass:

- **`no running frontend or workers found`** (exit 1 before any check): the pods are not Running yet, or the DGD name or `NAMESPACE` is wrong. Check `kubectl get pods -n default -l experiment.nvidia.com/dgd=qwen3-4b-shared`.
- **`missing` is not `none`**: a worker is Running but the frontend does not list its model, and requests for it return 404. The worker has not finished registering, or it is registered in a different Dynamo namespace (see [How the Frontend Finds Its Workers](#how-the-frontend-finds-its-workers)). Look at the worker's log for the model name it registered and at the frontend's log for the namespace it watches.
- **Fewer than all requests returned 200**: the frontend accepted the request but a worker failed it. Look at that worker's log; the usual causes are the engine still loading, or a request larger than its KV cache.
- **A worker line says `FAIL`**: the worker finished a different number of requests than were sent to its model, so a request went to the wrong worker or a counter could not be read. Re-run once, since a request that was still finishing when the counters were read also shows up here. If it repeats, compare the `--served-model-name` in each worker's args with the model names the frontend lists.

## Step 5: Call a Model

The frontend Service is a ClusterIP. Forward it and select a model by name:

```bash
kubectl port-forward -n default svc/qwen3-4b-shared-frontend 8000:8000 &
PF_PID=$!
curl -s localhost:8000/v1/models | jq -r '.data[].id'
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-4b-02","messages":[{"role":"user","content":"Say hello."}],"max_tokens":32}'
```

The first call prints one model name per line, `qwen3-4b-01` and `qwen3-4b-02`. The second returns a JSON chat completion whose `model` field is `qwen3-4b-02` and whose `choices[0].message.content` is the reply. A request that names a model the frontend does not list returns 404. Stop the port-forward with `kill $PF_PID` when you are done.

## How the Frontend Finds Its Workers

Two kinds of namespace are involved. The Kubernetes namespace is where the DGD's pods run, `default` in this guide. A Dynamo namespace is a name in Dynamo's service discovery under which workers register and frontends look for them; it is not a Kubernetes object. This section is about the Dynamo namespace.

Normally a worker sets its Dynamo namespace itself through the `DYN_NAMESPACE` environment variable. The generated workers set none. Instead, the operator does it: it sets `DYN_NAMESPACE_PREFIX` on the frontend to `<k8s-namespace>-<dgd-name>`, here `default-qwen3-4b-shared`, and registers the workers of the DGD under `default-qwen3-4b-shared-<8 hex>`. The suffix identifies the DGD's current worker generation, the set of worker pods created by the current version of the DGD spec, so a rolled-out change gets a fresh suffix. The frontend watches for namespaces with its prefix and lists the models of every worker it finds there. See `DYN_NAMESPACE_PREFIX` in the [Kubernetes API reference](../../../docs/fern/pages/reference/kubernetes-api/additional-resources/api-reference-k8s.md).

This is observed behaviour of the operator and runtime on 1.5.0 and 1.6.2, not a documented contract, and it may change. Two consequences were checked:

- **A frontend lists only the models of its own DGD.** With two DGDs that each have a frontend, each frontend lists its own models and returns 404 for the other's. To put a model behind this frontend, add it as a worker in the same DGD (checked on 1.5.0 and 1.6.2).
- **A DGD can declare more than one frontend component.** Each frontend gets its own Service and serves all the models of the DGD; the frontends do not divide the models between them (checked on 1.5.0 and 1.6.2).

## What Was Tested

The steps on this page were run as written, with the files in this directory, on one A100 80GB with driver 615.71.09 and the fork builds of KAI-Scheduler and kai-gpu-fractioning. With Dynamo platform and runtime 1.6.2, the two workers and the frontend were Running after about two minutes, the frontend listed both models, and `verify-routing.sh` passed: 7 of 7 requests returned 200 and were counted by the worker of the model they named. The same deployment passed with Dynamo 1.5.0.

The generator was also run with more workers on that GPU, using smaller shares per worker: four Qwen3-4B models, and a larger number of Qwen3-0.6B models. The frontend listed every model and routed every request correctly in those runs too.

Not tested: an A100 40GB, and any layout that spans several GPUs. The other pages that mention versions for this layout point here.

## Clean Up

```bash
kill "$PF_PID" 2>/dev/null   # the port-forward from Step 5, if it is still running
kubectl delete dgd qwen3-4b-shared -n default
kubectl delete job download-qwen3-4b -n default --ignore-not-found
```

To remove the Dynamo platform, run `helm uninstall dynamo-platform -n dynamo-system`. To remove the forks, follow the [uninstall steps](build-gpu-fractioning-forks.md#uninstall).

## Next Steps

- [GPU Sharing](../../../docs/fern/pages/use-cases/gpu-sharing/overview.mdx) compares dedicated GPUs, HAMi and GPU fractions for many-model serving.
- [Deploy the KAI + GPU Fractions Experiment](deploy-kai-gpu-fractions.md) runs the one-DGD-per-model layout with the same workers.
- [Per-GPU Compute and Memory Limits with MPS](mps-per-gpu-limits.md) explains the mechanism behind the per-worker caps.
