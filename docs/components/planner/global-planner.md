---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Global Planner Deployment Guide
---

This guide explains how to deploy `GlobalPlanner` and when to use it. `GlobalPlanner` is the centralized scaling execution layer for deployments where multiple DGDs should delegate scaling through one component, whether those DGDs expose separate endpoints or sit behind one shared endpoint.

## Why Global Planner?

Without `GlobalPlanner`, each DGD's local planner scales only its own deployment directly. That is fine for isolated deployments, but it becomes awkward when you want one place to:

- apply centralized scaling policy across multiple DGDs
- enforce shared constraints such as authorization or total GPU budget
- coordinate scaling for a single-endpoint, multi-pool deployment

`GlobalPlanner` solves that by becoming the common scale-execution endpoint for multiple local planners.

## Terminology

- **SLA Planner**: The normal `dynamo.planner` component that computes desired replica counts to maintain SLAs.
- **Local Planner**: A pool-local instance of a SLA planner inside one DGD.
- **Global Planner**: The centralized execution and policy layer that receives scale requests from local planners.
- **Single-endpoint multi-pool deployment**: One model endpoint backed by multiple DGDs for the same model. This pattern uses both `GlobalRouter` and `GlobalPlanner`.

## Deployment Patterns

Use `GlobalPlanner` in one of these two patterns:

| Pattern | Use when | Needs `GlobalRouter` | Public endpoint shape |
|---------|----------|----------------------|-----------------------|
| Multiple model endpoints or independent DGDs | Separate DGDs should share centralized scaling policy, such as authorization or total GPU budget | No | One endpoint per DGD, or however each DGD is exposed |
| One model endpoint, multiple DGDs | One model should be reachable through one public endpoint, but different request classes should land on different DGDs | Yes | One shared endpoint |

## Pattern 1: Multiple Model Endpoints Or Independent DGDs

Use this pattern when you have multiple DGDs, often for different models, and you want them to share centralized scaling policy without collapsing them into one endpoint.

Typical examples:

- DGD A: `qwen-0.6b` disaggregated deployment with its own local planner
- DGD B: `qwen-32b` disaggregated deployment with its own local planner
- one shared `GlobalPlanner` that all local planners delegate to

In this pattern:

- each DGD keeps its own normal local planner
- each local planner is configured with `environment: "global-planner"`
- all those planners point at the same `global_planner_namespace`
- each DGD keeps its own endpoint or frontend as needed
- you do **not** need `GlobalRouter`

This is the pattern to use when the goal is centralized scaling control across multiple deployments or models.

## Pattern 2: One Model Endpoint, Multiple DGDs

Use this pattern when all of the following are true:

- You want one public endpoint for a single model.
- You want different private pools for different request classes, such as short ISL vs. long ISL requests, or different latency targets.
- You want each pool to autoscale independently.
- You want routing and scale execution to be centralized instead of exposing multiple endpoints to clients.

Typical examples:

- short-input requests are cheaper on a smaller prefill pool
- long-input requests need a larger prefill pool
- decode capacity should scale independently from prefill capacity

If you only need one pool for one model, use a single Local Planner and DGD/DGDR instead.

## What You Deploy

In the current implementation, the single-endpoint pattern is composed from multiple resources:

| Resource | Purpose | Typical contents |
|----------|---------|------------------|
| Control DGD | Public entrypoint and centralized control plane | `Frontend`, `GlobalRouter`, `GlobalPlanner` |
| Prefill pool DGD(s) | Private prefill capacity pools | `LocalRouter`, prefill worker(s), `Planner` |
| Decode pool DGD(s) | Private decode capacity pools | `LocalRouter`, decode worker(s), `Planner` |
| Optional DGDR(s) | Generate or validate one optimized pool shape at a time | Model, workload, SLA, hardware inputs |

> **Current workflow**
>
> A single DGDR does **not** generate the full single-endpoint multi-pool topology today. Instead, run one DGDR or profiling job per intended pool, then compose the final control DGD plus pool DGDs manually.

## Architecture

```text
Client
  |
  v
Frontend (single public endpoint)
  |
  v
GlobalRouter
  |
  +--> Prefill pool 0 Dynamo namespace --> LocalRouter --> Prefill workers --> Pool Planner
  +--> Prefill pool 1 Dynamo namespace --> LocalRouter --> Prefill workers --> Pool Planner
  |
  +--> Decode pool 0 Dynamo namespace  --> LocalRouter --> Decode workers  --> Pool Planner
  +--> Decode pool 1 Dynamo namespace  --> LocalRouter --> Decode workers  --> Pool Planner

Pool Planners
  |
  v
GlobalPlanner
  |
  v
Kubernetes scaling updates on the target DGDs
```

The `Frontend` exposes a single model endpoint. `GlobalRouter` selects the best pool for each request. Each pool-local `Planner` decides how much capacity its own pool needs. `GlobalPlanner` receives those scale requests and applies the Kubernetes replica changes centrally.

## Prerequisites

- Dynamo Kubernetes Platform installed. See [Kubernetes Quickstart](../../kubernetes/README.md).
- Prometheus deployed and scraping router metrics. The global planner examples assume cluster Prometheus is available.
- Backend images available for your chosen framework (`vllm`, `sglang`, or `trtllm`).
- Secrets for model access, such as a Hugging Face token secret.
- A storage strategy for model weights if your workers should share a model cache PVC.

For throughput-based scaling, you also need profiling data for each pool. See [Profiler Guide](../profiler/profiler-guide.md).

## Inputs You Need To Decide Up Front

Before writing manifests, decide the following:

| Input | Why it matters | Example |
|-------|----------------|---------|
| Model name | All pools in one hierarchy serve the same model | `meta-llama/Llama-3.3-70B-Instruct` |
| Backend | Worker args and profiling flow depend on it | `vllm` |
| Pool inventory | Number of specialized prefill and decode pools | 2 prefill pools, 1 decode pool |
| Workload classes | Determines how many pool profiles you generate | short ISL, long ISL, long context decode |
| SLA targets | Guides profiling and routing decisions | `ttft: 200 ms`, `itl: 20 ms` |
| Worker shape | Tensor parallelism, GPUs per worker, and memory footprint | TP1 prefill vs. TP2 prefill |
| Routing policy | Maps requests to pools at runtime | low-ISL requests -> pool 0 |
| Optional global budget | Caps total GPUs across managed pools | `--max-total-gpus 16` |

## Step 1: Profile Each Intended Pool Independently

Start by deciding what each pool should specialize in. Common examples:

- Prefill pool 0: lower-cost pool for short prompts.
- Prefill pool 1: larger pool for long prompts.
- Decode pool 0: standard decode pool for most requests.

For each intended pool, run a separate DGDR or profiling job with the workload and SLA that represent that pool.

Example DGDR skeleton:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: llama-prefill-short
spec:
  model: meta-llama/Llama-3.3-70B-Instruct
  backend: vllm
  image: nvcr.io/nvidia/ai-dynamo/dynamo-frontend:<tag>
  workload:
    isl: 2048
    osl: 256
  sla:
    ttft: 200.0
    itl: 20.0
  searchStrategy: rapid
  autoApply: false
```

Repeat this once per planned pool, changing the workload and SLA inputs for each request class.

What to keep from each profiling result:

- Worker shape (`tensor-parallel-size`, GPUs per worker, memory/caching settings).
- Planner profile data directory or generated ConfigMaps.
- Planner settings such as `prefill_engine_num_gpu` or `decode_engine_num_gpu`.
- Any backend-specific flags that differ across pools.

See [Planner Examples](planner-examples.md) and [Profiler Guide](../profiler/profiler-guide.md) for DGDR details.

## Step 2: Create The Control DGD

Deploy one control DGD that contains:

- `Frontend`: the single public model endpoint.
- `GlobalRouter`: chooses which pool receives each request.
- `GlobalPlanner`: receives scale requests from pool planners and applies replica changes.

The vLLM example topology is in [examples/global_planner/global-planner-vllm-test.yaml](../../../examples/global_planner/global-planner-vllm-test.yaml).

The `GlobalPlanner` section is minimal:

```yaml
GlobalPlanner:
  componentType: default
  replicas: 1
  extraPodSpec:
    mainContainer:
      image: ${DYNAMO_IMAGE}
      command:
        - python3
        - -m
        - dynamo.global_planner
      args:
        - --managed-namespaces
        - ${K8S_NAMESPACE}-gp-prefill-0
        - ${K8S_NAMESPACE}-gp-prefill-1
        - ${K8S_NAMESPACE}-gp-decode-0
```

The values passed to `--managed-namespaces` are the pool planners' **Dynamo namespaces** (`caller_namespace`), not raw Kubernetes namespaces. In many examples they share the same string prefix, but they are logically different identifiers.

**Management modes**: When `--managed-namespaces` is set (explicit mode), only the listed Dynamo namespaces are authorized to send scale requests, and only their corresponding DGDs count toward the GPU budget. DGD names are derived from the Dynamo namespace using the operator convention `DYN_NAMESPACE = {k8s_namespace}-{dgd_name}`. When omitted (implicit mode), any caller is accepted and all DGDs in the Kubernetes namespace count toward the GPU budget.

If you want the central executor to reject scale requests that exceed a total GPU budget, add `--max-total-gpus`. See [examples/global_planner/global-planner-gpu-budget.yaml](../../../examples/global_planner/global-planner-gpu-budget.yaml).

## Step 3: Create One DGD Per Pool

Each private pool gets its own DGD. A pool DGD usually contains:

- `LocalRouter`
- one worker type (`prefill` or `decode`)
- one `Planner`

The planner inside each pool must be configured for `global-planner` mode so it delegates scaling to the control stack:

```json
{
  "environment": "global-planner",
  "global_planner_namespace": "${K8S_NAMESPACE}-gp-ctrl",
  "backend": "vllm",
  "mode": "prefill",
  "enable_load_scaling": false,
  "enable_throughput_scaling": true,
  "throughput_metrics_source": "router",
  "ttft": 2000,
  "prefill_engine_num_gpu": 2,
  "model_name": "${MODEL_NAME}",
  "profile_results_dir": "/workspace/tests/planner/profiling_results/H200_TP1P_TP1D"
}
```

`global_planner_namespace` must point to the control stack's **Dynamo namespace**. In the reference manifests, that is the namespace string passed to the control `Frontend` and `GlobalRouter`.

Use:

- `mode: "prefill"` for prefill-only pools
- `mode: "decode"` for decode-only pools

The worker and planner settings for each pool come from the pool-specific profiling result you created in Step 1.

In the reference vLLM example:

- `gp-prefill-0` uses a 1-GPU TP1 prefill worker
- `gp-prefill-1` uses a 2-GPU TP2 prefill worker
- `gp-decode-0` uses a 1-GPU TP1 decode worker

See [global-planner-vllm-test.yaml](../../../examples/global_planner/global-planner-vllm-test.yaml).

## Step 4: Configure GlobalRouter To Select Pools

`GlobalRouter` reads a JSON config that lists the pool namespaces and a routing grid for each request type.

Example:

```json
{
  "num_prefill_pools": 2,
  "num_decode_pools": 1,
  "prefill_pool_dynamo_namespaces": [
    "${K8S_NAMESPACE}-gp-prefill-0",
    "${K8S_NAMESPACE}-gp-prefill-1"
  ],
  "decode_pool_dynamo_namespaces": [
    "${K8S_NAMESPACE}-gp-decode-0"
  ],
  "prefill_pool_selection_strategy": {
    "ttft_min": 10,
    "ttft_max": 3000,
    "ttft_resolution": 2,
    "isl_min": 0,
    "isl_max": 32000,
    "isl_resolution": 2,
    "prefill_pool_mapping": [[0, 1], [0, 1]]
  },
  "decode_pool_selection_strategy": {
    "itl_min": 10,
    "itl_max": 500,
    "itl_resolution": 2,
    "context_length_min": 0,
    "context_length_max": 32000,
    "context_length_resolution": 2,
    "decode_pool_mapping": [[0, 0], [0, 0]]
  }
}
```

The `prefill_pool_dynamo_namespaces` and `decode_pool_dynamo_namespaces` entries are **Dynamo namespaces** that the pool-local routers register under.

Important runtime behavior:

- Prefill pool selection uses **ISL + TTFT target**
- Decode pool selection uses **context length + ITL target**
- OSL is useful for **designing and profiling pools**, but it is **not a direct routing key** in the current `GlobalRouter`

Clients can pass request targets through `extra_args`:

```json
{
  "extra_args": {
    "ttft_target": 200,
    "itl_target": 20
  }
}
```

For more details, see [Global Router README](../../../components/src/dynamo/global_router/README.md).

## Step 5: Deploy In Order

For a fresh cluster, the usual order is:

1. Install Dynamo platform and Prometheus.
2. Create secrets and PVCs needed by workers.
3. Create the `GlobalRouter` ConfigMap.
4. Apply the control DGD.
5. Apply the pool DGDs.
6. Wait for all DGDs to reach ready state.
7. Expose or port-forward the control `Frontend`.

Example:

```bash
export K8S_NAMESPACE=my-llama
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export DYNAMO_IMAGE=<dynamo-image>
export DYNAMO_VLLM_IMAGE=<vllm-image>
export STORAGE_CLASS_NAME=<rwx-storage-class>

kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${K8S_NAMESPACE}

envsubst < examples/global_planner/global-planner-vllm-test.yaml | \
  kubectl apply -n ${K8S_NAMESPACE} -f -
```

The single user-facing endpoint is the `Frontend` in the control DGD, not the pool DGDs.

## Step 6: Validate The Stack

Validate the deployment from outside in:

- Confirm the control `Frontend` is healthy and serving the model endpoint.
- Confirm `GlobalRouter` logs show requests being assigned to the expected pool namespaces.
- Confirm pool-local planners are producing scale requests.
- Confirm `GlobalPlanner` logs show accepted scale operations.
- Confirm the target DGDs' replica counts change as expected.

If you use Prometheus and Grafana, also inspect:

- TTFT and ITL over time
- per-pool worker counts
- per-pool request mix
- total GPU usage

## Recommended Workflow For New Deployments

For most teams, the easiest way to build this deployment is:

1. Design your pool classes from expected traffic patterns.
2. Run one DGDR per pool class to generate or validate the pool configuration.
3. Copy the selected worker shape and planner settings into the final pool DGDs.
4. Build one control DGD with `Frontend`, `GlobalRouter`, and `GlobalPlanner`.
5. Route all client traffic through the control `Frontend`.

This keeps profiling and pool selection simple while still giving you one public endpoint for the model.

## Current Limitations

- Single-endpoint `GlobalPlanner` deployments are assembled manually today. One DGDR does not emit the full control DGD plus pool DGDs topology.
- `GlobalRouter` routes by ISL/TTFT and context-length/ITL grids, not directly by OSL.
- In the single-endpoint pattern, all pools are expected to serve the same model.

## See Also

- [Planner README](README.md) — Planner overview and quick start
- [Planner Guide](planner-guide.md) — Planner configuration reference
- [Planner Examples](planner-examples.md) — DGDR examples for generating per-pool configs
- [Profiler Guide](../profiler/profiler-guide.md) — Pre-deployment profiling workflow
- [Global Planner README](../../../components/src/dynamo/global_planner/README.md) — Centralized scale execution
- [Global Router README](../../../components/src/dynamo/global_router/README.md) — Cross-pool request routing
- [vLLM global planner example](../../../examples/global_planner/global-planner-vllm-test.yaml) — End-to-end reference manifest
