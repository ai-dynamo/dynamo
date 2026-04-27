<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Global Planner Examples

Examples demonstrating **GlobalPlanner** — the centralized scaling execution layer that
enforces shared scaling policy across multiple DGDs.

## Example Manifests

| File | Pattern | Backend | Description |
|------|---------|---------|-------------|
| `global-planner-gpu-budget.yaml` | Multi-model, GPU budget | vLLM | 2 independent model DGDs + 1 control DGD with `--max-total-gpus` |
| `global-planner-vllm-test.yaml` | Single-endpoint, multi-pool | vLLM | 1 Frontend + GlobalRouter + GlobalPlanner, 2 prefill pools (TP1, TP2) + 1 decode pool |
| `global-planner-mocker-test.yaml` | Single-endpoint, multi-pool | Mocker | Same as above with Mocker workers; GlobalPlanner in `--no-operation` mode |
| `single-pool-priority-benchmark.yaml` | Single-endpoint, single-pool | Mocker | 1 Frontend with KV routing + 1 agg worker pool + local Planner autoscaling |
| `run_priority_benchmark_in_cluster.sh` | In-cluster benchmark runner | Mocker | Creates a ConfigMap from `benchmarks/router/global_router_priority_benchmark.py` and runs it as a Kubernetes Job against `gp-ctrl-frontend` |

## Deployment Patterns

### Pattern 1: Multi-Model with GPU Budget (`global-planner-gpu-budget.yaml`)

Multiple independent DGDs, each serving a different model with its own Frontend.
A shared GlobalPlanner enforces a cluster-wide GPU cap.

```
DGD gp-ctrl:    GlobalPlanner (--max-total-gpus)
DGD model-a:    Frontend + VllmPrefillWorker + VllmDecodeWorker + Planner  (MODEL_A)
DGD model-b:    Frontend + VllmPrefillWorker + VllmDecodeWorker + Planner  (MODEL_B)
```

- No GlobalRouter needed — each model has its own endpoint.
- Each DGD's local planner uses `environment: "global-planner"` to delegate scaling.
- GlobalPlanner rejects any scale request that would push total GPUs above the limit.

### Pattern 2: Single Endpoint, Multi-Pool (`global-planner-vllm-test.yaml`)

One public endpoint for a single model, backed by multiple specialized pools.
A GlobalRouter selects the best pool for each request.

```
DGD gp-ctrl:      Frontend + GlobalRouter + GlobalPlanner
DGD gp-prefill-0: LocalRouter + VllmPrefillWorker (TP1) + Planner
DGD gp-prefill-1: LocalRouter + VllmPrefillWorker (TP2) + Planner
DGD gp-decode-0:  LocalRouter + VllmDecodeWorker  (TP1) + Planner
```

- GlobalRouter routes prefill requests by (ISL, TTFT target) and decode by (context length, ITL target).
- Each pool planner delegates scaling to GlobalPlanner.

## Prerequisites

- Dynamo Kubernetes Platform installed (see [Kubernetes Quickstart](../../docs/kubernetes/README.md))
- Cluster Prometheus scraping router metrics via PodMonitor
- HuggingFace token secret:
  ```bash
  kubectl create secret generic hf-token-secret \
    --from-literal=HF_TOKEN=<your-token> -n ${K8S_NAMESPACE}
  ```
- A ReadWriteMany StorageClass for the shared model cache PVC

## Deploying

All manifests use `envsubst` for configuration. Set the required variables and apply:

### GPU Budget Example

```bash
export K8S_NAMESPACE=my-ns
export DYNAMO_IMAGE=<dynamo-image>
export DYNAMO_VLLM_IMAGE=<vllm-image>
export STORAGE_CLASS_NAME=<rwx-storage-class>
export MODEL_A=meta-llama/Llama-3.1-8B-Instruct
export MODEL_B=Qwen/Qwen3-8B
export MAX_TOTAL_GPUS=8

envsubst < global-planner-gpu-budget.yaml | kubectl apply -n ${K8S_NAMESPACE} -f -
```

### Single-Endpoint vLLM Example

```bash
export K8S_NAMESPACE=my-ns
export DYNAMO_IMAGE=<dynamo-image>
export DYNAMO_VLLM_IMAGE=<vllm-image>
export STORAGE_CLASS_NAME=<rwx-storage-class>
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct

envsubst < global-planner-vllm-test.yaml | kubectl apply -n ${K8S_NAMESPACE} -f -
```

### Mocker Example (No GPUs)

```bash
export K8S_NAMESPACE=my-ns
export DYNAMO_IMAGE=<dynamo-image>

envsubst < global-planner-mocker-test.yaml | kubectl apply -n ${K8S_NAMESPACE} -f -
```

### Single-Pool Priority Benchmark Baseline

```bash
export K8S_NAMESPACE=my-ns
export DYNAMO_IMAGE=<dynamo-image>

envsubst < single-pool-priority-benchmark.yaml | kubectl apply -n ${K8S_NAMESPACE} -f -
```

## Running The Priority Benchmark In Cluster

The priority benchmark script can be run as a Kubernetes `Job` in the same namespace as
the control DGD. This avoids `kubectl port-forward` bottlenecks at very high concurrency.
Both the single-pool and global-router benchmark manifests expose the same
frontend service name, `gp-ctrl-frontend`, so the same benchmark invocation can
be reused across the two deployment shapes.

```bash
export K8S_NAMESPACE=jthomson-bench
export DYNAMO_IMAGE=<dynamo-image>
export BENCHMARK_EXTRA_ARGS="--num-batches 1 --prompts-per-batch 32 --generations-per-prompt 8 --lag 2"

./run_priority_benchmark_in_cluster.sh
```

Useful knobs:

- `JOB_NAME`: Job name to create. Default: `gp-priority-benchmark`
- `BENCHMARK_URL`: In-cluster frontend URL. Default: `http://gp-ctrl-frontend:8000`
- `BENCHMARK_EXTRA_ARGS`: Extra CLI flags passed to `global_router_priority_benchmark.py`
- `LOCAL_OUTPUT_DIR`: Local directory for copied artifacts. Defaults to `examples/global_planner/benchmark_artifacts/<job-name>-<timestamp>`
- `BENCHMARK_NODE_NAME`: Optional node pin if you want to reuse a node that already has the image cached
- `IMAGE_PULL_SECRET_NAME`: Optional pull secret name. Set it to an empty string for public images
- `POST_RUN_HOLD_SECONDS`: How long the pod stays alive after the benchmark finishes so `kubectl cp` can pull artifacts. Defaults to `300`
- `FOLLOW_LOGS=0`: Launch without attaching to the Job logs

Example with a heavier workload and local artifact copy:

```bash
export BENCHMARK_EXTRA_ARGS="--num-batches 3 --prompts-per-batch 64 --generations-per-prompt 12 --lag 3 --straggler-fraction 1.0 --straggler-turns 16 --straggler-osl-mean 2200 --straggler-osl-std 400 --base-isl 1600 --user-tokens-per-turn 300"
export LOCAL_OUTPUT_DIR=./benchmark_artifacts/gp-priority-benchmark-results

./run_priority_benchmark_in_cluster.sh
```

## Verifying

```bash
# Check DGD status
kubectl get dgd -n ${K8S_NAMESPACE}

# Check pods
kubectl get pods -n ${K8S_NAMESPACE}

# Watch GlobalPlanner logs for scale requests
kubectl logs -n ${K8S_NAMESPACE} \
  -l nvidia.com/dynamo-component=GlobalPlanner -f
```

## Cleanup

```bash
envsubst < <manifest>.yaml | kubectl delete -n ${K8S_NAMESPACE} -f -
```

## SLA Planner Configuration

Each pool's local planner is configured via a JSON blob passed to `--config`.
Key fields for GlobalPlanner delegation:

| Field | Description |
|-------|-------------|
| `environment` | `"global-planner"` — delegates scaling to GlobalPlanner |
| `global_planner_namespace` | Dynamo namespace of the control DGD (e.g. `${K8S_NAMESPACE}-gp-ctrl`) |
| `mode` | `"disagg"`, `"prefill"`, or `"decode"` |
| `throughput_metrics_source` | `"router"` for multi-DGD setups (reads `dynamo_component_router_*` from Prometheus) |
| `max_gpu_budget` | Per-pool GPU limit (`-1` = unlimited, defer to GlobalPlanner) |

## GlobalPlanner Flags

| Flag | Description |
|------|-------------|
| `--max-total-gpus N` | Reject requests that would exceed N total GPUs across all managed DGDs. `0` = no GPU scaling allowed, `-1` (default) = unlimited |
| `--managed-namespaces NS...` | Only accept scale requests from listed Dynamo namespaces (default: accept all). See *Management Modes* below |
| `--no-operation` | Log scale requests without executing them (useful for dry-run testing) |

### Management Modes

GlobalPlanner operates in one of two modes depending on whether `--managed-namespaces` is set:

- **Explicit mode** (`--managed-namespaces` provided): Only the listed Dynamo
  namespaces are authorized to send scale requests, and only their corresponding
  DGDs count toward the GPU budget. DGD names are derived from the Dynamo
  namespace using the operator convention `DYN_NAMESPACE = {k8s_namespace}-{dgd_name}`.
- **Implicit mode** (no `--managed-namespaces`): Any caller is accepted, and all
  DGDs in the Kubernetes namespace count toward the GPU budget.

## Namespace Convention

The Dynamo operator prepends the Kubernetes namespace to the DGD's `dynamoNamespace`:
- K8s namespace: `my-ns`, DGD name: `gp-ctrl`
- Dynamo namespace: `my-ns-gp-ctrl`

This is why planner configs and router endpoints use the full `${K8S_NAMESPACE}-<dgd-name>` path.

## Further Reading

- [Global Planner Deployment Guide](../../docs/components/planner/global-planner.md)
- [Global Planner README](../../components/src/dynamo/global_planner/README.md)
- [Planner Configuration Guide](../../docs/components/planner/planner-guide.md)
- [Global Router README](../../components/src/dynamo/global_router/README.md)
