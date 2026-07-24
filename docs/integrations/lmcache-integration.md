---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: LMCache
subtitle: Integrate LMCache with Dynamo's vLLM backend for prefill-once, reuse-everywhere KV cache across requests.
---

## Introduction

LMCache is a high-performance KV cache layer that supercharges LLM serving by enabling **prefill-once, reuse-everywhere** semantics. As described in the [official documentation](https://docs.lmcache.ai/index.html), LMCache lets LLMs prefill each text only once by storing the KV caches of all reusable texts, allowing reuse of KV caches for any reused text (not necessarily prefix) across any serving engine instance.

This document describes how LMCache is integrated into Dynamo's vLLM backend to provide enhanced performance and memory efficiency.

> **Deployment paths** — this guide covers two ways to run Dynamo + LMCache. Jump to the one that matches your environment:
> - **Local / bare-metal** (sections immediately below): the `lmcache server` started by the `launch/` scripts.
> - **[Kubernetes](#kubernetes-deployment)**: the LMCache operator + an `LMCacheEngine` CR + a per-node DaemonSet, deployed via `DynamoGraphDeployment`.

## Installation Notes

Dynamo's vLLM runtime expects LMCache to be present in the same Python environment. On supported environments (x86_64, Python 3.10-3.13, PyTorch built against CUDA 12.x), the published wheel installs directly:

```bash
uv pip install lmcache
```

LMCache only publishes x86_64 manylinux wheels linked against CUDA 12. For aarch64 hosts, or hosts running PyTorch built against a different CUDA major version, build LMCache from source against your matching torch + CUDA stack — see the official [LMCache installation guide](https://docs.lmcache.ai/getting_started/installation.html).

> **Compatibility note**
>
> `LMCacheMPConnector` needs the fix from [LMCache#3282](https://github.com/LMCache/LMCache/pull/3282), which is on LMCache `main` but not yet released. Without it, the MP path fails on vLLM ≥ 0.20.0 (including the `vllm==0.21.0` Dynamo currently pins) with `RuntimeError: Unsupported GPUKVFormat: 7` — vLLM 0.20+ uses GPU KV formats 6 / 7 that the MP path doesn't yet handle.
>
> Until the next LMCache release, build LMCache from source against that PR.

## Aggregated Serving

### Configuration

LMCache runs the cache engine as an out-of-process sidecar (`lmcache server`); the Dynamo worker connects to it via the `LMCacheMPConnector`. Start the sidecar, then launch the worker:

```bash
lmcache server --l1-size-gb 100 --eviction-policy LRU &

python -m dynamo.vllm \
  --model <model_name> \
  --disable-hybrid-kv-cache-manager \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'
```

### Customization

The LMCache MP server is configured via CLI arguments. See the [Configuration Reference](https://docs.lmcache.ai/mp/configuration.html) for the full list of `lmcache server` flags.

LMCache MP uses a two-tier storage architecture: an in-memory L1 cache (sized with `--l1-size-gb`) plus optional persistent L2 adapters configured with `--l2-adapter`. The supported [L2 storage backends](https://docs.lmcache.ai/mp/l2_storage.html) are:

- **POSIX**: Standard POSIX file I/O on any file system
- **GDS** / **GDS_MT**: NVIDIA GPU Direct Storage (single- and multi-threaded), bypassing the CPU for NVMe SSDs that support GDS
- **HF3FS**: Distributed / shared file-system backend
- **OBJ**: Object store backend
- **AZURE_BLOB**: Azure Blob Storage

### Deployment

Use the provided launch script for quick setup:

```bash
./examples/backends/vllm/launch/agg_lmcache_mp.sh
```

This will:
1. Start the LMCache MP server
2. Start the Dynamo frontend
3. Launch a single vLLM worker with `LMCacheMPConnector` connected to the sidecar

### Architecture for Aggregated Mode

In aggregated mode, the system uses:

- **KV Connector**: `LMCacheMPConnector`
- **KV Role**: `kv_both` (handles both reading and writing)

## Disaggregated Serving

Disaggregated serving separates prefill and decode operations into dedicated workers. This provides better resource utilization and scalability for production deployments.

### Deployment

Use the provided disaggregated launch script (requires at least 2 GPUs):

```bash
./examples/backends/vllm/launch/disagg_lmcache.sh
```

This will:
1. Start the Dynamo frontend
2. Launch a decode worker on GPU 0
3. Wait for initialization
4. Launch a prefill worker on GPU 1 with LMCache enabled

### Worker Roles

#### Decode Worker

- **Purpose**: Handles token generation (decode phase)
- **GPU Assignment**: CUDA_VISIBLE_DEVICES=0
- **LMCache Config**: Uses `NixlConnector` only for KV transfer between prefill and decode workers

#### Prefill Worker

- **Purpose**: Handles prompt processing (prefill phase)
- **GPU Assignment**: CUDA_VISIBLE_DEVICES=1
- **LMCache Config**: Uses `MultiConnector` with both LMCache and NIXL connectors. This enables prefill worker to use LMCache for KV offloading and use NIXL for KV transfer between prefill and decode workers.
- **Flag**: `--disaggregation-mode prefill`

## Architecture

### KV Transfer Configuration

The system automatically configures KV transfer based on the deployment mode and worker type:

#### Aggregated Mode

```python
kv_transfer_config = KVTransferConfig(
    kv_connector="LMCacheMPConnector",
    kv_role="kv_both",
    kv_connector_extra_config={"lmcache.mp.port": 5555},
)
```

#### Prefill Worker (Disaggregated Mode)

```python
kv_transfer_config = KVTransferConfig(
    kv_connector="PdConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "connectors": [
            {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"},
            {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
        ]
    }
)
```

#### Decode Worker (Disaggregated Mode)

```python
kv_transfer_config = KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both"
)
```

#### Fallback (No LMCache)

```python
kv_transfer_config = KVTransferConfig(
    kv_connector="NixlConnector",
    kv_role="kv_both"
)
```

### Integration Points

1. **Argument Parsing** (`args.py`):
   - Configures appropriate KV transfer settings
   - Sets up connector configurations based on worker type

2. **Engine Setup** (`main.py`):
   - Creates vLLM engine with proper KV transfer config
   - Handles both aggregated and disaggregated modes

3. **Sidecar Lifecycle** (launch script):
   - Starts the `lmcache server` process before the Dynamo worker
   - Tears it down on exit via the script's cleanup trap

### Best Practices

1. **Chunk Size Tuning**: Pass `--chunk-size` to `lmcache server` based on your use case:
   - Smaller chunks (128-256): Better reuse granularity for varied content
   - Larger chunks (512-1024): More efficient for repetitive content patterns

2. **Memory Allocation**: Set `--l1-size-gb` on `lmcache server` conservatively:
   - Leave sufficient RAM for other system processes
   - Monitor memory usage during peak loads

3. **Workload Optimization**: LMCache performs best with:
   - Repeated prompt patterns (RAG, multi-turn conversations)
   - Shared context across sessions
   - Long-running services with warm caches

## Metrics and Monitoring

The LMCache MP server records metrics through the OpenTelemetry SDK and exposes them on its own HTTP admin port (default `:8080/metrics`), prefixed `lmcache_mp_`:

```bash
curl -s localhost:8080/metrics | grep '^lmcache_mp_'
```

vLLM and Dynamo metrics remain on Dynamo's `:8081/metrics` (set `DYN_SYSTEM_PORT=8081` on the worker to enable that endpoint).

For detailed information on LMCache metrics, including the complete list of available metrics and how to access them, see the **[LMCache Metrics section](../backends/vllm/vllm-observability.md#lmcache-metrics)** in the vLLM Prometheus Metrics Guide.

## Troubleshooting

### vLLM log: `Found PROMETHEUS_MULTIPROC_DIR was set by user`

vLLM v1 uses `prometheus_client.multiprocess` and stores intermediate metric values in `PROMETHEUS_MULTIPROC_DIR`.

- If you **set `PROMETHEUS_MULTIPROC_DIR` yourself**, vLLM warns that the directory must be wiped between runs to avoid stale/incorrect metrics.
- When running via Dynamo, the vLLM wrapper may set `PROMETHEUS_MULTIPROC_DIR` internally to a temporary directory to avoid vLLM cleanup issues. If you still see the warning, confirm you are not exporting `PROMETHEUS_MULTIPROC_DIR` in your shell or container environment.

## Kubernetes Deployment

Kubernetes manifests for running Dynamo aggregated vLLM serving with KV cache offloaded to a per-node LMCache MP DaemonSet, sharing tensors with the worker via cross-Pod CUDA IPC.

### Prerequisites

- K8s cluster with the NVIDIA GPU Operator.
- `kubectl` + `helm` with `KUBECONFIG` set.
- The Dynamo platform helm chart must include
  [PR #8414](https://github.com/ai-dynamo/dynamo/pull/8414) — it ships the
  relaxed `sharedMemory.disabled` CRD validation rule that our worker needs to
  opt out of the operator's auto-injected `/dev/shm` emptyDir. No NGC chart
  yet ships with this fix; clone `ai-dynamo/dynamo` and check out commit
  [`e7eb1c565f`](https://github.com/ai-dynamo/dynamo/commit/e7eb1c565f), then
  install from the local chart (see Step 1).

### Step 1 — Install the Dynamo platform

```bash
helm install dynamo-platform oci://nvcr.io/nvidia/ai-dynamo/dynamo-platform \
  --version my-version \
  --namespace dynamo-system --create-namespace \
  --wait
```

> **How it works today** — NGC hasn't published a helm chart with the PR #8414
> CRD fix yet, so the clean `helm install` above doesn't work as-is. Two pieces
> are needed: the **chart** (carries the updated CRD) and the **operator image**
> (whose baked-in CRDs an init container re-applies on every restart). Until NGC
> ships both, the workaround is:
>
> 1. **Chart from local clone**: `git clone https://github.com/ai-dynamo/dynamo.git` and check out
>    commit [`e7eb1c565f`](https://github.com/ai-dynamo/dynamo/commit/e7eb1c565f)
>    (the version validated for this recipe), then install from the local path
>    with the NGC `1.1.1` operator image:
>    ```bash
>    # Required for `helm dependency build` — the platform chart pulls in
>    # NATS and Bitnami subcharts and helm needs to know their repos.
>    helm repo add nats https://nats-io.github.io/k8s/helm/charts/
>    helm repo add bitnami https://charts.bitnami.com/bitnami
>    helm repo update
>
>    cd dynamo/deploy/helm/charts
>    helm dependency build ./platform/
>    helm install dynamo-platform ./platform/ \
>      --namespace dynamo-system --create-namespace \
>      --set "dynamo-operator.controllerManager.manager.image.tag=1.1.1" \
>      --wait
>    ```
> 2. **Patch out the operator's CRD-applying init container** (it would otherwise
>    overwrite the chart's new CRDs on every restart):
>    ```bash
>    kubectl -n dynamo-system patch deployment dynamo-platform-dynamo-operator-controller-manager \
>      --type=json -p='[{"op":"replace","path":"/spec/template/spec/initContainers","value":[]}]'
>    ```
> 3. **Replace the CRDs by hand** (helm doesn't update CRDs on install/upgrade):
>    ```bash
>    cd ../../..
>    kubectl replace -f platform/components/operator/crds/nvidia.com_dynamographdeployments.yaml
>    kubectl replace -f platform/components/operator/crds/nvidia.com_dynamocomponentdeployments.yaml
>    ```
>
> Alternative: build the operator image yourself from a commit that
> includes PR #8414 (so its init container re-applies the **new** CRDs) and
> point the chart at it via `--set ...image.tag=<your-tag>`. Avoids steps 2 and 3, but adds a registry of your own.

Verify the install:
```bash
kubectl get crd | grep nvidia.com
# expect:
#   dynamocomponentdeployments.nvidia.com
#   dynamographdeployments.nvidia.com
```

### Step 2 — Install the LMCache operator

```bash
kubectl apply -f https://github.com/LMCache/LMCache/releases/download/operator-latest/install.yaml
```

> Tested with
> the LMCache operator image `lmcache/lmcache-operator:v0.1.1`.

Verify the install:
```bash
kubectl get crd | grep lmcache
# expect:
#   lmcacheengines.lmcache.lmcache.ai
```

### Step 3 — Create the namespace

```bash
kubectl create namespace dynamo-lmcache
```

### Step 4 — Create the HF token Secret

Both `Frontend` and `VllmDecodeWorker` reference `hf-token-secret` via `envFromSecret`.
The Secret must exist or the pods fail to start with `secret "hf-token-secret" not found`.

```bash
# Replace the dummy token below with a real HF token.
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: Secret
metadata:
  name: hf-token-secret
  namespace: dynamo-lmcache
type: Opaque
stringData:
  HF_TOKEN: "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
EOF
```

### Step 5 — Deploy the LMCacheEngine

Replace `my-tag` below with the `lmcache/vllm-openai` image tag you want to run.

```bash
kubectl apply -f - <<'EOF'
apiVersion: lmcache.lmcache.ai/v1alpha1
kind: LMCacheEngine
metadata:
  name: lmcache-mp
  namespace: dynamo-lmcache
spec:
  image:
    repository: lmcache/vllm-openai
    tag: my-tag
    pullPolicy: IfNotPresent
  # L1 (CPU RAM) cache size — bump for production workloads.
  l1:
    sizeGB: 16
EOF
```

> Validated `my-tag` against `nightly-2026-04-25` (lmcache
> `0.4.5.dev31`). This pre-stable build is wire-compatible with the
> `vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.3` worker image (which ships
> lmcache `0.4.4`).

Verify with:
```bash
kubectl -n dynamo-lmcache get lmcacheengine lmcache-mp
# expect: STATUS  Running
```

### Step 6 — Deploy the Dynamo worker

Edit `agg_lmcache.yaml`: replace `nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag` (on
both `Frontend` and `VllmDecodeWorker`) with your Dynamo vllm-runtime image.

```bash
kubectl apply -n dynamo-lmcache -f examples/backends/vllm/deploy/agg_lmcache.yaml
```

> Validated `my-tag` against `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.3`
> (vLLM `0.20.1`, lmcache `0.4.4`). Pair with the LMCacheEngine pin from
> Step 5.

Verify with:
```bash
kubectl -n dynamo-lmcache get pods -l nvidia.com/dynamo-component-type=worker
# expect: READY 1/1, STATUS Running
```

### Step 7 — Verify

Send the same long prompt twice:

```bash
kubectl -n dynamo-lmcache port-forward svc/vllm-agg-lmcache-frontend 8000:8000 >/dev/null &
trap 'kill %1' EXIT
sleep 4

PROMPT=$(python3 -c "print('the quick brown fox jumps over the lazy dog '*60)")
REQ="{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":5}"

for label in cold warm; do
  echo "--- $label ---"
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" -d "$REQ" \
    | python3 -m json.tool | grep -E "prompt_tokens|cached_tokens"
done
```

Then check LMCache server metrics:

```bash
LMC=$(kubectl -n dynamo-lmcache get pod -l app.kubernetes.io/instance=lmcache-mp -o name | head -1)
kubectl -n dynamo-lmcache exec "$LMC" -- curl -s localhost:9090/metrics | grep '^lmcache_mp_'
```

Expected: warm response shows `cached_tokens > 0`, and
`lmcache_mp_lookup_hit_tokens_total > 0`.

#### Metrics endpoint by lmcache version

The LMCache operator declares three container ports on every DaemonSet pod:
`5555` (ZMQ), `8080` (control HTTP), `9090` (Prometheus). It also creates two
Services: `lmcache-mp` (5555 + 8080) and a headless `lmcache-mp-metrics` (9090).
Which port actually serves `/metrics` depends on the image:

| lmcache version | `/metrics` on `:9090` | `/metrics` on `:8080` | Metric labels |
|---|---|---|---|
| `nightly-2026-04-25` (lmcache `0.4.5.dev31`) | ✅ | 404 | bare counters, no labels |
| v0.4.5 stable (released 2026-05-15) and later | ✅ | ✅ | labeled with `{cache_salt, model_name}` |

### Cleanup

```bash
kubectl delete -n dynamo-lmcache -f examples/backends/vllm/deploy/agg_lmcache.yaml
kubectl -n dynamo-lmcache delete lmcacheengine lmcache-mp
kubectl delete namespace dynamo-lmcache
kubectl delete -f https://github.com/LMCache/LMCache/releases/download/operator-latest/install.yaml
helm uninstall dynamo-platform -n dynamo-system
kubectl delete namespace dynamo-system lmcache-operator-system
```

## References and Additional Resources

- [LMCache Documentation](https://docs.lmcache.ai/index.html) - Comprehensive guide and API reference
- [Configuration Reference](https://docs.lmcache.ai/mp/configuration.html) - `lmcache server` CLI arguments
- [LMCache Observability Guide](https://docs.lmcache.ai/mp/observability.html) - Metrics and monitoring details
