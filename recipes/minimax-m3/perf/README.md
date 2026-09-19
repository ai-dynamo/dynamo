<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MiniMax M3 Benchmark Recipe

A single [AIPerf](https://github.com/ai-dynamo/aiperf) trace-replay Job — [`perf.yaml`](perf.yaml) — covers both MiniMax M3 GB200 DGD variants. The benchmark is identical across the aggregate and disaggregated deployments; only `ENDPOINT` and the `podAffinity` target need to change.

The Job waits for `GET /v1/models` on the DGD frontend to return `nvidia/MiniMax-M3-NVFP4` (up to ~1 hour by default), runs a short warmup, then replays the configured trace at a single `CONCURRENCY` value and writes raw artifacts to the shared `model-cache` PVC.

The benchmark pod is **co-located with the DGD frontend** (`podAffinity` on the frontend's host) so client-to-server traffic stays on a single node.

## Targeting a variant

Edit the `env` block in [`perf.yaml`](perf.yaml), then update the `podAffinity` `values` list to contain only the target DGD name:

| Variant target | DGD name | `ENDPOINT` | `SYNTHESIS_MAX_ISL` | `TRACE_FILE` |
| --- | --- | --- | --- | --- |
| GB200 aggregate agentic | `minimax-m3-agg-gb200-agentic` | `minimax-m3-agg-gb200-agentic-frontend:8000` | `1000000` | `/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` |
| GB200 disaggregated agentic | `minimax-m3-disagg-gb200-agentic` | `minimax-m3-disagg-gb200-agentic-frontend:8000` | `1000000` | `/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` |

Both DGDs serve `nvidia/MiniMax-M3-NVFP4` and replay the same agentic trace. If you run more than one benchmark in the same namespace, also update `metadata.name` and `labels.app` so Jobs and artifact directories stay distinct.

## Dataset

The benchmark replays a [Mooncake-format](https://github.com/kvcache-ai/Mooncake) trace via `aiperf --custom-dataset-type mooncake_trace`. Each JSONL line describes one request (`input_length`, `output_length`, and `hash_ids`).

The expected trace on the PVC is:

- **Agentic** — `/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl`

This is the same 64K-ISL / 400-OSL / 90%-KV-reuse agentic trace used by the Kimi-K2.6 recipe. Rather than duplicate the Git LFS blob, the MiniMax recipe references it through a symlink under [`traces`](traces):

```text
traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl
  -> ../../../kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl
```

The 15% trace contains 3,541 requests. Its SHA-256 is `f20d3f2bc83dd1306cda659fbe34e7c4d85ca5497626c98bc0b1c4d2211379d0`. For shorter or longer runs, point `TRACE_FILE` at another subset of the same trace rather than imposing a time limit.

## Workflow

```bash
export NAMESPACE=your-namespace
```

### 1. Deploy the DGD

See the deployment instructions in the [recipe README](../README.md).

### 2. Stage the trace on the PVC

Materialize the Git LFS trace, then use a helper pod that mounts `model-cache` to copy it onto the PVC:

```bash
git lfs pull --include='recipes/kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl'

kubectl run pvc-helper -n ${NAMESPACE} \
  --image=busybox:1.36 --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"helper","image":"busybox:1.36","command":["sleep","3600"],"volumeMounts":[{"name":"model-cache","mountPath":"/model-cache"}]}],"volumes":[{"name":"model-cache","persistentVolumeClaim":{"claimName":"model-cache"}}]}}' \
  --command -- sleep 3600

TRACE_SOURCE="$(git rev-parse --show-toplevel)/recipes/kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl"
kubectl exec -n "${NAMESPACE}" pvc-helper -- mkdir -p /model-cache/traces
kubectl cp "${TRACE_SOURCE}" \
  "${NAMESPACE}/pvc-helper:/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl"
```

Keep `pvc-helper` around for fetching artifacts later, or delete it after staging.

### 3. Run the benchmark

```bash
kubectl apply -f perf.yaml -n ${NAMESPACE}
kubectl logs -n ${NAMESPACE} -l job-name=minimax-m3-bench -f
kubectl wait --for=condition=Complete job/minimax-m3-bench \
  -n ${NAMESPACE} --timeout=10800s
```

The Job uses `nvcr.io/nvidia/ai-dynamo/aiperf:0.11.0` directly and does
not install or patch AIPerf at runtime.

### 4. Fetch artifacts

```bash
kubectl cp \
  ${NAMESPACE}/pvc-helper:/model-cache/perf/<epoch>_minimax-m3-bench \
  ./results
```

### 5. Cleanup

```bash
kubectl delete job minimax-m3-bench -n ${NAMESPACE}
kubectl delete pod pvc-helper -n ${NAMESPACE}
```

## Running a concurrency sweep

`perf.yaml` runs a **single** `CONCURRENCY` value. To measure multiple concurrencies, clear vLLM KV state and Dynamo frontend/router state between runs; otherwise residual KV and prefix-cache state can skew the results.

For each concurrency value:

```bash
kubectl delete job minimax-m3-bench -n ${NAMESPACE} --ignore-not-found

DGD=minimax-m3-agg-gb200-agentic # Choose one of the two DGD names above.
kubectl delete pods -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD}
kubectl wait --for=condition=Ready pod -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD} \
  --timeout=7200s

# Update CONCURRENCY in perf.yaml before each run.
kubectl apply -f perf.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/minimax-m3-bench \
  -n ${NAMESPACE} --timeout=10800s
```

Do not compare partial runs. A completed run must account for successful,
errored, and unfinished requests before reporting aggregate throughput.

The Job's readiness loop handles the DGD restart window by polling `/v1/models` until the frontend reports the expected model again.

## Tunable environment variables

Edit the `env` block on the Job to adjust:

| Variable | Default | Notes |
| --- | --- | --- |
| `ENDPOINT` | `minimax-m3-agg-gb200-agentic-frontend:8000` | Change per DGD variant |
| `TRACE_FILE` | `/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` | 3,541-request 15% agent trace |
| `SYNTHESIS_MAX_ISL` | `1000000` | Matches the MiniMax M3 recipe context limit |
| `CONCURRENCY` | `64` | Single value; reset server state between values |
| `TARGET_MODEL` | `nvidia/MiniMax-M3-NVFP4` | Must match `--served-model-name` |

## Artifacts

Results are written to:

```text
/model-cache/perf/<epoch>_<job-name>/
  warmup/
  MiniMax-M3-NVFP4_trace_c<concurrency>_<timestamp>/
    profile_export_aiperf.json
    inputs.json
    ...
```
