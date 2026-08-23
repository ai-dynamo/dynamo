<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.8-Flash-Next Benchmark Recipe

A single [AIPerf](https://github.com/ai-dynamo/aiperf) trace-replay Job — [perf.yaml](perf.yaml) — covers every Qwen3.8-Flash-Next DGD variant (agg/disagg × chat/agent traces). The benchmark is identical across variants; only `ENDPOINT`, `TRACE_FILE`, and `TARGET_MODEL` need to change.

The Job waits for `GET /v1/models` on the DGD frontend to return the configured `TARGET_MODEL` (up to ~1h by default), runs a short warmup, then replays the configured trace at a single `CONCURRENCY` value and writes raw artifacts to the shared `model-cache` PVC.

The bench pod is **co-located with the DGD frontend** (`podAffinity` on the frontend's host) so client → server traffic stays on a single node.

## Targeting a variant

Edit the `env` block in [perf.yaml](perf.yaml):

| Variant target           | `ENDPOINT`                                         | `TARGET_MODEL`                     | `TRACE_FILE` (chat / agent)                                             |
| ------------------------ | -------------------------------------------------- | ----------------------------------- | ----------------------------------------------------------------------- |
| B200 agg, agent workload | `qwen38fn-agg-b200-agentic-frontend:8000`     | `Inferact/Qwen3.8-Flash-Next-NVFP4` | `/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` |
| B200 disagg, agent       | `qwen38fn-disagg-b200-agentic-frontend:8000`  | `Inferact/Qwen3.8-Flash-Next-NVFP4` | `/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` |

If you run more than one benchmark in the same namespace, also update `metadata.name` / `labels.app` so jobs and artifact directories stay distinct.

## Dataset

The benchmark replays a [Mooncake-format](https://github.com/kvcache-ai/Mooncake) trace via `aiperf --custom-dataset-type mooncake_trace`. Each JSONL line describes one request (`input_length`, `output_length`, `hash_ids`).

**Agentic trace** — `64k_400_90kv_agent_new_noschedule_short_15perc.jsonl`:

| Metric | Value |
| ------ | ----- |
| Requests | 3,541 (15% subset) |
| ISL median | 67,585 |
| ISL p90 | 101,392 |
| OSL median | 399 |
| OSL p90 | 6,943 |
| KV cache hit rate | 90% (prefix reuse) |
| Shared system prompt | ~57,600 tokens |

For shorter runs (smoke tests, faster iteration), use a smaller subset. Typical staging:

```
/model-cache/traces/<flavour>.jsonl                 # full
/model-cache/traces/<flavour>_short_30perc.jsonl    # ~30% subset
/model-cache/traces/<flavour>_short_15perc.jsonl    # ~15% subset
```

These are workload-shape traces (not model-specific). Stage your own Mooncake-format JSONLs at the path you set in `TRACE_FILE`.

## Workflow

```bash
export NAMESPACE=your-namespace
```

### 1. Deploy the DGD

See instructions in the [recipe README](../README.md).

### 2. Stage the trace on the PVC

The benchmark pod **auto-downloads** the trace from GitHub if it's not
already on the PVC. No manual staging is required.

If you prefer to pre-stage traces manually (e.g. for air-gapped clusters),
spin up a short-lived helper pod that mounts `model-cache`, then `kubectl cp`
the traces in:

```bash
kubectl run pvc-helper -n ${NAMESPACE} \
  --image=busybox:1.36 --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"helper","image":"busybox:1.36","command":["sleep","3600"],"volumeMounts":[{"name":"model-cache","mountPath":"/model-cache"}]}],"volumes":[{"name":"model-cache","persistentVolumeClaim":{"claimName":"model-cache"}}]}}' \
  --command -- sleep 3600

kubectl cp ./traces ${NAMESPACE}/pvc-helper:/model-cache/
```

Keep `pvc-helper` around for fetching artifacts later, or `kubectl delete pod pvc-helper -n ${NAMESPACE}` once you're done staging.

### 3. Run the benchmark

```bash
kubectl apply -f perf.yaml -n ${NAMESPACE}

# Stream logs
kubectl logs -n ${NAMESPACE} -l job-name=qwen38fn-bench -f

# Wait for completion (2h hard cap on the Job)
kubectl wait --for=condition=Complete \
  job/qwen38fn-bench \
  -n ${NAMESPACE} --timeout=7200s
```

### 4. Fetch artifacts

```bash
kubectl cp ${NAMESPACE}/pvc-helper:/model-cache/perf/<epoch>_qwen38fn-bench ./results
```

### 5. Cleanup

```bash
kubectl delete job qwen38fn-bench -n ${NAMESPACE}
kubectl delete pod pvc-helper -n ${NAMESPACE}   # if you kept it around
```

## Running a concurrency sweep

`perf.yaml` runs a **single** `CONCURRENCY` value. To measure multiple concurrencies you must clear server state between runs — otherwise residual KV cache / prefix-cache hits from the previous run skew results.

For each concurrency value you want to measure:

```bash
# 1. Delete the previous bench job
kubectl delete job qwen38fn-bench -n ${NAMESPACE} --ignore-not-found

# 2. Drop KV / prefix-cache by deleting the worker pods; Grove respawns them
DGD=qwen38fn-agg-b200-agentic   # or qwen38fn-disagg-b200-agentic
kubectl delete pods -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD},nvidia.com/dynamo-component-type=worker

# 3. Bump CONCURRENCY in perf.yaml, then re-apply
kubectl apply -f perf.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/qwen38fn-bench -n ${NAMESPACE} --timeout=7200s
```

Recommended concurrency sweep for B200: C=4, 8, 16, 24, 32.

## Tunable environment variables

Edit the `env` block on the `Job` to adjust:

| Variable       | Default                                                                      | Notes                                                                           |
| -------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `ENDPOINT`     | `qwen38fn-agg-b200-agentic-frontend:8000`                               | DGD frontend service:port — change per variant                                  |
| `TRACE_FILE`   | `/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` | Swap to a smaller subset for shorter runs                                       |
| `CONCURRENCY`  | `24`                                                                         | Single value — see [Running a concurrency sweep](#running-a-concurrency-sweep)  |
| `TARGET_MODEL` | `Inferact/Qwen3.8-Flash-Next-NVFP4`                                          | Must match `--served-model-name` on the DGD frontend                             |

## Artifacts

Results are written to:

```
/model-cache/perf/<epoch>_<job-name>/
  warmup/
  Qwen3.8-Flash-Next-NVFP4_trace_c<concurrency>_<timestamp>/
    profile_export.json
    inputs.json
    ...
```
