<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4 Benchmark Recipe

This directory provides a shared AIPerf trace-replay Job for DeepSeek-V4 Dynamo
DGDs. It follows the `recipes/kimi-k2.6/perf` pattern: one Job, configured by
environment variables, writes artifacts to the model-cache PVC.

## Targeting a Variant

Edit the `env` block in `perf.yaml`.

| Target | `ENDPOINT` | `TARGET_MODEL` | Default trace |
|---|---|---|---|
| Pro AGG B200 | `dsv4-pro-agg-b200-agentic-frontend:8000` | `nvidia/DeepSeek-V4-Pro-NVFP4` | agentic |
| Pro AGG H200 | `dsv4-pro-agg-h200-agentic-frontend:8000` | `deepseek-ai/DeepSeek-V4-Pro` | agentic |
| Pro DisAgg B200 | `dsv4-pro-disagg-b200-agentic-frontend:8000` | `nvidia/DeepSeek-V4-Pro-NVFP4` | agentic |
| Pro DisAgg H200 | `dsv4-pro-disagg-h200-agentic-frontend:8000` | `deepseek-ai/DeepSeek-V4-Pro` | agentic |
| Flash AGG B200 | `dsv4-flash-agg-b200-agentic-frontend:8000` | `nvidia/DeepSeek-V4-Flash-NVFP4` | agentic |
| Flash AGG H200 | `dsv4-flash-agg-h200-agentic-frontend:8000` | `deepseek-ai/DeepSeek-V4-Flash` | agentic |
| Flash DisAgg B200 | `dsv4-flash-disagg-b200-agentic-frontend:8000` | `nvidia/DeepSeek-V4-Flash-NVFP4` | agentic |
| Flash DisAgg H200 | `dsv4-flash-disagg-h200-agentic-frontend:8000` | `deepseek-ai/DeepSeek-V4-Flash` | agentic |

`TARGET_MODEL` must match each manifest's `--served-model-name` (B200 → `…-NVFP4`,
H200 → the public checkpoint) as reported by `GET /v1/models`.

## Dataset

The Job expects Mooncake-format JSONL traces on the model-cache PVC:

```text
/model-cache/traces/64k_400_90kv_agent_new_noschedule.jsonl
/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_30perc.jsonl
/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl
```

These agentic traces are vendored in [`traces/`](traces) via Git LFS (full +
15% / 30% subsets). Stage them onto the model-cache PVC before running the Job.

## Workflow

Run all commands below from the `recipes/deepseek-v4/` directory (paths are relative to it).

```bash
export NAMESPACE=your-namespace
```

### 1. Stage Traces

Trace JSONL files are vendored via Git LFS; fetch their content first (a fresh clone has only LFS pointer files), then copy them onto the PVC. `kubectl run` returns before the pod is Ready, so wait for it before `kubectl cp`:

```bash
git lfs pull --include="recipes/deepseek-v4/perf/traces/*.jsonl"

kubectl run pvc-helper -n ${NAMESPACE} \
  --image=busybox:1.36 --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"helper","image":"busybox:1.36","command":["sleep","3600"],"volumeMounts":[{"name":"model-cache","mountPath":"/model-cache"}]}],"volumes":[{"name":"model-cache","persistentVolumeClaim":{"claimName":"model-cache"}}]}}' \
  --command -- sleep 3600

kubectl wait --for=condition=Ready pod/pvc-helper -n ${NAMESPACE} --timeout=120s

kubectl cp perf/traces ${NAMESPACE}/pvc-helper:/model-cache/
```

### 2. Run One Concurrency

```bash
kubectl apply -f perf/perf.yaml -n ${NAMESPACE}
kubectl logs -n ${NAMESPACE} -l job-name=dsv4-bench -f
kubectl wait --for=condition=Complete job/dsv4-bench -n ${NAMESPACE} --timeout=7200s
```

### 3. Fetch Artifacts

```bash
kubectl cp ${NAMESPACE}/pvc-helper:/model-cache/perf/<epoch>_dsv4-bench ./results
```

### 4. Reset Between Independent Runs

For a concurrency sweep, run one concurrency per Job and reset server state
between rows:

1. Delete the previous benchmark Job.
2. Stop clients and wait for in-flight requests to drain.
3. Delete/restart worker pods to clear backend KV/prefix-cache state.
4. Restart the frontend/router if testing KV-aware routing so router state is
   also cleared.
5. Wait for `/v1/models`, run a chat smoke, then launch the next Job.

Do not rely on `cache_salt` for Dynamo unless that support has been explicitly
validated for the deployed frontend/backend path.

## Tunable Environment Variables

| Variable | Default | Notes |
|---|---|---|
| `ENDPOINT` | `dsv4-pro-agg-b200-agentic-frontend:8000` | DGD frontend service and port |
| `TARGET_MODEL` | `nvidia/DeepSeek-V4-Pro-NVFP4` | Must match `/v1/models` |
| `TRACE_FILE` | `/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` | Mooncake JSONL on PVC |
| `CONCURRENCY` | `16` | Single value per Job |
| `REQUEST_TIMEOUT_SECONDS` | `1200` | Long enough for long-context requests |
| `AIPERF_VERSION` | `0.10.0` | Match local benchmark runs unless updated intentionally |

## Required Evidence for a Result Row

```text
deploy manifest path
image ref/digest
endpoint
target model
trace path and sha256
concurrency
reset sequence
profile_export_aiperf.json
metrics summary
frontend/router metrics when KV-aware routing is enabled
worker prefix-cache metrics
```
