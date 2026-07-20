<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.2 Benchmark Recipe

A single [AIPerf](https://github.com/ai-dynamo/aiperf) trace-replay Job —
[`perf.yaml`](perf.yaml) — covers all four GLM-5.2 DGDs. Set `ENDPOINT` for the
target DGD and `SYNTHESIS_MAX_ISL` for its context limit.

The Job waits for the target model on the DGD frontend, runs a short warmup,
replays the configured trace at one `CONCURRENCY` value, and writes raw
artifacts to the shared `model-cache` PVC. The benchmark pod is co-located with
a DGD frontend through `podAffinity`.

## Targeting a variant

Edit the `env` block in [`perf.yaml`](perf.yaml):

| Variant target | `ENDPOINT` | `SYNTHESIS_MAX_ISL` | `TRACE_FILE` |
| --- | --- | --- | --- |
| B200 aggregate agentic | `glm52-agg-b200-agentic-frontend:8000` | `500000` | `/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` |
| B200 disaggregated agentic | `glm52-disagg-b200-agentic-frontend:8000` | `500000` | `/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` |
| H200 aggregate agentic | `glm52-agg-h200-agentic-frontend:8000` | `250000` | `/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` |
| H200 disaggregated agentic | `glm52-disagg-h200-agentic-frontend:8000` | `250000` | `/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` |

If you run more than one benchmark in the same namespace, also update
`metadata.name` and `labels.app` so Jobs and artifact directories stay
distinct.

## Dataset

The benchmark replays a
[Mooncake-format](https://github.com/kvcache-ai/Mooncake) trace through
`--custom-dataset-type mooncake_trace`. Each JSONL line describes one request
with `input_length`, `output_length`, and `hash_ids`.

This recipe benchmarks the same 64K-ISL / 400-OSL / 90%-KV-reuse agentic trace
shared across the agentic recipes, so rather than duplicate the Git-LFS blob it
is referenced from the Kimi-K2.6 recipe via a symlink under [`traces`](traces):

```text
traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl
  -> ../../../kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl
```

The default 15% trace contains 3,541 requests. Its SHA-256 is
`f20d3f2bc83dd1306cda659fbe34e7c4d85ca5497626c98bc0b1c4d2211379d0`.

## Workflow

```bash
export NAMESPACE=your-namespace
```

### 1. Deploy the DGD

See the deployment instructions in the [recipe README](../README.md).

### 2. Stage the trace on the PVC

Materialize the Git LFS trace files, then copy them through a helper pod that
mounts `model-cache`:

```bash
git lfs pull --include='recipes/kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl'

kubectl run pvc-helper -n ${NAMESPACE} \
  --image=busybox:1.36 --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"helper","image":"busybox:1.36","command":["sleep","3600"],"volumeMounts":[{"name":"model-cache","mountPath":"/model-cache"}]}],"volumes":[{"name":"model-cache","persistentVolumeClaim":{"claimName":"model-cache"}}]}}' \
  --command -- sleep 3600

kubectl cp ./traces ${NAMESPACE}/pvc-helper:/model-cache/
```

Keep `pvc-helper` for fetching artifacts, or delete it after staging.

### 3. Run the benchmark

```bash
kubectl apply -f perf.yaml -n ${NAMESPACE}
kubectl logs -n ${NAMESPACE} -l job-name=glm52-bench -f
kubectl wait --for=condition=Complete job/glm52-bench \
  -n ${NAMESPACE} --timeout=10800s
```

The Job uses `nvcr.io/nvidia/ai-dynamo/aiperf:0.11.0` directly and does
not install or patch AIPerf at runtime.

### 4. Fetch artifacts

```bash
kubectl cp \
  ${NAMESPACE}/pvc-helper:/model-cache/perf/<epoch>_glm52-bench \
  ./results
```

### 5. Cleanup

```bash
kubectl delete job glm52-bench -n ${NAMESPACE}
kubectl delete pod pvc-helper -n ${NAMESPACE}
```

## Running a concurrency sweep

`perf.yaml` runs one `CONCURRENCY` value. Clear SGLang KV state and Dynamo
frontend/router state between independent runs:

```bash
kubectl delete job glm52-bench -n ${NAMESPACE} --ignore-not-found

DGD=glm52-agg-b200-agentic # Choose one of the four variant names above.
kubectl delete pods -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD}
kubectl wait --for=condition=Ready pod -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD} \
  --timeout=7200s

# Update CONCURRENCY in perf.yaml before each run.
kubectl apply -f perf.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/glm52-bench \
  -n ${NAMESPACE} --timeout=10800s
```

Do not compare partial runs. A completed run must account for successful,
errored, and unfinished requests before reporting aggregate throughput.

## Tunable environment variables

| Variable | Default | Notes |
| --- | --- | --- |
| `ENDPOINT` | `glm52-agg-b200-agentic-frontend:8000` | Change per DGD variant |
| `TRACE_FILE` | `/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` | 3,541-request 15% agent trace |
| `SYNTHESIS_MAX_ISL` | `500000` | Use `250000` for H200 recipes |
| `CONCURRENCY` | `64` | Single value; reset server state between values |
| `TARGET_MODEL` | `zai-org/GLM-5.2` | Must match `--served-model-name` |

## Artifacts

Results are written to:

```text
/model-cache/perf/<epoch>_<job-name>/
  warmup/
  GLM-5.2_trace_c<concurrency>_<timestamp>/
    profile_export_aiperf.json
    inputs.json
    ...
```
