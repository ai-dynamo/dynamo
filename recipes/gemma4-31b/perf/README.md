<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Gemma 4 31B Benchmark Recipe

A single [AIPerf](https://github.com/ai-dynamo/aiperf) trace-replay Job,
[`perf.yaml`](perf.yaml), covers all three Gemma 4 31B DGDs. Set `ENDPOINT`,
`TARGET_MODEL`, `TRACE_FILE`, and `CONCURRENCY` for the target variant.

The Job waits for the target model on the Dynamo frontend, runs a short warmup,
replays the configured trace at one `CONCURRENCY` value, and writes raw
artifacts to the shared `model-cache` persistent volume claim (PVC). The
benchmark pod is co-located with a DGD frontend through `podAffinity`.

## Targeting a Variant

Edit the `env` block in [`perf.yaml`](perf.yaml) with the values from the target
row. Also update the `podAffinity` `values` list to contain only the target DGD
name so the benchmark pod is co-located with the correct frontend.

| Variant | DGD affinity | `ENDPOINT` | `TARGET_MODEL` | `TRACE_FILE` | `CONCURRENCY` |
| --- | --- | --- | --- | --- | --- |
| B200 aggregate agentic | `gemma4-31b-agg-b200-agentic` | `gemma4-31b-agg-b200-agentic-frontend:8000` | `nvidia/Gemma-4-31B-IT-NVFP4` | `/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` | `192` |
| GB200 aggregate agentic | `gemma4-31b-agg-gb200-agentic` | `gemma4-31b-agg-gb200-agentic-frontend:8000` | `nvidia/Gemma-4-31B-IT-NVFP4` | `/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` | `192` |
| H200 aggregate agentic | `gemma4-31b-agg-h200-agentic` | `gemma4-31b-agg-h200-agentic-frontend:8000` | `google/gemma-4-31B-it` | `/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` | `32` |

<!--
When adding a variant, add its values to this table and keep the matching
defaults in `perf.yaml` on the recommended target. If multiple benchmark Jobs
run in one namespace, give each Job a distinct `metadata.name` and
`metadata.labels.app` so their logs and artifact directories remain separate.
-->

## Dataset

The benchmark uses an
[AIPerf](https://github.com/ai-dynamo/aiperf) Mooncake-format trace with
`--custom-dataset-type mooncake_trace`. Each JSONL line contains
`input_length`, `output_length`, and `hash_ids` fields.

The trace represents an agentic workload with a median 64K input sequence
length (ISL), median 400-token output sequence length (OSL), and approximately
90% KV-cache reuse. The checked-in path is a symbolic link to the shared trace
under the Kimi K2.6 recipe:

```text
traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl
  -> ../../../kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl
```

The 15% subset contains 3,541 requests. Its SHA-256 is
`f20d3f2bc83dd1306cda659fbe34e7c4d85ca5497626c98bc0b1c4d2211379d0`.

## Workflow

```bash
export NAMESPACE=your-namespace
```

### 1. Deploy Gemma 4

Follow the deployment instructions in the [Gemma 4 recipe README](../README.md)
and wait for the selected DGD to become ready. Use the DGD affinity value from
the target table when configuring `perf.yaml`.

### 2. Stage the trace

Materialize the Git LFS object, then copy it to the shared PVC through a helper
pod:

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

Keep `pvc-helper` running to fetch the benchmark artifacts.

### 3. Run the benchmark

Delete any previous benchmark Job before creating a run. Kubernetes does not
allow updates to a Job's pod template after the Job is created.

```bash
kubectl delete job gemma4-31b-bench -n ${NAMESPACE} --ignore-not-found
kubectl create -f perf.yaml -n ${NAMESPACE}
kubectl logs -n ${NAMESPACE} -l job-name=gemma4-31b-bench -f
kubectl wait --for=condition=Complete job/gemma4-31b-bench \
  -n ${NAMESPACE} --timeout=10800s
```

The Job uses `nvcr.io/nvidia/ai-dynamo/aiperf:0.11.0` directly. It does not
install or patch AIPerf at runtime.

### 4. Fetch the artifacts

```bash
kubectl cp \
  ${NAMESPACE}/pvc-helper:/model-cache/perf/<epoch>_gemma4-31b-bench \
  ./results
```

### 5. Clean up

```bash
kubectl delete job gemma4-31b-bench -n ${NAMESPACE}
kubectl delete pod pvc-helper -n ${NAMESPACE}
```

## Concurrency Sweep

`perf.yaml` runs one concurrency value at a time. Restart the TensorRT-LLM
workers and Dynamo frontend between independent points to clear KV-cache and
router state:

```bash
kubectl delete job gemma4-31b-bench -n ${NAMESPACE} --ignore-not-found

DGD=gemma4-31b-agg-b200-agentic # Choose a DGD affinity value from the target table.
kubectl delete pods -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD}
kubectl wait --for=condition=Ready pod -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD} \
  --timeout=7200s

# Update CONCURRENCY in perf.yaml before each run.
kubectl create -f perf.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/gemma4-31b-bench \
  -n ${NAMESPACE} --timeout=10800s
```

Do not compare partial runs. A completed run must account for successful,
errored, and unfinished requests before reporting aggregate throughput.

## Tunable Environment Variables

| Variable | Initial value | Notes |
| --- | --- | --- |
| `ENDPOINT` | `gemma4-31b-agg-b200-agentic-frontend:8000` | Change per DGD variant |
| `TRACE_FILE` | `/model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` | 3,541-request agentic trace |
| `CONCURRENCY` | `192` | B200 starting point; sweep and record a value for GB200 or H200 |
| `TARGET_MODEL` | `nvidia/Gemma-4-31B-IT-NVFP4` | Change per DGD variant; must match `--served-model-name` |

## Artifacts

Results are written under:

```text
/model-cache/perf/<epoch>_gemma4-31b-bench/
  warmup/
  <model-name>_trace_c<concurrency>_<timestamp>/
    profile_export_aiperf.json
    inputs.json
    ...
```
