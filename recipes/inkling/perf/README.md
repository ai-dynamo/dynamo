<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Inkling-NVFP4 Benchmark Recipe

A single [AIPerf](https://github.com/ai-dynamo/aiperf) trace-replay Job —
[`perf.yaml`](perf.yaml) — covers both Inkling vLLM GB300 DGDs. The benchmark is
identical across variants; only `ENDPOINT` and `CONCURRENCY` change.

The Job waits for the target model on the DGD frontend, runs a short warmup,
replays the configured trace at one `CONCURRENCY` value, and writes raw
artifacts to the shared `model-cache` PVC. The benchmark pod is co-located with
a DGD frontend through `podAffinity`.

## Targeting a variant

Edit the `env` block in [`perf.yaml`](perf.yaml):

| Variant target | `ENDPOINT` | `CONCURRENCY` |
| --- | --- | --- |
| GB300 aggregated agentic | `inkling-vllm-gb300-agg-agentic-frontend:8000` | `20` |
| GB300 disaggregated agentic | `inkling-vllm-gb300-disagg-agentic-frontend:8000` | `16` |

If you run more than one benchmark in the same namespace, also update
`metadata.name` and `labels.app` so Jobs and artifact directories stay
distinct.

## Dataset

The benchmark replays a
[Mooncake-format](https://github.com/kvcache-ai/Mooncake) trace through
`--custom-dataset-type mooncake_trace`. Each JSONL line describes one request
with `input_length`, `output_length`, `hash_ids`, and `timestamp`.

Use the 3,541-request agentic trace in [`traces`](traces) (Git LFS) — 64K median
ISL, 400 median OSL, 90% KV cache hit. AIPerf builds its replay schedule from a
`timestamp` field; the shipped rows have none, and without it AIPerf replays a
small default instead of the file, so add one:

```bash
git lfs install
git lfs pull --include "recipes/*/perf/traces/*"
python3 -c "
import json
src='traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl'
for line in open(src):
    print(json.dumps({'timestamp': 0, **json.loads(line)}))
" > mooncake_agentic.jsonl
```

Confirm a run replayed the whole file: `Request Count` in `profile_export_aiperf.csv`
should account for every line, successes and errors together.

## Workflow

```bash
export NAMESPACE=your-namespace
```

### 1. Deploy the DGD

See the deployment instructions in the [recipe README](../README.md).

### 2. Stage the trace on the PVC

Copy `mooncake_agentic.jsonl` through a helper pod that mounts `model-cache`:

```bash
kubectl run pvc-helper -n ${NAMESPACE} \
  --image=busybox:1.36 --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"helper","image":"busybox:1.36","command":["sleep","3600"],"volumeMounts":[{"name":"model-cache","mountPath":"/model-cache"}]}],"volumes":[{"name":"model-cache","persistentVolumeClaim":{"claimName":"model-cache"}}]}}' \
  --command -- sleep 3600

kubectl exec -n ${NAMESPACE} pvc-helper -- mkdir -p /model-cache/traces
kubectl cp mooncake_agentic.jsonl ${NAMESPACE}/pvc-helper:/model-cache/traces/
```

Keep `pvc-helper` for fetching artifacts, or delete it after staging.

### 3. Run the benchmark

```bash
kubectl apply -f perf.yaml -n ${NAMESPACE}
kubectl logs -n ${NAMESPACE} -l job-name=inkling-bench -f
kubectl wait --for=condition=Complete job/inkling-bench \
  -n ${NAMESPACE} --timeout=10800s
```

The Job uses `nvcr.io/nvidia/ai-dynamo/aiperf:0.11.0` directly and does not
install or patch AIPerf at runtime.

### 4. Fetch artifacts

```bash
kubectl cp \
  ${NAMESPACE}/pvc-helper:/model-cache/perf/<epoch>_inkling-bench \
  ./results
```

### 5. Cleanup

```bash
kubectl delete job inkling-bench -n ${NAMESPACE}
kubectl delete pod pvc-helper -n ${NAMESPACE}
```

## Running a concurrency sweep

`perf.yaml` runs one `CONCURRENCY` value. Clear vLLM KV state and Dynamo
frontend/router state between independent runs by restarting the DGD pods:

```bash
kubectl delete job inkling-bench -n ${NAMESPACE} --ignore-not-found

DGD=inkling-vllm-gb300-agg-agentic # or inkling-vllm-gb300-disagg-agentic
kubectl delete pods -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD}
kubectl wait --for=condition=Ready pod -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD} \
  --timeout=7200s

# Update CONCURRENCY in perf.yaml before each run.
kubectl apply -f perf.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/inkling-bench \
  -n ${NAMESPACE} --timeout=10800s
```

Do not compare partial runs. A completed run must account for successful,
errored, and unfinished requests before reporting aggregate throughput.

## Reference results

Measured against the SLO pair `user_tps p50 >= 50` and `TTFT p50 <= 5000 ms`.
Both variants clear both gates.

| Variant | GPUs | Concurrency | tok/s/GPU | user_tps p50 | TTFT p50 |
| --- | --- | --- | --- | --- | --- |
| Aggregated | 8 (2x TP4) | 20 | 192.41 | 84.87 | 361 ms |
| Disaggregated | 8 (4P + 4D) | 16 | 144.01 | 77.94 | 1305 ms |

Measured on the `aws-roce` fabric variant.

> [!NOTE]
> Aggregated beat every disaggregated split tested on this workload: prefill is
> only about 5% of the GPU-time budget, so dedicating half the fleet to it costs
> more per-GPU throughput than the prefill/decode separation returns.

## Artifacts

Results are written to:

```text
/model-cache/perf/<epoch>_<job-name>/
  warmup/
  Inkling-NVFP4_trace_c<concurrency>_<timestamp>/
    profile_export_aiperf.json
    inputs.json
    ...
```
