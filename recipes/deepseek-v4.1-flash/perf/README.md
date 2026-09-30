<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4.1-Flash vLLM benchmark

A single [AIPerf](https://github.com/ai-dynamo/aiperf) trace-replay Job —
[`perf.yaml`](perf.yaml) — covers all five vLLM DGDs. It replays the trace
at one `CONCURRENCY` value against a ready DGD frontend and
writes raw artifacts to the shared `shared-model-cache` PVC. It uses the AIPerf
0.10.0 container to match the recorded client version.

## Targeting a variant

Edit the `env` block in [`perf.yaml`](perf.yaml). The `podAffinity` already
lists every deployment name, so it needs no edit: only one target is deployed
at a time and the Job lands beside whichever frontend exists.

| Variant | `ENDPOINT` |
| --- | --- |
| B200 aggregated | `dsv41-flash-vllm-b200-agg-frontend:8000` |
| B200 disaggregated | `dsv41-flash-vllm-b200-disagg-frontend:8000` |
| GB200 aggregated | `dsv41-flash-vllm-gb200-agg-frontend:8000` |
| GB200 disaggregated | `dsv41-flash-vllm-gb200-disagg-frontend:8000` |
| H200 aggregated | `dsv41-flash-vllm-h200-agg-frontend:8000` |

Set `CONCURRENCY` for the operating point being measured. The Job's default
of 168 matches the recorded B200 aggregated point. The recorded B200
disaggregated point uses 184; both GB200 replay points use 168; H200 aggregated uses 80. These are
measured points, not a claim that every other concurrency was worse. Running more than one benchmark in the same namespace needs a
distinct `metadata.name` and `labels.app` so Jobs and artifacts remain separate.

## SLO

Evaluate each operating point against both p50 requirements:

| Metric | Floor |
| --- | --- |
| Output token throughput per user | >= 50 tok/s |
| Time to first token | < 5000 ms |

> [!NOTE]
> `total_output_tokens` counts the non-reasoning subset only. Add
> `total_reasoning_tokens` to get what the GPU actually produced.

## Dataset

The benchmark replays a
[Mooncake-format](https://github.com/kvcache-ai/Mooncake) trace through
the AIPerf 0.10.0 `mooncake_trace` dataset format with sequential sampling. Each JSONL line describes one request
with `input_length`, `output_length`, and `hash_ids`.

This is the same 64K-ISL / 400-OSL / 90%-KV-reuse agentic trace the other
agentic recipes use, so rather than duplicate the Git LFS blob it is referenced
from the DeepSeek-V4 family recipes through a symlink under [`traces`](traces):

```text
traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl
  -> ../../../deepseek-v4/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl
```

The trace contains 3,541 requests. The profiling Job saves its resolved
`client.yaml` and trace checksum with the results.

## Workflow

```bash
export NAMESPACE=your-namespace
```

### 1. Deploy the DGD

See the deployment instructions in [the recipe README](../README.md).

### 2. Stage the trace on the PVC

Materialize the Git LFS trace file, then copy it through a helper pod that
mounts `shared-model-cache`:

```bash
git lfs pull --include='recipes/deepseek-v4/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl'

kubectl run pvc-helper -n ${NAMESPACE} \
  --image=busybox:1.36 --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"helper","image":"busybox:1.36","command":["sleep","86400"],"volumeMounts":[{"name":"shared-model-cache","mountPath":"/shared-model-cache"}]}],"volumes":[{"name":"shared-model-cache","persistentVolumeClaim":{"claimName":"shared-model-cache"}}]}}' \
  --command -- sleep 86400

kubectl wait --for=condition=Ready pod/pvc-helper -n "${NAMESPACE}" --timeout=300s
TRACE_SOURCE="$(git rev-parse --show-toplevel)/recipes/deepseek-v4/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl"
kubectl exec -n "${NAMESPACE}" pvc-helper -- mkdir -p /shared-model-cache/traces
kubectl cp "${TRACE_SOURCE}" \
  "${NAMESPACE}/pvc-helper:/shared-model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl"
```

Keep `pvc-helper` for fetching artifacts afterwards, or delete it once staging
is done. It sleeps for 24 h so it outlives the benchmark Job.

### 3. Run the benchmark

Run after the deployment is ready and the recipe smoke test passes.

```bash
kubectl apply -f perf.yaml -n ${NAMESPACE}
kubectl logs -n ${NAMESPACE} -l job-name=dsv41-flash-vllm-bench -f
kubectl wait --for=condition=Complete job/dsv41-flash-vllm-bench -n ${NAMESPACE} --timeout=86400s
```

Results land under `/shared-model-cache/perf/<epoch>_<job-name>/trace_c<CONCURRENCY>/`.

## Measured Results

Measured vLLM configurations on the 64K-input / 400-output agentic workload,
using eight GPUs per Blackwell target and 16 GPUs for H200 aggregated. Output throughput includes reasoning tokens.
These are selected operating points, not a controlled topology-only comparison;
Blackwell runtime qualification remains pending. H200 P0 passed on one TP4
worker, including near-1M context; it does not qualify four-worker routing.

Each run completed 3,526 requests with 15 over-context errors (AIPerf 0.10.0).

The archived H200 measurement (H20) used 350 warm-up requests followed by a KV
reset without recreating the frontend; the generic `perf.yaml` does not
reproduce that exact replay.

| Target | Concurrency | Output tok/s/GPU | Output tok/s/user p50 |
| --- | ---: | ---: | ---: |
| B200 aggregated | 168 | 990.57 | 54.69 |
| B200 disaggregated | 184 | 1,087.71 | 82.12 |
| GB200 aggregated | 168 | 953.08 | 51.83 |
| GB200 disaggregated | 168 | 1,154.87 | 80.85 |
| H200 aggregated | 80 | 209.18 | 51.32 |

### TTFT Distribution

Milliseconds across successful requests:

| Target | Mean | p50 | p75 | p90 | p95 | p99 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B200 aggregated | 1,408.61 | 178.66 | 775.00 | 3,068.28 | 5,552.72 | 26,864.62 | 59,884.76 |
| B200 disaggregated | 18,597.21 | 135.12 | 2,002.87 | 90,076.83 | 126,866.47 | 168,049.68 | 256,824.39 |
| GB200 aggregated | 1,601.63 | 286.75 | 951.37 | 3,524.82 | 6,980.67 | 26,793.16 | 52,808.90 |
| GB200 disaggregated | 12,149.79 | 169.02 | 1,126.26 | 57,116.54 | 100,309.52 | 124,167.34 | 160,272.44 |
| H200 aggregated | 2,314.58 | 171.29 | 1,365.00 | 6,368.76 | 11,694.66 | 30,075.76 | 108,304.39 |

### ITL Distribution

Milliseconds across **per-request average token intervals**. These are not
percentiles of all individual token gaps pooled together.

| Target | Mean | p50 | p75 | p90 | p95 | p99 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B200 aggregated | 25.06 | 18.28 | 25.07 | 41.42 | 76.17 | 120.94 | 218.89 |
| B200 disaggregated | 12.64 | 12.18 | 14.34 | 16.52 | 18.58 | 25.03 | 44.31 |
| GB200 aggregated | 26.40 | 19.29 | 27.22 | 48.76 | 83.17 | 110.46 | 321.77 |
| GB200 disaggregated | 13.19 | 12.37 | 14.61 | 17.75 | 21.14 | 30.59 | 81.71 |
| H200 aggregated | 29.33 | 19.49 | 29.90 | 54.01 | 84.55 | 183.57 | 381.82 |

At these operating points, disaggregated configurations show higher output
throughput and lower ITL, with longer TTFT tails. GB200 records 21% higher
output tok/s/GPU and 56% higher p50 output tok/s/user; TTFT p90 rises from
3.52 s to 57.12 s.
