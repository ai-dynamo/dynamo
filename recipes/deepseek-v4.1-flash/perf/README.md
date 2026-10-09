<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4.1-Flash vLLM benchmark

Use the [AIPerf 0.10.0 Job](perf.yaml) to replay the agentic trace against a
running deployment. Results are saved to the `shared-model-cache` volume.

## Select a Target

Set `ENDPOINT` and `CONCURRENCY` in [`perf.yaml`](perf.yaml):

| Target | `ENDPOINT` | `CONCURRENCY` |
| --- | --- | ---: |
| B200 aggregated | `dsv41-flash-vllm-b200-agg-agentic-frontend:8000` | 168 |
| B200 disaggregated | `dsv41-flash-vllm-b200-disagg-agentic-frontend:8000` | 184 |
| GB200 aggregated | `dsv41-flash-vllm-gb200-agg-agentic-frontend:8000` | 168 |
| GB200 disaggregated | `dsv41-flash-vllm-gb200-disagg-agentic-frontend:8000` | 168 |
| H200 aggregated | `dsv41-flash-vllm-h200-agg-agentic-frontend:8000` | 80 |
| H200 disaggregated | `dsv41-flash-vllm-h200-disagg-agentic-frontend:8000` | 64 |

Run one target per namespace. The benchmark Job is scheduled on the same node
as the frontend.

## Performance Targets

The benchmark targets are:

| Metric (p50) | Target |
| --- | --- |
| Output token throughput per user | >= 50 tok/s |
| Time to first token | < 5000 ms |

> [!NOTE]
> `total_output_tokens` counts the non-reasoning subset only. Add
> `total_reasoning_tokens` when calculating total output throughput.

## Dataset

The benchmark replays a
[Mooncake-format](https://github.com/kvcache-ai/Mooncake) trace through
the AIPerf 0.10.0 `mooncake_trace` dataset format with sequential sampling. Each JSONL line describes one request
with `input_length`, `output_length`, and `hash_ids`.

The [trace](traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl)
contains 3,541 requests with approximately 64K input tokens, 400 output tokens,
and 90% KV reuse. The Job saves its resolved `client.yaml` and trace checksum
with the results.

## Workflow

```bash
export NAMESPACE=your-namespace
```

### 1. Deploy the DGD

See the [Dynamo recipe documentation](https://docs.nvidia.com/dynamo/dev/recipes/deepseek-v4-1-flash) for deployment instructions.

### 2. Stage the trace on the PVC

The Flash trace links to a shared Git LFS file under `deepseek-v4`. Pull that
file, then copy the resolved trace through a helper pod that mounts
`shared-model-cache`:

```bash
git lfs pull --include='recipes/deepseek-v4/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl'

kubectl run pvc-helper -n ${NAMESPACE} \
  --image=busybox:1.36 --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"helper","image":"busybox:1.36","command":["sleep","86400"],"volumeMounts":[{"name":"shared-model-cache","mountPath":"/shared-model-cache"}]}],"volumes":[{"name":"shared-model-cache","persistentVolumeClaim":{"claimName":"shared-model-cache"}}]}}' \
  --command -- sleep 86400

kubectl wait --for=condition=Ready pod/pvc-helper -n "${NAMESPACE}" --timeout=300s
TRACE_SOURCE="$(realpath "$(git rev-parse --show-toplevel)/recipes/deepseek-v4.1-flash/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl")"
kubectl exec -n "${NAMESPACE}" pvc-helper -- mkdir -p /shared-model-cache/traces
kubectl cp "${TRACE_SOURCE}" \
  "${NAMESPACE}/pvc-helper:/shared-model-cache/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl"
```

Keep `pvc-helper` for fetching artifacts afterwards, or delete it once staging
is done. It sleeps for 24 h so it outlives the benchmark Job.

### 3. Run the benchmark

From `recipes/deepseek-v4.1-flash/perf`, run:

```bash
kubectl apply -f perf.yaml -n ${NAMESPACE}
kubectl logs -n ${NAMESPACE} -l job-name=dsv41-flash-vllm-bench -f
kubectl wait --for=condition=Complete job/dsv41-flash-vllm-bench -n ${NAMESPACE} --timeout=86400s
```

Results land under `/shared-model-cache/perf/<epoch>_<job-name>/trace_c<CONCURRENCY>/`.

To rerun, wait for the previous Job to finish and save its logs. Results remain
on the PVC. Delete the completed Job, update `perf.yaml`, then run the commands
above again. Do not delete an active benchmark.

```bash
kubectl delete job dsv41-flash-vllm-bench -n "${NAMESPACE}"
```

## Measured Results

Results for the agentic workload (64K input tokens, 400 output tokens), using
eight B200/GB200 GPUs, 16 H200 GPUs for aggregated serving, or eight H200 GPUs for disaggregated serving. Output throughput includes reasoning tokens.

Each run completed 3,526 requests with 15 over-context errors (AIPerf 0.10.0).

| Workload | Recipe | Framework | SKU | Concurrency | System output tok/s/GPU | User output tok/s (P50) | TTFT P50 (ms) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Agentic (64K input, 400 output) | Aggregated (2 × TP4) | vLLM | B200 | 168 | 990.57 | 54.69 | 178.66 |
| Agentic (64K input, 400 output) | Disaggregated (1 prefill, 1 decode; TP4 each) | vLLM | B200 | 184 | 1,087.71 | 82.12 | 135.12 |
| Agentic (64K input, 400 output) | Aggregated (2 × TP4) | vLLM | GB200 | 168 | 953.08 | 51.83 | 286.75 |
| Agentic (64K input, 400 output) | Disaggregated (1 prefill, 1 decode; TP4 each) | vLLM | GB200 | 168 | 1,154.87 | 80.85 | 169.02 |
| Agentic (64K input, 400 output) | Aggregated (4 × TP4) | vLLM | H200 | 80 | 209.18 | 51.32 | 171.29 |
| Agentic (64K input, 400 output) | Disaggregated (1 prefill, 1 decode; TP4 each) | vLLM | H200 | 64 | 359.41 | 50.63 | 177.59 |

### TTFT Distribution

Milliseconds across successful requests:

| Target | Mean | p50 | p75 | p90 | p95 | p99 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B200 aggregated | 1,408.61 | 178.66 | 775.00 | 3,068.28 | 5,552.72 | 26,864.62 | 59,884.76 |
| B200 disaggregated | 18,597.21 | 135.12 | 2,002.87 | 90,076.83 | 126,866.47 | 168,049.68 | 256,824.39 |
| GB200 aggregated | 1,601.63 | 286.75 | 951.37 | 3,524.82 | 6,980.67 | 26,793.16 | 52,808.90 |
| GB200 disaggregated | 12,149.79 | 169.02 | 1,126.26 | 57,116.54 | 100,309.52 | 124,167.34 | 160,272.44 |
| H200 aggregated | 2,314.58 | 171.29 | 1,365.00 | 6,368.76 | 11,694.66 | 30,075.76 | 108,304.39 |
| H200 disaggregated | 7,957.03 | 177.59 | 514.62 | 35,043.23 | 54,419.43 | 82,772.24 | 118,023.32 |

### ITL Distribution

Milliseconds, calculated from each request's average interval between tokens.

| Target | Mean | p50 | p75 | p90 | p95 | p99 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B200 aggregated | 25.06 | 18.28 | 25.07 | 41.42 | 76.17 | 120.94 | 218.89 |
| B200 disaggregated | 12.64 | 12.18 | 14.34 | 16.52 | 18.58 | 25.03 | 44.31 |
| GB200 aggregated | 26.40 | 19.29 | 27.22 | 48.76 | 83.17 | 110.46 | 321.77 |
| GB200 disaggregated | 13.19 | 12.37 | 14.61 | 17.75 | 21.14 | 30.59 | 81.71 |
| H200 aggregated | 29.33 | 19.49 | 29.90 | 54.01 | 84.55 | 183.57 | 381.82 |
| H200 disaggregated | 20.19 | 19.75 | 21.64 | 24.00 | 26.56 | 34.31 | 53.45 |

For GB200 at the selected concurrency, disaggregated serving delivers 21%
more output tok/s/GPU and 56% more p50 user tok/s. TTFT p90 rises from
3.52 s to 57.12 s.
