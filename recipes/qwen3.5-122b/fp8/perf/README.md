<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.5-122B-A10B-FP8 Benchmark Recipe

An [AIPerf](https://github.com/ai-dynamo/aiperf) trace-replay against a deployed DGD. The
benchmark is identical for both profiles; only `ENDPOINT` changes. The client waits for the
model on the frontend, runs a short warmup, replays the configured trace at one
`CONCURRENCY` value, and writes raw artifacts to the shared `model-cache` PVC.

## Targeting a variant

| Profile | `ENDPOINT` |
| --- | --- |
| Aggregated (tp1 + MTP) | `qwen35-122b-agg-h200-agentic-frontend:8000` |
| Disaggregated (1P2D)   | `qwen35-122b-disagg-h200-agentic-frontend:8000` |

Both serve `Qwen/Qwen3.5-122B-A10B`. Deploy from
[`../vllm/agg-h200-agentic/deploy.yaml`](../vllm/agg-h200-agentic/deploy.yaml) or
[`../vllm/disagg-h200-agentic/deploy.yaml`](../vllm/disagg-h200-agentic/deploy.yaml) (see
the recipe [README](../README.md)).

## Dataset

The benchmark replays a [Mooncake-format](https://github.com/kvcache-ai/Mooncake) trace via
`aiperf --custom-dataset-type mooncake_trace`. Each JSONL line describes one request
(`input_length`, `output_length`, `hash_ids`; no `timestamp` for no-schedule replay). The
agentic target is median ISL ~64k, OSL ~400, ~90% token-weighted cache hit, block size
**512**, replayed **no-schedule** (timestamps zeroed) at fixed **concurrency 8**
(closed-loop). The 15% subset is ~3,541 requests.

## Workflow

```bash
export NAMESPACE=your-namespace
```

### 1. Deploy the variant

See the deployment instructions in the recipe [README](../README.md).

### 2. Stage the trace on the PVC

Copy the Mooncake JSONL onto the `model-cache` PVC via a helper pod that mounts it:

```bash
kubectl -n ${NAMESPACE} cp mooncake_trace.jsonl \
  ${NAMESPACE}/<pvc-helper-pod>:/model-cache/traces/mooncake_trace.jsonl
```

### 3. Run AIPerf

An AIPerf client pod (image `nvcr.io/nvidia/ai-dynamo/aiperf`, mounts the PVC) run against
the frontend service:

```bash
aiperf profile Qwen/Qwen3.5-122B-A10B --tokenizer Qwen/Qwen3.5-122B-A10B-FP8 \
  --url http://${ENDPOINT} --endpoint-type chat \
  --input-file ${TRACE_FILE} \
  --custom-dataset-type mooncake_trace --prompt-input-tokens-block-size 512 \
  --concurrency ${CONCURRENCY} --workers-max ${CONCURRENCY} \
  --extra-inputs ignore_eos:true --streaming --use-server-token-count \
  --artifact-dir /model-cache/perf/<run> --ui none
```

Metrics land in `profile_export_aiperf.{csv,json}`: `Output Token Throughput`,
`Request Throughput`, `Time to First Token`, `Inter Token Latency`,
`Output Token Throughput Per User`. KV-cache hit rate is on the frontend `/metrics`
(`dynamo_component_router_kv_hit_rate_{sum,count}`).

> For representative **aggregated** numbers, force MTP to the SpeedBench-measured
> acceptance length via the `speculative-config-synthetic` ConfigMap key in the agg deploy
> (ship the real `speculative-config` key, benchmark with the synthetic one). The
> disaggregated profile runs without MTP.

## Running a concurrency sweep

Run one `CONCURRENCY` at a time; reset vLLM KV and Dynamo router state between independent
runs by restarting the DGD pods:

```bash
DGD=qwen35-122b-agg-h200-agentic # or qwen35-122b-disagg-h200-agentic
kubectl delete pods -n ${NAMESPACE} -l nvidia.com/dynamo-graph-deployment-name=${DGD}
kubectl wait --for=condition=Ready pod -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD} --timeout=7200s
```

In mooncake mode AIPerf replays the whole trace file (`--num-requests` is ignored); subset
the file to cap request count. Do not compare partial runs — account for successful,
errored, and unfinished requests before reporting aggregate throughput.

## Tunable environment variables

| Variable | Default | Notes |
| --- | --- | --- |
| `ENDPOINT` | per profile — see the table above | |
| `CONCURRENCY` | `8` (agentic SLA operating point) | Single value; reset server state between values |
| `TRACE_FILE` | `/model-cache/traces/mooncake_trace.jsonl` | 15% agentic trace = 3,541 requests |
| `TARGET_MODEL` | `Qwen/Qwen3.5-122B-A10B` | Must match `--served-model-name` |

## Artifacts

Results are written to `/model-cache/perf/<run>/profile_export_aiperf.{csv,json}`,
`inputs.json`, and warmup/log files.
