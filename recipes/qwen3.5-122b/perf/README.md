# Benchmark — Qwen3.5-122B-A10B-FP8 tp1 + MTP, KV routing

Benchmarks the recipe with AIPerf against the deployed DGD: an agentic Mooncake
trace-replay, and the round_robin-vs-KV-router comparison. All commands assume the DGD
from [`../vllm/agg-h200/deploy.yaml`](../vllm/agg-h200/deploy.yaml) is deployed in
`${NAMESPACE}`; the frontend is `qwen35-122b-a10b-fp8-tp1mtp-agg-h200-frontend:8000` and
each worker exposes vLLM `/metrics` on `qwen35-122b-a10b-fp8-tp1mtp-agg-h200-agg:9090`.

## Workload

Agentic Mooncake trace transformed to the PRD target (median ISL ~64k, OSL ~400,
~90% token-weighted cache hit). Block size **512**, replayed **no-schedule** (timestamps
zeroed) at fixed **concurrency 8** (closed-loop). SLA = ≥ 50 output tok/s/user. Use a 15%
subset (~3,541 requests) for a quick comparison, or the full trace for PRD-scale numbers.

## 1. Stage the trace on the PVC

Copy the Mooncake JSONL (fields `input_length`, `output_length`, `hash_ids`; no `timestamp`
for no-schedule) onto the `model-cache` PVC via a helper pod that mounts it:

```bash
kubectl -n ${NAMESPACE} cp mooncake_trace.jsonl \
  ${NAMESPACE}/<pvc-helper-pod>:/model-cache/traces/mooncake_trace.jsonl
```

## 2. Run AIPerf (per router mode)

Run each router mode on a **cold** deployment for a fair comparison — either patch the
frontend `DYN_ROUTER_MODE` and restart pods between runs, or deploy two mode-fixed DGDs
(`round_robin` and `kv`). Dynamo ignores `cache_salt`, so reset by restarting the frontend
and worker pods between independent runs.

An AIPerf client pod (image `nvcr.io/nvidia/ai-dynamo/aiperf`, mounts the PVC) run against
the frontend service:

```bash
aiperf profile Qwen/Qwen3.5-122B-A10B-FP8 --tokenizer Qwen/Qwen3.5-122B-A10B-FP8 \
  --url http://${ENDPOINT} --endpoint-type chat \
  --input-file ${TRACE_FILE} \
  --custom-dataset-type mooncake_trace --prompt-input-tokens-block-size 512 \
  --concurrency ${CONCURRENCY} --workers-max ${CONCURRENCY} \
  --extra-inputs ignore_eos:true --streaming --use-server-token-count \
  --artifact-dir /model-cache/perf/<mode> --ui none
```

| Variable      | Default                                                    |
| ------------- | ---------------------------------------------------------- |
| `ENDPOINT`    | `qwen35-122b-a10b-fp8-tp1mtp-agg-h200-frontend:8000`       |
| `CONCURRENCY` | `8` (agentic SLA operating point)                          |
| `TRACE_FILE`  | `/model-cache/traces/mooncake_trace.jsonl`                 |

> In mooncake mode AIPerf replays the whole trace file (`--num-requests` is ignored); subset
> the file to cap request count. Do not compare partial runs — account for successful,
> errored, and unfinished requests before reporting aggregate throughput.

## 3. Metrics

- **AIPerf** → `/model-cache/perf/<mode>/profile_export_aiperf.{csv,json}`:
  `Output Token Throughput`, `Request Throughput`, `Time to First Token`,
  `Inter Token Latency`, `Output Token Throughput Per User`.
- **KV cache hit rate** from the frontend `/metrics`:
  `dynamo_component_router_kv_hit_rate_{sum,count}` (kv mode) and
  `dynamo_frontend_cached_tokens_{sum,count}`.

## Results

Agentic 15% Mooncake trace (3,541 reqs / 3,411 completed / 130 errors each), concurrency 8,
tp1 + MTP(nst=3) forced to the SpeedBench-measured AL=2.937 (`speculative-config-synthetic`).

| Router       | Output tok/s | Req/s | TTFT mean (ms) | ITL (ms) | KV hit rate |
| ------------ | ------------ | ----- | -------------- | -------- | ----------- |
| round_robin  | 620.7        | 0.27  | 11,194         | 10.4     | ~0 (routing off) |
| **kv**       | **759.5**    | 0.33  | **4,391**      | 9.2      | **59.0%**   |
| **Δ (kv)**   | **+22.4%**   | +22%  | **−61% (2.6×)** | −11%    | —           |

KV-aware routing is the recommended configuration: +22.4% output throughput and 2.6× lower
TTFT by landing shared-prefix requests on the replica holding the cache.

## Speculative decoding — measuring the acceptance length (SpeedBench)

Spec-decode throughput on synthetic traces is unrepresentative unless the acceptance length
(AL) is forced to the value **measured on SpeedBench** (real prompts). The recipe carries
this as the `speculative-config-synthetic` ConfigMap key; set the worker's
`SPECULATIVE_CONFIG` `configMapKeyRef.key` to it for benchmarking (never ship it).

### Measure AL

1. Serve with **real MTP** (ship the default `speculative-config` key), exposing `/metrics`.
2. Drive with SpeedBench qualitative (agentic-representative):
   ```bash
   aiperf profile Qwen/Qwen3.5-122B-A10B-FP8 --tokenizer Qwen/Qwen3.5-122B-A10B-FP8 \
     --url http://${ENDPOINT} --endpoint-type chat \
     --public-dataset speed-bench-qualitative \
     --concurrency 8 --workers-max 8 --request-count 150 --num-warmup-requests 5 \
     --streaming --use-server-token-count \
     --extra-inputs temperature:1.0 --extra-inputs max_tokens:4096 --ui none
   ```
   (No `ignore_eos` — natural EOS, or AL is skewed by junk tails.)
3. Read `AL = 1 + accepted/draft_steps` from the worker's vLLM spec-decode counters. Use a
   **freshly-started worker** (or the run right after a restart) so the cumulative counters
   cover only this SpeedBench run:
   ```bash
   curl -s http://<worker>:9090/metrics | grep -E 'vllm:spec_decode_num_(accepted_tokens_total|drafts_total)\{'
   # AL = 1 + spec_decode_num_accepted_tokens_total / spec_decode_num_drafts_total
   ```

### Measured (this recipe)

| nst | measured AL | SpeedBench-qualitative tok/s |
| --- | ----------- | ---------------------------- |
| 1   | 1.825       | 681 |
| **3 (recipe)** | **2.937** | 895 |
| 5   | 3.518       | 910 |

**Measured AL(nst=3)=2.937** (accepted 118,967 / drafts 61,422 → ~65% token acceptance).
This exact value is forced in the router benchmark (`synthetic_acceptance_length:2.937`) —
measured = forced; the live runtime confirmed mean acceptance length 2.94–2.95 during the
run. nst=5 edges nst=3 by <2% on the short-ISL SpeedBench split but loses on the 64k-context
(pool-bound) workload; nst=3 is the shipped depth.
