# AIPerf Invocation Reference

Authoritative source: NVIDIA AIPerf (`https://github.com/ai-dynamo/aiperf`).
The invocations below are stable for `aiperf >= 0.7.0`; verify the
version pinned by the Dynamo release the user is benchmarking against —
the `[mocker]` extra pulls a known-good AIPerf version per.

---

## Install

```bash
pip install ai-perf
```

Pin to the version the target release tested against. The 1.2.0 line
pulls AIPerf via the recipe `benchmark/run.sh` scripts; read those for
the exact pin.

---

## Chat Completions (most common)

```bash
aiperf benchmark \
  --model <model-id-from-/v1/models> \
  --endpoint-type chat \
  --url http://<frontend>:<port>/v1/chat/completions \
  --synthetic-input-tokens-mean <isl> \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean <osl> \
  --output-tokens-stddev 0 \
  --concurrency <n> \
  --measurement-interval <sec> \
  --output-format json \
  --artifact-dir ./aiperf-artifacts
```

Use `--synthetic-input-tokens-stddev` > 0 to introduce ISL variance
(simulates real workloads). Same for OSL.

For request-rate mode instead of concurrency:

```bash
aiperf benchmark ... --request-rate <req/s> --measurement-mode rate
```

---

## Completions (non-chat)

```bash
aiperf benchmark \
  --model <model-id> \
  --endpoint-type completions \
  --url http://<frontend>:<port>/v1/completions \
  ... (same other flags)
```

Use for base models (no chat template) or when the test specifically
needs the `/v1/completions` API.

---

## KV-Aware Routing

When the deployment includes a KV router, AIPerf can drive
conversation-style turns where each turn reuses the prior turn's KV
cache.

```bash
aiperf benchmark \
  --model <model-id> \
  --endpoint-type chat \
  --url http://<frontend>:<port>/v1/chat/completions \
  --multi-turn-mean 4 \
  --synthetic-input-tokens-mean 500 \
  --output-tokens-mean 200 \
  --concurrency 16 \
  --measurement-interval 600 \
  --output-format json \
  --artifact-dir ./aiperf-kv-aware
```

`--multi-turn-mean` configures the average number of turns per
conversation. With KV-aware routing enabled on the deployment, the
second-through-Nth turns should land on workers with warm KV cache.

Measure `cache_hit_rate` in the output to verify the routing works.

---

## Disaggregated Serving

Disagg deploys benefit from longer-duration measurements because the
prefill/decode split takes longer to reach steady state.

```bash
aiperf benchmark \
  --model <model-id> \
  --endpoint-type chat \
  --url http://<frontend>:<port>/v1/chat/completions \
  --synthetic-input-tokens-mean 3000 \
  --output-tokens-mean 500 \
  --concurrency 32 \
  --measurement-interval 1800 \
  --output-format json \
  --artifact-dir ./aiperf-disagg
```

Compare prefill TTFT and decode ITL separately — disagg's value is
that they can be tuned independently.

---

## Output Schema

AIPerf writes `./aiperf-artifacts/results.json` with:

```json
{
  "model": "...",
  "endpoint_type": "chat",
  "concurrency": 32,
  "request_rate": null,
  "duration_sec": 300,
  "request_count": 16860,
  "time_to_first_token_ms": {"p50": ..., "p90": ..., "p95": ..., "p99": ...},
  "inter_token_latency_ms": {"p50": ..., "p90": ..., "p95": ..., "p99": ...},
  "request_latency_ms": {"p50": ..., "p95": ..., "p99": ...},
  "throughput_req_per_sec": ...,
  "throughput_output_tokens_per_sec": ...,
  "cache_hit_rate": null,                       // populated when KV router is in the graph
  "errors": []
}
```

---

## Comparison to GenAI-Perf

AIPerf replaces NVIDIA GenAI-Perf for LLM inference benchmarking. The
flags overlap but are not identical. Key differences:

- AIPerf is Dynamo-aware (recognises the OpenAI-compatible Frontend
  shape directly without a Triton adapter).
- AIPerf reports `cache_hit_rate` natively when the deployment has a KV
  router.
- AIPerf supports `--multi-turn-mean` for conversational workloads.

Don't mix AIPerf and GenAI-Perf invocations in the same recipe
benchmark — pick one tool per release line. Per the Dynamo
recipes target AIPerf.
