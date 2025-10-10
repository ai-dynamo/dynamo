# vLLM Prometheus Metrics (Actual)

This document lists all Prometheus metrics collected from a running vLLM engine.

**Source:** Captured from running vLLM v0.10.2 engine (Qwen/Qwen3-0.6B)
**Date:** 2025-10-09
**vLLM Reference:** [vLLM v0.10.2 Metrics Documentation](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/engine/metrics.py)

<!--
AI MAINTENANCE GUIDE:
To update this document with the latest vLLM metrics:

1. Start a vLLM worker with metrics enabled:
   ```bash
   DYN_LOG=error DYN_ENGINE_METRICS_ENABLED=1 DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 \
     python -m dynamo.vllm --model Qwen/Qwen3-0.6B &
   VLLM_PID=$!
   sleep 15  # Wait for engine to initialize
   ```

2. Capture HELP and TYPE metadata (authoritative source):
   ```bash
   curl -s localhost:8081/metrics | grep -E "^# (TYPE|HELP) vllm:" | sort
   ```

3. Use the HELP text exactly as the metric descriptions in this document
   - HELP text is the authoritative source for metric descriptions
   - TYPE text shows the metric type (counter, gauge, histogram, info)
   - Keep deprecated warnings verbatim (e.g., "DEPRECATED: Use vllm:xyz instead")

4. Cleanup:
   ```bash
   kill $VLLM_PID
   ```

Note: Each histogram metric produces multiple time series (_bucket, _count, _sum)
but in vLLM's HELP text, they document only the base metric name.

The metrics are exposed via the `register_engine_metrics_callback()` function which uses
`get_prometheus_expfmt()` to fetch and filter metrics by prefix.
-->

## vLLM Engine Metrics

This document shows only vLLM engine-specific metrics (prefixed with `vllm:`).
Python runtime and process metrics (e.g., `python_gc_*`, `process_*`) are not included.

### Base Metrics (without suffixes)

### Gauges
- `vllm:num_requests_running` - Number of requests currently running on GPU
- `vllm:num_requests_waiting` - Number of requests waiting to be processed
- `vllm:kv_cache_usage_perc` - KV-cache usage. 1 means 100 percent usage
- `vllm:gpu_cache_usage_perc` - GPU KV-cache usage. 1 means 100 percent usage. DEPRECATED: Use vllm:kv_cache_usage_perc instead

### Counters
Note: All counters include both `_total` and `_created` suffixes

- `vllm:num_preemptions_total` / `_created` - Cumulative number of preemptions from the engine
- `vllm:prompt_tokens_total` / `_created` - Number of prefill tokens processed
- `vllm:generation_tokens_total` / `_created` - Number of generation tokens processed
- `vllm:request_success_total` / `_created` - Count of successfully processed requests (by finished_reason)
- `vllm:prefix_cache_queries_total` / `_created` - Prefix cache queries, in terms of number of queried tokens
- `vllm:prefix_cache_hits_total` / `_created` - Prefix cache hits, in terms of number of cached tokens
- `vllm:gpu_prefix_cache_queries_total` / `_created` - GPU prefix cache queries, in terms of number of queriedtokens. DEPRECATED: Use vllm:prefix_cache_queries instead
- `vllm:gpu_prefix_cache_hits_total` / `_created` - GPU prefix cache hits, in terms of number of cached tokens. DEPRECATED: Use vllm:prefix_cache_hits instead

### Histograms
Note: Each histogram produces `_bucket`, `_count`, `_sum`, and `_created` time series

- `vllm:time_to_first_token_seconds` - Histogram of time to first token in seconds
- `vllm:time_per_output_token_seconds` - Histogram of time per output token in seconds. DEPRECATED: Use vllm:inter_token_latency_seconds instead
- `vllm:inter_token_latency_seconds` - Histogram of inter-token latency in seconds
- `vllm:e2e_request_latency_seconds` - Histogram of end-to-end request latency in seconds
- `vllm:request_queue_time_seconds` - Histogram of time spent in WAITING phase for request
- `vllm:request_inference_time_seconds` - Histogram of time spent in RUNNING phase for request
- `vllm:request_prefill_time_seconds` - Histogram of time spent in PREFILL phase for request
- `vllm:request_decode_time_seconds` - Histogram of time spent in DECODE phase for request
- `vllm:iteration_tokens_total` - Histogram of number of tokens per engine_step
- `vllm:request_max_num_generation_tokens` - Histogram of maximum number of requested generation tokens
- `vllm:request_params_n` - Histogram of the n request parameter
- `vllm:request_params_max_tokens` - Histogram of the max_tokens request parameter
- `vllm:request_prompt_tokens` - Number of prefill tokens processed
- `vllm:request_generation_tokens` - Number of generation tokens processed

### Info
- `vllm:cache_config_info` - Information of the LLMEngine CacheConfig

## Notes

- These are vLLM engine-specific metrics only (Python/Process metrics not shown)
- Metrics collected from the default Prometheus REGISTRY
- vLLM v1 uses single-process mode (not multiprocess like SGLang)
- All metrics are prefixed with `vllm:`
- Histogram metrics include bucket, count, sum, and created timestamps
- Metrics appear after engine initialization completes
