# SGLang Prometheus Metrics (Actual)

This document lists all Prometheus metrics collected from a running SGLang engine.

**Source:** Captured from running SGLang engine (Qwen/Qwen3-0.6B)
**Date:** 2025-10-09
**SGLang Reference:** [SGLang Metrics Documentation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/metrics/collector.py)

<!--
AI MAINTENANCE GUIDE:
To update this document with the latest SGLang metrics:

1. Start an SGLang worker with metrics enabled:
   ```bash
   DYN_LOG=error DYN_ENGINE_METRICS_ENABLED=1 DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 \
     python -m dynamo.sglang --model Qwen/Qwen3-0.6B &
   SGLANG_PID=$!
   sleep 15  # Wait for engine to initialize
   ```

2. Capture HELP and TYPE metadata (authoritative source):
   ```bash
   curl -s localhost:8081/metrics | grep -E "^# (TYPE|HELP) sglang:" | sort
   ```

3. Use the HELP text exactly as the metric descriptions in this document
   - HELP text is the authoritative source for metric descriptions
   - TYPE text shows the metric type (counter, gauge, histogram, summary)
   - Keep all metric descriptions verbatim

4. Cleanup:
   ```bash
   kill $SGLANG_PID
   ```

Note: SGLang uses multiprocess metrics collection via prometheus_client.multiprocess.MultiProcessCollector

The metrics are exposed via the `register_engine_metrics_callback()` function which uses
`get_prometheus_expfmt()` to fetch and filter metrics by prefix.
-->

## SGLang Engine Metrics

This document shows only SGLang engine-specific metrics (prefixed with `sglang:`).
Python runtime and process metrics (e.g., `python_gc_*`, `process_*`) are not included.

### Scheduler Metrics (Gauges)

#### Request Counters
- `sglang:num_retracted_reqs` - The number of retracted requests
- `sglang:num_paused_reqs` - The number of paused requests by async weight sync
- `sglang:num_running_reqs` - The number of running requests
- `sglang:num_queue_reqs` - The number of requests in the waiting queue
- `sglang:num_grammar_queue_reqs` - The number of requests in the grammar waiting queue
- `sglang:num_running_reqs_offline_batch` - The number of running low-priority offline batch requests(label is 'batch')

#### Token Metrics
- `sglang:num_used_tokens` - The number of used tokens
- `sglang:token_usage` - The token usage
- `sglang:swa_token_usage` - The token usage for SWA layers

#### Performance Metrics
- `sglang:gen_throughput` - The generation throughput (token/s)
- `sglang:cache_hit_rate` - The prefix cache hit rate
- `sglang:avg_request_queue_latency` - The average request queue latency for the last batch of requests in seconds
- `sglang:spec_accept_length` - The average acceptance length of speculative decoding

#### Disaggregation Queue Metrics
- `sglang:num_prefill_prealloc_queue_reqs` - The number of requests in the prefill prealloc queue
- `sglang:num_prefill_inflight_queue_reqs` - The number of requests in the prefill inflight queue
- `sglang:num_decode_prealloc_queue_reqs` - The number of requests in the decode prealloc queue
- `sglang:num_decode_transfer_queue_reqs` - The number of requests in the decode transfer queue

#### KV Transfer Metrics
- `sglang:kv_transfer_speed_gb_s` - The transfer speed of the KV cache in GB/s
- `sglang:kv_transfer_latency_ms` - The transfer latency of the KV cache in ms

#### Resource Usage
- `sglang:total_retracted_reqs` - The total number of retracted requests due to kvcache full
- `sglang:utilization` - The utilization

#### Engine Initialization Metrics
- `sglang:engine_startup_time` - The time taken for the engine to start up
- `sglang:engine_load_weights_time` - The time taken for the engine to load weights

### Counter Metrics
- `sglang:num_bootstrap_failed_reqs_total` - The number of bootstrap failed requests
- `sglang:num_bootstrap_failed_reqs_created` - The number of bootstrap failed requests (created timestamp)
- `sglang:num_transfer_failed_reqs_total` - The number of transfer failed requests
- `sglang:num_transfer_failed_reqs_created` - The number of transfer failed requests (created timestamp)
- `sglang:prompt_tokens_total` - Number of prefill tokens processed
- `sglang:prompt_tokens_created` - Number of prefill tokens processed (created timestamp)
- `sglang:generation_tokens_total` - Number of generation tokens processed
- `sglang:generation_tokens_created` - Number of generation tokens processed (created timestamp)
- `sglang:cached_tokens_total` - Number of cached prompt tokens
- `sglang:cached_tokens_created` - Number of cached prompt tokens (created timestamp)
- `sglang:num_requests_total` - Number of requests processed
- `sglang:num_requests_created` - Number of requests processed (created timestamp)
- `sglang:num_so_requests_total` - Number of structured output requests processed
- `sglang:num_so_requests_created` - Number of structured output requests processed (created timestamp)
- `sglang:num_aborted_requests_total` - Number of requests aborted
- `sglang:num_aborted_requests_created` - Number of requests aborted (created timestamp)

### Histogram Metrics

Each histogram produces: `_bucket`, `_count`, `_sum`, `_created` suffixes

#### Latency Histograms
- `sglang:time_to_first_token_seconds` - Histogram of time to first token in seconds
- `sglang:inter_token_latency_seconds` - Histogram of inter-token latency in seconds
- `sglang:e2e_request_latency_seconds` - Histogram of End-to-end request latency in seconds

#### Token Distribution Histograms (when enabled)
- `sglang:prompt_tokens_histogram` - Histogram of prompt token length
- `sglang:generation_tokens_histogram` - Histogram of generation token length

### Function Latency Metrics (when enabled)
- `sglang:func_latency_seconds` - Function latency in seconds (with label: name)

## Metric Labels

SGLang metrics include the following labels for filtering and aggregation:

- `model_name` - The name of the model being served (e.g., "Qwen/Qwen3-0.6B")
- `engine_type` - The engine type (e.g., "unified", "prefill", "decode")
- `tp_rank` - Tensor parallel rank
- `pp_rank` - Pipeline parallel rank
- `pid` - Process ID (for multiprocess metrics)

## Summary

**Key Differences from vLLM:**
- SGLang uses multiprocess metrics collection (`CollectorRegistry` + `MultiProcessCollector`)
- All metrics are prefixed with `sglang:`
- Includes disaggregation-specific metrics (prefill/decode queues, KV transfer)
- Includes structured output request metrics
- More granular queue metrics for different stages

**Total metric categories:**
- Scheduler gauges: ~20 metrics
- Counters: ~14 base metrics (each with _total and _created)
- Histograms: 3-5 base metrics (each with buckets, count, sum, created)
- Function latency: When enabled
- Python/Process metrics: 16 (standard)

## Notes

- Metrics collected from `CollectorRegistry` with `multiprocess.MultiProcessCollector`
- SGLang supports multiprocess mode for better scalability
- All metrics are prefixed with `sglang:`
- Histogram metrics include bucket, count, sum, and created timestamps
- Metrics appear after engine initialization completes
- Function latency metrics require `enable_func_timer()` to be called
- Token histograms are optional and controlled by `collect_tokens_histogram` parameter

