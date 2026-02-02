# Prometheus Metrics Comparison: vLLM vs SGLang vs TensorRT-LLM

This document compares the Prometheus metrics exposed by the three inference backends supported by Dynamo.

## Overview

| Framework | Metric Prefix | Total Unique Metrics |
|-----------|---------------|---------------------|
| vLLM | `vllm:` | ~30 |
| SGLang | `sglang:` | ~40+ |
| TensorRT-LLM | `trtllm_` | ~5 |

All frameworks share the common `dynamo_component_*` metrics from the Dynamo runtime.

---

## Common Dynamo Runtime Metrics

These metrics are available across all backends:

| Metric Name | Type | Description |
|-------------|------|-------------|
| `dynamo_component_inflight_requests` | gauge | Number of requests currently being processed |
| `dynamo_component_kvstats_active_blocks` | gauge | Number of active KV cache blocks in use |
| `dynamo_component_kvstats_gpu_cache_usage_percent` | gauge | GPU cache usage as percentage (0.0-1.0) |
| `dynamo_component_kvstats_gpu_prefix_cache_hit_rate` | gauge | GPU prefix cache hit rate (0.0-1.0) |
| `dynamo_component_kvstats_total_blocks` | gauge | Total number of KV cache blocks available |
| `dynamo_component_request_bytes_total` | counter | Total bytes received in requests |
| `dynamo_component_request_duration_seconds` | histogram | Time spent processing requests |
| `dynamo_component_requests_total` | counter | Total number of requests processed |
| `dynamo_component_response_bytes_total` | counter | Total bytes sent in responses |
| `dynamo_component_uptime_seconds` | gauge | Total uptime of the DistributedRuntime |

---

## Framework-Specific Metrics Comparison

| Category | vLLM | SGLang | TensorRT-LLM |
|----------|------|--------|--------------|
| **REQUEST STATE & QUEUE** | | | |
| Running requests | `num_requests_running` | `num_running_reqs` | - |
| Waiting/queued requests | `num_requests_waiting` | `num_queue_reqs` | - |
| Queue time | `request_queue_time_seconds` | `queue_time_seconds` | `request_queue_time_seconds` |
| Grammar queue | - | `num_grammar_queue_reqs` | - |
| Offline batch running | - | `num_running_reqs_offline_batch` | - |
| Prefill prealloc queue | - | `num_prefill_prealloc_queue_reqs` | - |
| Prefill inflight queue | - | `num_prefill_inflight_queue_reqs` | - |
| Decode prealloc queue | - | `num_decode_prealloc_queue_reqs` | - |
| Decode transfer queue | - | `num_decode_transfer_queue_reqs` | - |
| **LATENCY** | | | |
| Time to first token | `time_to_first_token_seconds` | `time_to_first_token_seconds` | `time_to_first_token_seconds` |
| Inter-token latency | `inter_token_latency_seconds` | `inter_token_latency_seconds` | - |
| E2E request latency | `e2e_request_latency_seconds` | `e2e_request_latency_seconds` | `e2e_request_latency_seconds` |
| Time per output token | `request_time_per_output_token_seconds` | - | `time_per_output_token_seconds` |
| Inference time | `request_inference_time_seconds` | - | - |
| Prefill time | `request_prefill_time_seconds` | - | - |
| Decode time | `request_decode_time_seconds` | - | - |
| Per-stage latency | - | `per_stage_req_latency_seconds` | - |
| **TOKEN METRICS** | | | |
| Prompt/prefill tokens | `prompt_tokens_total` | `prompt_tokens_total` | - |
| Generation tokens | `generation_tokens_total` | `generation_tokens_total` | - |
| Request prompt tokens (histogram) | `request_prompt_tokens` | - | - |
| Request generation tokens (histogram) | `request_generation_tokens` | - | - |
| Iteration tokens | `iteration_tokens_total` | - | - |
| Max generation tokens | `request_max_num_generation_tokens` | - | - |
| Realtime tokens | - | `realtime_tokens_total` | - |
| Used tokens | - | `num_used_tokens` | - |
| Prefill KV computed tokens | `request_prefill_kv_computed_tokens` | - | - |
| **REQUEST SUCCESS** | | | |
| Request success (by reason) | `request_success_total` | - | `request_success_total` |
| Total requests | - | `num_requests_total` | - |
| **KV CACHE & MEMORY** | | | |
| KV cache usage % | `kv_cache_usage_perc` | - | - |
| Token usage | - | `token_usage` | - |
| Max total tokens | - | `max_total_num_tokens` | - |
| SWA token usage | - | `swa_token_usage` | - |
| Mamba usage | - | `mamba_usage` | - |
| Pending prealloc token usage | - | `pending_prealloc_token_usage` | - |
| **PREFIX CACHE** | | | |
| Cache hit rate | - | `cache_hit_rate` | - |
| Prefix cache queries | `prefix_cache_queries_total` | - | - |
| Prefix cache hits | `prefix_cache_hits_total` | - | - |
| External prefix cache queries | `external_prefix_cache_queries_total` | - | - |
| External prefix cache hits | `external_prefix_cache_hits_total` | - | - |
| **MULTI-MODAL CACHE** | | | |
| MM cache queries | `mm_cache_queries_total` | - | - |
| MM cache hits | `mm_cache_hits_total` | - | - |
| **ENGINE STATE** | | | |
| Engine sleep state | `engine_sleep_state` | - | - |
| Engine startup time | - | `engine_startup_time` | - |
| Engine load weights time | - | `engine_load_weights_time` | - |
| Cache config info | `cache_config_info` | - | - |
| CUDA graph state | - | `is_cuda_graph` | - |
| CUDA graph passes | - | `cuda_graph_passes_total` | - |
| Utilization | - | `utilization` | - |
| New token ratio | - | `new_token_ratio` | - |
| **PREEMPTION & RETRACTION** | | | |
| Preemptions | `num_preemptions_total` | - | - |
| Retracted requests | - | `num_retracted_reqs` | - |
| Number of retractions | - | `num_retractions` | - |
| Paused requests | - | `num_paused_reqs` | - |
| **REQUEST PARAMETERS** | | | |
| Request param n | `request_params_n` | - | - |
| Request param max_tokens | `request_params_max_tokens` | - | - |
| **THROUGHPUT & PERFORMANCE** | | | |
| Generation throughput | - | `gen_throughput` | - |
| Decode sum sequence lens | - | `decode_sum_seq_lens` | - |
| **SPECULATIVE DECODING** | | | |
| Spec accept length | - | `spec_accept_length` | - |
| Spec accept rate | - | `spec_accept_rate` | - |
| **KV TRANSFER** | | | |
| KV transfer speed (GB/s) | - | `kv_transfer_speed_gb_s` | - |
| KV transfer latency (ms) | - | `kv_transfer_latency_ms` | - |
| KV transfer bootstrap (ms) | - | `kv_transfer_bootstrap_ms` | - |
| KV transfer alloc (ms) | - | `kv_transfer_alloc_ms` | - |
| KV transfer total (MB) | - | `kv_transfer_total_mb` | - |

**Note:** Metric names shown without prefix. Actual metrics use `vllm:`, `sglang:`, or `trtllm_` prefix.

---

## Summary

| Feature | vLLM | SGLang | TensorRT-LLM |
|---------|------|--------|--------------|
| **Latency Metrics** | Comprehensive (TTFT, ITL, E2E, prefill, decode, inference) | Comprehensive (TTFT, ITL, E2E, per-stage) | Basic (TTFT, E2E, TPOT, queue) |
| **Token Metrics** | Detailed histograms | Basic counters + realtime | None |
| **Cache Metrics** | Extensive (prefix, external, multi-modal) | Basic cache hit rate | None |
| **Queue Metrics** | Basic (running, waiting) | Detailed (multiple queue types) | Basic (queue time only) |
| **Speculative Decoding** | None | Yes (accept length/rate) | None |
| **KV Transfer** | None | Yes (speed, latency, bootstrap) | None |
| **Engine State** | Sleep state, config info | Startup time, CUDA graph | None |

**Key Takeaways:**
- **vLLM** has comprehensive observability with detailed latency breakdowns and cache metrics
- **SGLang** provides good coverage with unique metrics for speculative decoding and KV transfer
- **TensorRT-LLM** has minimal metrics, relying mainly on common Dynamo metrics
