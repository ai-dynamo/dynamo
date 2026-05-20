# Debug: TRT-LLM publish_events_and_metrics Stall

**Date**: 2026-05-19
**Source**: user report
**Status**: investigating
**Environment**: GPU unknown (`nvidia-smi` not available), Linux 6.17.0-22-generic, Python 3.12.3

## Problem
TRT-LLM worker occasionally stalls after enabling `--publish-events-and-metrics` during a KV-event and metrics intensive benchmark.

## Investigation Log

### 2026-05-19
- `--publish-events-and-metrics` enables TRT-LLM `return_perf_metrics`, `enable_iter_perf_stats`, and `kv_cache_config.event_buffer_max_size=1024`.
- The worker starts two publisher threads: stats and KV events. In this checkout each `ManagedThread` owns a private asyncio event loop, so publisher polling should no longer run directly on the request-handler uvloop.
- The likely hot-path bottleneck is the TRT-LLM publisher's KV event drain path:
  - `Publisher._publish_kv_cache_events_task()` polls `engine.llm.get_kv_cache_events_async(timeout=0.0)`.
  - `Publisher._handle_kv_event()` performs per-event Python traversal, token list construction, debug logging, and then ZMQ or Rust/NATS publication.
  - `publisher.py` calls `logging.basicConfig(level=logging.DEBUG)` and logs full event payloads/token lists on the hot path. In a KV-heavy workload this can dominate the publisher thread with GIL-held string formatting and log I/O.
- The metrics path is less suspicious:
  - `WorkerMetricsPublisher.publish()` writes to a Rust `watch` channel.
  - `FpmDirectPublisher.publish()` writes to Rust unbounded channels.
  - These can add overhead, but they are not the obvious blocking point.
- With KVBM/consolidator enabled, the path adds ZMQ serialization/deserialization and per-block tracker work:
  - Python publishes one message per engine event.
  - The consolidator splits stored events back into per-block chunks, computes hashes, updates hash maps, and republishes every 50 ms.

## Root Cause Hypothesis
The engine-side KV event buffer can fill when Python-side event handling cannot drain fast enough. Under this benchmark, the primary suspect is hot-path DEBUG logging of full KV event payloads and token lists, followed by Python per-event serialization/traversal. Once the engine buffer fills, TRT-LLM event emission can backpressure generation, surfacing as a worker/uvloop stall.

## Next Checks
- Run with Python logging level INFO/WARNING and compare stall frequency.
- Inspect logs for high-volume `KV cache event received` / `publish stored event` lines.
- Track event-buffer drops/gaps via `Non-consecutive engine event_id` warnings.
- If still stalling, profile the publisher thread around `_handle_kv_event`, `msgpack.packb`, and the consolidator tracker.

### 2026-05-19 Update
- DEBUG logging was deprioritized based on user repro evidence: switching to default Dynamo logging did not resolve the stall.
- Stronger root-cause candidate found in legacy `HandlerBase._generate_locally_impl`: `publisher.start()` was called only after the first `generation_result` item was yielded.
- In prefill-heavy/KV-event-heavy traffic, TRT-LLM can produce KV events before the first streamed result. With `event_buffer_max_size=1024`, that creates a circular wait: engine fills finite KV event buffer before first result; publisher has not started draining; worker stalls.
- Patch moved `publisher.start()` to immediately after `generate_async()` returns, before consuming `generation_result`.
