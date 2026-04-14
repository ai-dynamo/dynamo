# KV Router Queue Threshold Causes 4x Throughput Regression on Large Multi-Turn Requests

**Linear ticket**: [DIS-1722](https://linear.app/nvidia/issue/DIS-1722)

## Summary

`DYN_ROUTER_QUEUE_THRESHOLD` (default `4.0`) causes the KV router's scheduler queue to park requests indefinitely when per-request prefill token counts are large (8500+ tokens). This produces a 4x throughput regression compared to round-robin routing — from 31.79 req/s down to ~7.4 req/s.

Setting `DYN_ROUTER_QUEUE_THRESHOLD=999` (effectively disabling the queue) restores throughput to 30.75 req/s.

## Root Cause

The scheduler queue in `lib/kv-router/src/scheduling/queue.rs` gates request dispatch with `all_workers_busy()`:

```rust
// queue.rs:387
if (tokens as f64) <= threshold * (max_batched as f64) {
    return false; // worker has capacity
}
```

A worker is considered "busy" when its frontend-tracked `active_prefill_tokens` exceeds `threshold × max_num_batched_tokens`. With default values:

- `threshold = 4.0` (from `DYN_ROUTER_QUEUE_THRESHOLD`)
- `max_num_batched_tokens = 8192` (vLLM default)
- **Busy when**: `active_prefill_tokens > 32,768` per worker

For large multi-turn requests (8500 tokens each), only ~3.8 requests per worker trigger the threshold. At 32 QPS across 8 workers (4 QPS/worker), the threshold is exceeded within the first second. Once all workers are marked "busy," every subsequent request is parked in the pending `BinaryHeap` and never dispatched — because `update()` (which drains the queue) only runs when `mark_prefill_completed` fires, but queued requests never reach a worker to begin prefilling.

This creates a deadlock-like starvation: requests can't be dispatched because workers appear busy, and workers can't become un-busy because no new requests are being dispatched to complete and free capacity.

## Reproduce

### Environment

- **Hardware**: 8x B200 GPUs (tested on computelab `umbriel-b200-074`)
- **Model**: `Qwen/Qwen3-VL-30B-A3B-Instruct` (MoE, multimodal), TP=1, 8 workers
- **Workload**: Multi-turn multimodal conversations (pinassistant: 5500 system tokens + 3000 user tokens + 3 images in turn 1)
- **Container**: `nvcr.io/nvstaging/ai-dynamo/vllm-runtime:qiwa-vllm-x86-04-07`
- **Benchmark**: aiperf with `raw_payload` dataset type, 32 QPS constant rate, 120s duration

### Steps

1. Launch Dynamo with KV router (1 frontend + 8 workers):

```bash
# Frontend
DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD=1.0 \
  python -m dynamo.frontend --router-mode kv &

# Workers (one per GPU)
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i \
  python3 -m dynamo.vllm \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --enable-multimodal \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:$((20080+i))\",\"enable_kv_cache_events\":true}" &
done
```

2. Send multi-turn traffic at 32 QPS. Any workload with >4000 prefill tokens per request will trigger the issue.

### Expected vs Actual

| Config | TTFT avg | Req/s | Completed |
|--------|---------|-------|-----------|
| Dynamo round-robin (1 FE, 8 workers) | 459ms | 31.79 | 3,839/3,839 |
| **Dynamo KV router (threshold=4.0, default)** | **62,824ms** | **~7.4** | **636/3,839** |
| Dynamo KV router (threshold=999) | 500ms | 30.75 | 3,787/3,839 |

### Diagnostic: span timing confirms queue starvation

Enable span timing on the frontend:

```bash
DYN_LOGGING_JSONL=1 DYN_LOGGING_SPAN_EVENTS=1 \
DYN_LOG=info,dynamo_llm::kv_router=debug,kv_router=debug
```

The frontend log (`SPAN_CLOSED` events) shows:

**Early requests (before queue fills):**
```
kv_router.schedule:       duration_us=115, busy_us=7,   idle_us=108
kv_router.select_worker:  duration_us=314, busy_us=189, idle_us=125
```

**Late requests (after queue fills):**
```
kv_router.schedule:       duration_us=312000007, busy_us=7,   idle_us=312000000
kv_router.select_worker:  duration_us=312000225, busy_us=225, idle_us=312000000
```

The `busy_us` (actual CPU work) is **microseconds** in both cases. The 312-second `idle_us` on late requests proves they sat in the pending queue for the entire benchmark duration. The KV router's compute path is not slow — the queue admission check (`all_workers_busy`) is incorrectly blocking dispatch.

## Workaround

Set `DYN_ROUTER_QUEUE_THRESHOLD=999` on the frontend to effectively disable the scheduler queue.

## Analysis

The threshold is expressed as a fraction of `max_num_batched_tokens`, which is a vLLM per-batch tuning knob — not a measure of worker capacity. This makes the threshold sensitive to prompt size in a way that doesn't scale:

| Prompt size | Requests to fill one worker (threshold=4.0, max_batched=8192) |
|-------------|---------------------------------------------------------------|
| 500 tokens  | ~65 requests |
| 2000 tokens | ~16 requests |
| 8500 tokens | ~3.8 requests |

For workloads with large prompts (multimodal, multi-turn, long-context), the default threshold is too aggressive and causes starvation.

Additionally, the queue creates a deadlock-like feedback loop:
1. Workers marked "busy" from frontend token accounting
2. New requests parked in pending queue
3. Parked requests never reach workers → `mark_prefill_completed` never fires
4. Workers stay "busy" indefinitely → queue never drains

The `update()` drain function (`queue.rs:196`) is called on `mark_prefill_completed` and on a recheck interval, but when no requests are being dispatched, no prefills complete, so the drain never triggers meaningfully.

### Code references

- Queue admission: `lib/kv-router/src/scheduling/queue.rs:148-170` (`enqueue()`)
- Busy check: `lib/kv-router/src/scheduling/queue.rs:348-392` (`all_workers_busy()`)
- Threshold config: `lib/kv-router/src/scheduling/config.rs:192` (default `Some(4.0)`)
- CLI/env var: `components/src/dynamo/common/configuration/groups/kv_router_args.py:255` (`DYN_ROUTER_QUEUE_THRESHOLD`)
- Active token tracking: `lib/kv-router/src/sequences/single.rs:197-199` (`active_tokens()` = `active_prefill_tokens_at()`)
- Drain trigger: `lib/kv-router/src/scheduling/local.rs:229-234` (`mark_prefill_completed()` → `queue.update()`)
