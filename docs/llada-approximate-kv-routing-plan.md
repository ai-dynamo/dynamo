# LLaDA 2.0 — Approximate KV Routing Implementation & Benchmark Plan

Companion to `docs/llada-dynamo-routing-analysis.md`. Concrete plan for the only Dynamo-only recommendation that touches the routing path: approximate KV routing with `--page-size 32` aligned to LLaDA's semantic block size.

## Hypothesis

Approximate KV routing (`--router-mode kv --no-router-kv-events`) co-locates requests sharing a token prefix on the same LLaDA worker. Even though SGLang's `ChunkCache` frees KV at request completion, co-location yields a small latency improvement via process-level locality (warm tokenizer, JIT-compiled kernels, kernel-cache resident, HTTP keep-alive). The expected magnitude is 1-3% TTFB reduction on prefix-heavy workloads, 0% on uncorrelated, and zero regression in either case.

## What we are NOT doing

- Not modifying SGLang. No re-enabling radix cache, no scheduler hooks, no `process_batch_result_dllm` changes.
- Not implementing prompt-KV capture-replay or KVBM integration. Those live in the analysis doc's out-of-scope section.
- Not modifying the existing `diffusion_llada.sh`. A new script variant goes in for the two-worker setup; the single-worker script stays untouched as a clean reference.

## Architecture (target)

```
┌─────────────────────────────────────────────────────────────┐
│                  Dynamo Frontend  :8001                     │
│  --router-mode {round-robin | kv}                           │
│  --router-kv-events / --no-router-kv-events                 │
│  --kv-cache-block-size 32                                   │
└────────────────────┬────────────────────────────────────────┘
                     │ dyn://dynamo.backend.generate (etcd discovery)
        ┌────────────┴────────────┐
        ▼                         ▼
┌────────────────────┐  ┌────────────────────┐
│ Worker 0 (GPU 0)   │  │ Worker 1 (GPU 1)   │
│ dynamo.sglang      │  │ dynamo.sglang      │
│ --page-size 32     │  │ --page-size 32     │
│ --dllm-algorithm   │  │ --dllm-algorithm   │
│   LowConfidence    │  │   LowConfidence    │
└────────────────────┘  └────────────────────┘

Infra:  etcd:2379    nats:4222   (already up via deploy/docker-compose.yml)
```

## Files added/edited

| Path | Purpose |
|---|---|
| `examples/backends/sglang/launch/diffusion_llada_multi.sh` | New. Worker-only launcher with `--gpu-id` and `--page-size 32`. No frontend. |
| `examples/backends/sglang/launch/frontend_router.sh` | New. Launches just the Dynamo frontend with a `--router-mode` arg. |
| `bench/llada-approx-kv/run_sweep.sh` | New. Drives the 3-mode benchmark sweep against aiperf. |
| `bench/llada-approx-kv/analyse.py` | New. Parses aiperf JSON outputs and produces the comparison table. |
| `aa-working-notes/2026-05-11-llada-approximate-kv-routing.md` | Worklog. |

No existing files are edited.

## Configuration matrix

| Mode | `--router-mode` | KV events flag | Notes |
|---|---|---|---|
| **A. Round-robin** | `round-robin` | n/a | Stateless baseline. |
| **B. Approximate KV** | `kv` | `--no-router-kv-events` | Hypothesis. Indexer self-records. |
| **C. Event-driven KV** | `kv` | `--router-kv-events` (default) | Negative control. Empty indexer → degenerates to load-only. |

Common across all: `--kv-cache-block-size 32` (matches `--page-size 32` on workers).

## Workload

Prefix-heavy chat workload via `aiperf profile`:

- **Model**: `inclusionAI/LLaDA2.0-mini-preview`
- **Prefix pool**: 4 distinct system prompts × 2000 tokens each (`--prefix-prompt-pool-size 4 --prefix-prompt-length 2000`).
- **User turn**: 64-token random input (`--isl 64`).
- **Output**: 64 tokens (`--osl 64`).
- **Concurrency**: 8 (with 2 workers, this puts each worker around its `max_running_requests=8` ceiling).
- **Request count**: 200 + 20 warmup.
- **Streaming**: enabled (TTFT measurement).
- **Random seed**: 42 (deterministic across modes).

This creates 4 "user groups" sharing a system prompt. A router that pins each group to a worker (2 groups per worker) sees half the requests warm-locality-wise. With `prompt-encode ≈ 6%` of total compute at these dimensions, and approximate routing only saving the process-locality portion of that, expected magnitude is the single-digit ceiling.

## Metrics

Primary:
- **TTFT** (= time-to-first-block for LLaDA): proxy for prompt-encode cost.
- **Request latency p50, p95, p99**.
- **Output-token throughput** (tokens/sec aggregated).

Secondary:
- Per-worker request count (verify routing actually moved requests).
- Per-worker peak `decode_blocks` (verify load balance preserved in mode B and C).

## Pass criteria

- Mode B vs Mode A on prefix workload: **TTFT reduction ≥ 0%, ≤ 5%**. If > 5% — investigate (could be measurement artifact or unexpected SGLang cache state).
- Mode C vs Mode A on prefix workload: **|delta| < 1.5%**. Should look like load-only.
- All modes: **per-worker request count within 30% of perfect balance** (~100 ± 30 per worker over 200 requests).

## Run procedure

1. Verify infra: `docker compose -f deploy/docker-compose.yml ps` shows etcd + nats up.
2. Kill any prior LLaDA processes; clear GPU memory.
3. `tmux new-session -d -s llada-bench` with 4 panes: worker0, worker1, frontend, controller.
4. Start worker 0 in pane 0 via `diffusion_llada_multi.sh --gpu-id 0`.
5. Start worker 1 in pane 1 via `diffusion_llada_multi.sh --gpu-id 1`.
6. Wait for both workers to log `chat endpoints enabled`.
7. For each mode in [A, B, C]:
   a. Restart frontend in pane 2 with mode-specific flags.
   b. Wait until frontend logs `chat endpoints enabled` and both workers visible.
   c. Run aiperf in pane 3, artifact dir = `bench/llada-approx-kv/results/<mode>`.
   d. Capture per-worker request counts from worker logs.
8. Run `analyse.py` to produce the comparison table.

## Cleanup

`tmux kill-session -t llada-bench` after summary is written.
