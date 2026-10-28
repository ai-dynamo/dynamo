---
name: dynamo-perf-best-practices
description: Checks a Dynamo change for performance problems in code that runs per output token, KV block, KV event, request, connection or worker. Compares it against the cost mechanisms behind Dynamo's merged performance fixes, guards external semantics, and checks the evidence behind performance claims. Use when making or reviewing a pull request that touches hot-path code; pr-checklist runs it for its Performance section. Read-only unless asked to apply fixes. Not for deployment tuning (consult-perf-knowledge) or benchmark campaigns (dynamo-frontend-benchmark).
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - performance
    - code-review
    - rust
    - python
    - hot-path
---

# Dynamo Performance Best Practices

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Usage

Use this skill when making or reviewing a pull request that changes hot-path code. Read the PR under review, or the branch diff: `git diff $(git merge-base HEAD refs/remotes/<remote>/main)` plus the new files from `git ls-files --others --exclude-standard`, where `<remote>` points to `ai-dynamo/dynamo` (`upstream` in a fork, `origin` in a direct clone). Don't fetch; if the diff pulls in main commits that aren't the branch's, say the ref is stale. Read the callers of every changed function, not only the hunks. An existing cost counts when the diff adds a caller or raises its rate; cite the new call site. Stay read-only unless asked to apply fixes.

The mechanisms below come from about 90 merged performance PRs. The linked PRs illustrate them; they are not a checklist. Flag any changed code whose cost × rate is concrete, whether or not a past PR matches it, and don't flag code that looks like an example but runs rarely. General Rust review and clone audits belong to `graham-code-review` and `dynamo-clone-hotpath-audit`.

## 1. How Often It Runs

Trace each changed function's callers and work out how often it runs across the whole process. The cost that matters is per-call cost × that rate. Concurrency matters separately, through shared state (mechanism 1).

| Rate | Where it usually lives (starting points, not a boundary) |
|---|---|
| Per output token or stream chunk: millions/s per frontend | `lib/llm/src/http/service`, `lib/llm/src/protocols/openai`, `lib/llm/src/backend.rs`, `components/src/dynamo/{vllm,sglang,trtllm}`, the response transport |
| Per KV block: blocks per request × workers; millions of entries held | `lib/kv-router/src/indexer`, `lib/kv-router/src/sequences`, `lib/mocker` |
| Per KV event: every engine step on every worker | event ingest, `lib/runtime/src/transports` |
| Per request: thousands/s | `lib/llm/src/preprocessor`, `lib/llm/src/kv_router`, `lib/kv-router/src/scheduling`, request parsing, `lib/bindings/python/rust` |
| Per connection: thousands open | HTTP accept and serve, transport clients |
| Per worker or publisher: N in the thousands, at startup and on every discovery change | discovery, subscriptions, sockets, threads |

Code that runs once, tests, and error paths that fail rarely are cold; note them in one line. An error path that every request can hit during an outage or a burst of bad input is hot, and so is a hot call inside a cold-looking file.

## 2. Mechanisms

1. **Shared state touched per item.** Anything every stream or request touches per item (atomics, metrics, locks, `DashMap` iteration, `len` or `is_empty`, which lock every shard, a shared `Notify` or `CancellationToken`, one actor) becomes contention that grows with total load. Ask how many times per second the whole process writes or locks it. Fix: accumulate per request and flush every N items and on drop, serve read-mostly data from a copy-on-write snapshot, shard, or touch it once per stream. [#17067](https://github.com/ai-dynamo/dynamo/pull/17067), [#16119](https://github.com/ai-dynamo/dynamo/pull/16119), [#16082](https://github.com/ai-dynamo/dynamo/pull/16082)
2. **Work repeated per item or never read.** Look for per-item work that could run once per stream, request or discovery change; per-item messages that could be batched; full rescans when one entry changed; results no consumer reads; and work for a feature that is off. Fix: hoist, batch, maintain aggregates incrementally, gate on the consumer's condition. [#17349](https://github.com/ai-dynamo/dynamo/pull/17349), [#10273](https://github.com/ai-dynamo/dynamo/pull/10273), [#12863](https://github.com/ai-dynamo/dynamo/pull/12863)
3. **Copies and re-encoding.** Look for copied prompts, buffers and frames, `serde_json::Value` or other intermediate trees, `serde(flatten)` in request parsing, unset wire fields, and buffers held past their last use. Fix: move or share (`Arc`, `Bytes`), borrow, transcode directly, release early. [#17238](https://github.com/ai-dynamo/dynamo/pull/17238), [#11104](https://github.com/ai-dynamo/dynamo/pull/11104), [#16223](https://github.com/ai-dynamo/dynamo/pull/16223)
4. **Per-element memory at block × worker scale.** Indexer, tracker and mocker state holds millions of elements, so every byte, allocation, `Arc` and padding slot per element is multiplied. Ask the element's size (in Rust, `size_of` with padding and enum or `Option` tags), whether it allocates, whether the container fits the access pattern, and when dead state is reclaimed. Fix: reorder fields, box rare variants, intern wide IDs, use an arena instead of an `Arc` per element, reclaim eagerly, and pin sizes with a `const` assert. [#17109](https://github.com/ai-dynamo/dynamo/pull/17109), [#17143](https://github.com/ai-dynamo/dynamo/pull/17143), [#12244](https://github.com/ai-dynamo/dynamo/pull/12244)
5. **Scaling with workers.** Look for sockets, contexts, threads or tasks per worker; O(N) work per event; O(N²) work at startup or on discovery changes; and ingest serialized through one hop. Ask what the code does at 2,048 workers. Fix: pool, reconcile incrementally, give each publisher its own lane. [#17253](https://github.com/ai-dynamo/dynamo/pull/17253), [#16568](https://github.com/ai-dynamo/dynamo/pull/16568)
6. **Per-item async, syscall and thread cost.** Look for a task, `select!`, timer or channel hop per item; a syscall per item; blocking calls on executor threads; locks held across `.await` or around slow work; and one thread doing all the work. Fix: one long-lived reader per stream, reused timers, blocking work moved off the executor, short critical sections, and saturated threads split. [#16351](https://github.com/ai-dynamo/dynamo/pull/16351), [#11149](https://github.com/ai-dynamo/dynamo/pull/11149), [#11065](https://github.com/ai-dynamo/dynamo/pull/11065)
7. **Python and the GIL.** A busy Python worker gets one core of Python, and per-item conversions, temporary objects and per-chunk asyncio orchestration all spend it. Fix: exact-type fast paths with a general fallback, interned keys, one reader per stream. [#17337](https://github.com/ai-dynamo/dynamo/pull/17337), [#11331](https://github.com/ai-dynamo/dynamo/pull/11331)
8. **Hot-path dependencies.** Adding, swapping or bumping a crate or library on a hot path (HTTP server, async runtime, transport, serializer, tokenizer) changes what runs per connection and per poll. Read its source at the locked version: what it does per item, and whether it shares state across connections. [#15564](https://github.com/ai-dynamo/dynamo/pull/15564) swapped the frontend listener and introduced a convoy that [#16728](https://github.com/ai-dynamo/dynamo/pull/16728) removed.
9. **Inexact fast paths and skips.** A fast path checks its guard on every use, falls back to the unchanged general path, and has a test showing both give the same result. Skip work only when a predicate proves it unnecessary: the feature is off, or the input is empty or equal to what was already processed. Equal sizes or counts don't prove equal contents, and a hash fingerprint is not equality; it needs an exact fallback or a documented, accepted collision bound. [#17149](https://github.com/ai-dynamo/dynamo/pull/17149), [#16265](https://github.com/ai-dynamo/dynamo/pull/16265)

## 3. Semantics

A performance change, or a fix for loss, lag or overload, must not do any of the following, whatever the gain:

- Grow queues, high-water marks or channel capacity to hide drops. Find the slow path instead.
- Change streaming semantics, such as coalescing tokens into fewer SSE events.
- Break wire compatibility, in either direction, between frontends, routers and workers.
- Change KV-event, routing or scheduling semantics as a side effect.
- Add mock-only behavior. Mocks and benchmarks send production wire types and follow the real discovery and event paths.

## 4. Performance Claims

When the PR description or commit messages claim a speedup, a memory saving, or a fix for loss, lag or overload, check that the diagnosis matches the code the diff changes, and list what the evidence still needs. Don't run benchmark campaigns. `pr-checklist` covers the PR description.

- **Metric.** Lead with the metric that matches the mechanism: blocked threads for contention, shared writes per second for atomics, CPU ms per request or µs per token for per-item CPU, bytes per element and RSS for memory, and utilization or lag for one saturated thread. Report throughput and latency as guardrails unless the changed component limits them.
- **Fair A/B.** Recent main at a stated SHA, the same machine, alternating fresh processes, the same offered load, identical outputs for completed requests, and medians over enough runs to show the noise. For a loss or overload fix, report errors and drops for both arms.
- **Claims that don't hold.** Counted `perf record -F` samples (use period-weighted `perf report`), closed-loop requests per second on a component that isn't the bottleneck, mocks that don't send production types, numbers from an older head that don't say so, and wall-clock timing in a GIL-saturated process (use `py-spy --gil`).
- **Tests.** A test that fails on the parent commit when the mechanism is observable (a skip, a removed copy, fewer shared writes), a parity test against the old path for a rewrite, a `const` size assert for size-sensitive Rust types, and no timing thresholds in CI.
- **Where to reproduce.** The `dynamo-frontend-benchmark` skill; `lib/bench/kv_router/` (protocol in `INDEXER_BENCH.md`); `lib/bench/offline_replay_bench.rs` and the `dynamo-kv-replay-parity` skill; Criterion benches in `lib/{kv-router,llm,runtime}/benches/`.

## Report

```markdown
## Performance check: <branch, PR or patch> vs <base sha>

### Must fix
- `lib/llm/src/path/file.rs:123` — per output token — <mechanism> — <fix>

### Consider
- `lib/kv-router/src/path/file.rs:45` — per request — <mechanism> — <fix>

### Not on a hot path
- `lib/runtime/src/path/config.rs` — runs once at startup
```

**Must fix** covers:
- per-token, per-block, per-event or per-connection costs, except trivial ones such as a single hash lookup, which go under Consider;
- bad scaling with worker count;
- per-request costs that can stall other requests, such as blocking I/O on an executor thread, a process-wide lock, or work that grows with payload size;
- semantics violations;
- claims the diff doesn't support.

**Consider** covers smaller per-request costs, missing evidence, and findings whose rate you could not confirm. Say what you could not confirm.

Write one bullet per code site, naming every mechanism it hits, with new-side line numbers from the diff. Leave out empty sections. Cite a past PR only when it really matches. Suggest the simplest fix that removes the mechanism, such as sharding, a snapshot, per-request state, batching or a move, before lock-free designs. If nothing hot has a problem, write "Clean: no hot-path findings" and list what you checked. Note any non-performance bug you happened to find in one line at the end.
