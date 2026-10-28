---
name: dynamo-perf-best-practices
description: >-
  Checks a Dynamo branch for performance problems before its pull request is opened. Reads the diff
  against main, works out which changed code runs per output token, KV block, KV event, request,
  connection or worker, and checks it against the cost mechanisms behind Dynamo's merged performance
  fixes: shared state touched by every stream, repeated or discarded work, copies and serde round trips,
  per-element memory at block scale, scaling with worker count, per-item async and GIL overhead, and
  hot-path dependencies. Also guards external semantics and, when the change claims a speedup, checks its
  evidence and tests. Reports what to fix before opening, with the mechanism and a fix for each finding.
  Use before opening a pull request. Read-only unless asked to apply fixes. Not for deployment knob tuning
  (consult-perf-knowledge) or benchmark campaigns (dynamo-frontend-benchmark).
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

A pre-PR check. Run it on your branch before opening a pull request. It reads what the branch
changes, works out how often each changed piece of code runs, and checks that code against the cost
mechanisms Dynamo's performance fixes have had to remove. The output is a short verdict: what to fix
before opening, what to consider, and what the PR description still needs.

The principles below are the mechanisms behind about 90 merged performance PRs. The PRs linked here
and in [references/examples.md](references/examples.md) illustrate them; they are not a checklist.
Reason from the mechanism. Flag new code whose cost times rate is concrete, whether or not Dynamo has
hit that exact pattern before, and don't flag code that resembles an example but runs rarely.

Out of scope: deployment and engine knob tuning (`consult-perf-knowledge`), benchmark campaigns
(`dynamo-frontend-benchmark`), general Rust review (`graham-code-review`), and clone-by-clone audits
(`dynamo-clone-hotpath-audit`).

## Inputs

- The branch diff, including uncommitted and untracked changes. Use the remote that points to
  `ai-dynamo/dynamo` (`upstream` in a fork checkout, `origin` in a direct clone):
  `git diff $(git merge-base HEAD refs/remotes/<remote>/main)`, plus the new files listed by
  `git ls-files --others --exclude-standard`. The full ref avoids a local branch that happens to be
  named `<remote>/main`. Don't fetch: the existing ref already holds the commit the branch started
  from. If the diff pulls in main commits that aren't the branch's, the ref is stale; say so and ask
  before fetching. If the user names a PR, commit range or patch, use that instead.
- The checkout to read callers from. If the change is not applied there, read the patched lines from
  the diff and the callers from the checkout.
- The PR description draft and commit messages, if any. Step 4 needs them.
- The surrounding code and the callers of every changed function, not only the hunks. Callers decide
  how often code runs.

Stay read-only. Report findings, and edit code only when the user asks you to apply fixes.

## Step 1: Find the multiplier

For every changed function, trace its callers and work out how often it runs in production across the
whole process. The cost that matters is per-call cost × that process-wide rate. The rates below are
already process-wide, so don't multiply by concurrency again. Concurrency matters separately when the
code touches shared state (principle 1).

| Rate | Typical process-wide scale | Where it usually lives (starting points, not a boundary) |
|---|---|---|
| Per output token or stream chunk | millions per second per frontend | `lib/llm/src/http/service`, stream converters in `lib/llm/src/protocols/openai`, detokenization in `lib/llm/src/backend.rs`, engine handlers in `components/src/dynamo/{vllm,sglang,trtllm}`, the response transport |
| Per KV block | blocks per request × workers; indexer state holds millions of entries | `lib/kv-router/src/indexer`, `lib/kv-router/src/sequences`, `lib/mocker` |
| Per KV event or frame | every engine step on every worker | event ingest, `lib/runtime/src/transports` (ZMQ, TCP, event plane) |
| Per request | thousands per second | `lib/llm/src/preprocessor`, `lib/llm/src/kv_router`, `lib/kv-router/src/scheduling`, request parsing, `lib/bindings/python/rust` |
| Per connection | thousands open at once | HTTP accept and serve, transport clients |
| Per worker or publisher | N in the thousands, at startup and on every discovery change | discovery, subscriptions, sockets, threads |

Code that runs once (configuration, startup at small N) and tests are cold. Error paths are cold only
when failures are rare: an error path that every request can hit during an outage or a burst of bad
input (logging, retries, per-failure allocation) is hot. Note cold code in one line and move on. A hot
call inside a cold-looking file still counts, so follow the calls.

## Step 2: Check the principles

For each hot hunk, ask the questions under every principle that applies. Each principle lists the
usual fixes and a few PRs that illustrate it. Read [references/examples.md](references/examples.md)
when you need a concrete before/after shape for a fix, or to decide whether an illustration really
matches a finding.

### 1. Shared state touched per item grows with total concurrency

Anything that every in-flight stream, request or worker touches on every item turns into contention
whose cost rises with the total rate, not with one request's work. That includes shared atomics and
counters, Prometheus metrics, mutexes and read-write locks, `DashMap` iteration, a shared `Notify` or
cloned `CancellationToken`, a single actor or mailbox, and `ArcSwap` stores. Cache-line bouncing and
convoys rarely show up in single-request tests.

Ask: what in this hunk is shared across requests, and how many times per second does the whole
process write or lock it?

Usual fixes: keep the state per request or per connection (accumulate locally, flush every N items
and in `Drop`); serve read-mostly data from a copy-on-write snapshot; shard by key; touch shared state
once per stream instead of once per item. Illustrations:
[#17067](https://github.com/ai-dynamo/dynamo/pull/17067),
[#16119](https://github.com/ai-dynamo/dynamo/pull/16119),
[#16118](https://github.com/ai-dynamo/dynamo/pull/16118),
[#16082](https://github.com/ai-dynamo/dynamo/pull/16082).

### 2. Do each piece of work once, at the coarsest correct granularity

Look for work done per item that could be done per stream, per request or per discovery change. Also
look for per-item messages that could be batched, for rebuilding or rescanning a whole structure when
one entry changed, for computing something no consumer reads, and for feature work that runs while the
feature is off.

Ask: does this result change between items? Who reads it? Is the feature always on?

Usual fixes: hoist (resolve handles, compile, and look up once per stream); batch with caps that never
split a logical unit; maintain an aggregate incrementally and check it with a debug-only recompute;
gate the work on the consumer's own condition. Illustrations:
[#17349](https://github.com/ai-dynamo/dynamo/pull/17349),
[#10645](https://github.com/ai-dynamo/dynamo/pull/10645),
[#12863](https://github.com/ai-dynamo/dynamo/pull/12863),
[#10273](https://github.com/ai-dynamo/dynamo/pull/10273).

### 3. Don't move bytes you don't need

Look for copies and clones of prompts, buffers and frames. Look for code that builds a
`serde_json::Value` or another intermediate tree and then converts it again, for `serde(flatten)` and
similar buffering in request parsing, for unset fields on wire frames, and for large buffers held past
their last use.

Ask: how many times are these bytes copied or re-encoded between the wire and their consumer? Does the
size grow with the prompt or the stream?

Usual fixes: move or share (`Arc`, `Bytes`) instead of copying; borrow when serializing; transcode
directly between formats; skip unset fields while staying wire-compatible; release early.
Illustrations: [#16223](https://github.com/ai-dynamo/dynamo/pull/16223),
[#11104](https://github.com/ai-dynamo/dynamo/pull/11104),
[#17238](https://github.com/ai-dynamo/dynamo/pull/17238),
[#11574](https://github.com/ai-dynamo/dynamo/pull/11574).

### 4. Per-element memory at block × worker scale

Indexers, trackers and mocker KV state hold one element per block, often per worker, which is millions
of entries. Every byte per element, every per-element allocation or refcount, and every wasted slot is
multiplied by that count.

Ask:
- How many elements are there at 2,048 workers and long prompts?
- What is `size_of` the element, counting padding, enum variants and `Option` tags?
- Does each element allocate or carry an `Arc`?
- Does the container fit the access pattern (LRU, monotonic IDs, ordered scans)?
- When is dead state reclaimed?

Usual fixes: order fields to avoid padding; box rare large variants; use sentinels instead of
`Option`; intern wide IDs; use an arena with plain counts instead of an `Arc` per element; use an
intrusive list or a direct-indexed vector where it fits; reclaim eagerly. Pin sizes with a `const`
size assertion. Illustrations: [#17109](https://github.com/ai-dynamo/dynamo/pull/17109),
[#17143](https://github.com/ai-dynamo/dynamo/pull/17143),
[#11508](https://github.com/ai-dynamo/dynamo/pull/11508),
[#12244](https://github.com/ai-dynamo/dynamo/pull/12244).

### 5. Scaling with the number of workers, publishers or connections

Look for sockets, contexts, threads or tasks created per worker; O(N) work on each event; O(N²) work at
startup or on every discovery change; and ingest serialized through one thread or a lossy hop.

Ask: what does this code do at 2,048 workers? How many threads, sockets and RPCs does it create? What
runs on each discovery change?

Usual fixes: pool shared resources; reconcile incrementally per change; give each publisher its own
lane instead of one serial hop. Illustrations:
[#17253](https://github.com/ai-dynamo/dynamo/pull/17253),
[#16568](https://github.com/ai-dynamo/dynamo/pull/16568),
[#14167](https://github.com/ai-dynamo/dynamo/pull/14167).

### 6. Per-item async, syscall and thread overhead

Look for:
- a task, `select!` or `asyncio.wait`, a future, a timer or a channel hop created per item;
- a syscall per item (timers, file I/O);
- blocking calls on async executor threads;
- work done under a lock or held across `.await`;
- an async mutex guarding a short critical section;
- one thread or actor doing all the work.

Ask: what does each item cost the runtime beyond the work itself? Can this block an executor thread?
What runs while a lock is held?

Usual fixes: one long-lived reader or adapter per stream; pin futures once; reuse timers; move blocking
work off executor threads; keep critical sections short under a synchronous mutex; split work that
saturates one thread. Illustrations: [#16351](https://github.com/ai-dynamo/dynamo/pull/16351),
[#11149](https://github.com/ai-dynamo/dynamo/pull/11149),
[#14340](https://github.com/ai-dynamo/dynamo/pull/14340),
[#11065](https://github.com/ai-dynamo/dynamo/pull/11065).

### 7. The Python boundary and the GIL

Python worker and binding code runs under the GIL, so a busy worker is capped at one core of Python.
Per-item conversions (generic pyo3 extraction, temporary strings and dicts), per-chunk asyncio
orchestration and object churn all spend that core.

Ask: how much GIL-held time does each item add? Does the code convert data element by element?

Usual fixes: exact-type fast paths with the general path as fallback; interned keys; one reader per
stream. Illustrations: [#17337](https://github.com/ai-dynamo/dynamo/pull/17337),
[#11331](https://github.com/ai-dynamo/dynamo/pull/11331).

### 8. A dependency on a hot path is hot-path code

Adding, swapping or bumping a crate or library on a hot path (HTTP server, async runtime, transport,
serialization, tokenizer) changes what runs per connection and per poll. In
[#15564](https://github.com/ai-dynamo/dynamo/pull/15564), switching the frontend listener to a server
whose connection tasks polled shared notifiers created a convoy;
[#16728](https://github.com/ai-dynamo/dynamo/pull/16728) removed it.

Ask: what does the new dependency do per item and per connection? Does it share state across
connections? Is it maintained?

### 9. Fast paths and skips must be exact

A fast path for a common case is fine when its guard is checked on every use, it falls back to the
unchanged general path, and a test shows both paths give the same result. Skipping work is safe only
when an exact predicate proves it unnecessary: the feature is off, the input is empty, or the input
equals (or fingerprints the same as) what was already processed. Never skip on a heuristic.
Illustrations: a scalar walk guarded at every step
([#17149](https://github.com/ai-dynamo/dynamo/pull/17149)); an equal-size skip that overcounted until
it also compared a fingerprint ([#16265](https://github.com/ai-dynamo/dynamo/pull/16265)).

## Step 3: Semantics guard

These apply to any change made for performance, and to any fix for loss, lag or overload under
load. A change whose purpose is new behavior is out of scope here, but its cost is not: for example,
editing every prompt changes KV prefix reuse. Treat any of these as must-fix, whatever the gain:

- Hiding loss or overload by growing queues, high-water marks or channel capacity. Drops mean a path
  does more work than it should, so find that work.
- Changing external streaming semantics, for example coalescing tokens into fewer SSE events.
- Breaking wire compatibility, in either direction, between frontends, routers and workers.
- Changing KV-event, routing or scheduling semantics as a side effect.
- Adding mock-only behavior. Mocks and benchmarks must send the same wire types and follow the same
  discovery and event paths as real workers.

## Step 4: Performance claims

Run this step when the branch claims a speedup, a memory saving, or a fix for loss, lag or overload
(in commits or the description), or when it changes a hot-path dependency or a block-scale data
structure. Also check that the diagnosis in the claim matches the code path the diff changes. Read
[references/perf-claims.md](references/perf-claims.md) and list what the PR still needs: evidence that
matches the mechanism, the base it was measured on, what was not re-measured, and tests that show
behavior is unchanged and, where the mechanism is observable, fail without the change. Don't run
benchmark campaigns from this skill. Name where the numbers can be reproduced.

## Prefer simple fixes

When you suggest a fix, start with the smallest well-known change that removes the mechanism:
sharding, a copy-on-write snapshot, per-request state, local accumulation, batching, or moving a value.
Mention more elaborate designs (lock-free structures, epochs) in one line at most. Keep suggested tests
in proportion to the fix, and don't ask for benchmarks on cold code.

## Report

Reply with the verdict in this shape. Leave out empty sections.

```markdown
## Pre-PR performance check: <branch> vs <merge-base sha>

### Fix before opening
- `lib/llm/src/path/file.rs:123` — per output token — <mechanism, one sentence> — <suggested fix>

### Consider
- `lib/kv-router/src/path/file.rs:45` — per request — <mechanism> — <suggested fix>

### Add to the PR description
- <missing evidence, caveat or test> (Step 4 only)

### Not on a hot path
- `lib/runtime/src/path/config.rs` — runs once at startup

### Outside this check
- <one line per non-performance problem noticed in passing, e.g. a correctness issue>
```

Sort findings by cost × rate, not by rate alone:

- **Fix before opening**:
  - per-token, per-block, per-event or per-connection costs;
  - anything that scales badly with worker count;
  - per-request costs that are large or can stall others: blocking I/O or syscalls on an async
    executor thread, a process-wide lock, or work that grows with prompt or payload size;
  - semantics-guard violations.
- **Consider**: smaller per-request costs with a concrete mechanism, and findings whose rate you could
  not confirm. Say what you could not confirm.
- Write one bullet per mechanism, even when two share a line. Cite line numbers in the changed version
  of the file.
- Cite an example PR only when it really matches.
- **Outside this check** is for problems you happened to notice, one line each. Don't do a general
  review.
- If nothing is hot, or nothing hot has a problem, say "Clean: no hot-path findings" and list what you
  checked.
