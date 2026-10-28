# Performance claims

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

This step applies when a branch claims a speedup, a memory saving, or a fix for loss, lag or overload,
or changes a hot-path dependency or a block-scale data structure. The goal is a PR in which a reviewer
can see three things: what got cheaper and by how much, which base it was measured against, and that
behavior did not change. List what is missing. Don't run a benchmark campaign to fill the gaps.

## Lead with the metric that matches the mechanism

| What the change does | Lead with | Don't lead with |
|---|---|---|
| Removes lock contention or a convoy | Threads blocked on that lock, from an off-CPU profile attributed by stack, or blocked time per request | CPU ms per request, requests/s |
| Removes writes to a shared atomic or metric | Shared writes per second, before and after (e.g. per token → per 64 tokens) | Whole-process CPU |
| Removes per-item CPU work | CPU ms per request or µs per token, overall and for the affected symbols | Closed-loop requests/s |
| Shrinks memory | Bytes per element (`size_of` in Rust), plus live heap or RSS at a stated scale | Throughput |
| Unblocks one thread (actor, GIL, ingest loop) | That thread's utilization, lag, or applied events/s | — |
| Speeds up startup or reconfiguration | Median wall time over several fresh starts | — |
| Speeds up a data structure or codec | ns/op or ops/s from the existing bench binary, as a ratio of medians | — |

Report throughput and latency as guardrails ("inside noise", "unchanged") unless the changed component
is what limits them.

## A fair A/B

- **Base.** Measure against recent main and state its SHA. If the PR is meant to stack on other open
  PRs, name them.
- **Same conditions.** Run both arms on the same machine or allocation, in alternating order, as fresh
  processes, with the same warmup and measurement window.
- **Same work.** Both arms get the same offered load and complete the same requests with zero errors.
  For a fix to loss, lag or overload, the base's errors or drops are what is measured, so report them
  for both arms. Where outputs are deterministic (query checksums, replay reports, dumps), the
  completed requests' outputs are identical.
- **Enough repetitions to see the noise.** Report medians, with ranges or a bootstrap interval for
  microbenchmarks. A single run per arm is acceptable when it is labeled a point comparison.
- **Quiet machine.** Don't compile or run other heavy work during runs, and check that the load
  generator is not the bottleneck.
- **The repo's own benches first.** Prefer them over a bespoke harness. If a bespoke harness is needed,
  describe it in one line and say how to rerun it.

## Claims that don't hold

- **Counted perf samples.** On-CPU shares from `perf record -F` that count samples instead of weighting
  by period over-weight wake, park and lock paths by orders of magnitude. Use `perf report`, which is
  period-weighted, or fixed-period sampling (`-c N -P`).
- **Closed-loop requests/s on a component that isn't saturated.** With a fixed-concurrency client,
  throughput ≈ concurrency / latency, so requests/s follows the client, not the change.
- **Mocks that don't match production.** Mocks or stand-in types must send the same wire objects as
  production.
- **Stale numbers.** Numbers measured on an older head must say so: "measured at `<sha>`; not
  re-measured after `<commit>`".
- **Wall-clock time in a GIL-saturated process.** When a Python process's threads together use about
  one core, wall-clock timing measures GIL handoffs. Use GIL-held time instead (`py-spy record --gil`).
- **One run presented as a statistical result.**

## PR description

Follow the repository template (Overview, Details, Where should the reviewer start?, Related Issues):

- **Overview.** One paragraph on the mechanism ("Every X does Y, which ...; this PR ..."). Then one
  compact table (base | PR | change, with the setup in the first header cell). Then one setup line.
- **Details.** Short bullets:
  - what changed;
  - what behavior is preserved;
  - a **Cost:** bullet for anything the change adds (memory, a lag bound such as "at most 63 tokens
    unpublished", an extra code path);
  - the validation commands.
- Leave out campaign narrative, unrelated findings and future work.

## Tests

- A test that fails without the change, checked against the parent commit, whenever the change alters
  behavior or has a mechanism a test can observe (a skip, a removed copy, fewer shared writes). A pure
  rewrite with nothing observable needs the parity test below instead.
- A data structure or algorithm rewrite gets a parity or differential test against the old path or a
  simple reference model (for example a `HashMap`). Seed it if it is randomized.
- Maintained aggregates and fast paths get a check that recomputes the answer from scratch: a
  `debug_assert!` in Rust debug builds, or an assertion in the tests in Python.
- Size-sensitive Rust types get `const _: () = assert!(std::mem::size_of::<T>() == N);`.
- Removed copies get a pointer or allocation identity check.
- Keep tests in proportion to the change, and put no timing thresholds in CI tests.

## Where numbers can be reproduced

- **Frontend with mock workers:** the `dynamo-frontend-benchmark` skill.
- **KV indexer:** `lib/bench/kv_router/mooncake_bench.rs`. The protocol is in
  `lib/bench/kv_router/INDEXER_BENCH.md`; there is also a lower-tier mode.
- **Active-sequence tracking:** `lib/bench/kv_router/active_sequences_bench.rs`.
- **Mocker and offline replay:** `lib/bench/offline_replay_bench.rs`. For replay parity, use the
  `dynamo-kv-replay-parity` skill.
- **Criterion benches:** `lib/kv-router/benches/`, `lib/llm/benches/`, `lib/runtime/benches/`.
