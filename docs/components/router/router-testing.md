---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Router Testing
subtitle: Test layers for router changes
---

## Overview

The router has three useful test layers. When you add a non-trivial or potentially breaking feature, do not stop at the smallest local test by default. Consider extending the relevant layer or layers below so the change is covered at the same level where it can fail.

## 1. Rust Unit and Integration Tests

Use Rust tests in `lib/kv-router` and `lib/llm` for local correctness:

- cost-model math
- indexer behavior
- event application
- recovery and persistence logic
- sequence-tracking invariants
- remote-indexer query behavior

These tests should be the first line of defense for narrowly scoped logic. They are the right place to pin exact edge cases and regressions close to the implementation.

Examples:

- `lib/kv-router/src/indexer/tests.rs`
- `lib/kv-router/src/sequences/*`
- `lib/llm/src/kv_router/indexer/worker_query.rs`

Typical commands:

```bash
cargo test -p dynamo-kv-router
cargo test -p dynamo-llm --no-default-features
```

## 2. Bench-Backed E2E Invariant Tests

Use the fixture-backed tests in `lib/bench/tests` when you want a realistic replay path without launching the full router stack. These tests share the same replay machinery as the Mooncake and active-sequences benches, but run in the Rust test profile and assert invariants instead of reporting benchmark numbers.

Current coverage uses the checked-in 1000-line Mooncake trace fixture:

- [active_sequences_trace.rs](../../../lib/bench/tests/active_sequences_trace.rs)
- [mooncake_trace.rs](../../../lib/bench/tests/mooncake_trace.rs)
- [mooncake_trace_1000.jsonl](../../../lib/bench/testdata/mooncake_trace_1000.jsonl)

These tests are useful for catching regressions such as:

- state not draining at the end of replay
- unexpected `WARN` or `ERROR` logs on hot paths
- duplicate-store or similar warning metrics
- indexer-specific replay behavior differences across implementations

Typical command:

```bash
cargo test --package dynamo-bench --all-targets
```

Use this layer when a feature changes router behavior over time, depends on realistic event orderings, or should hold across multiple indexer implementations.

## 3. Full Router E2E Process Tests

Use the Python tests in `tests/router` when you need the full request plane and event plane in play. These tests launch router and mocker or backend processes and exercise cross-process behavior that bench-backed replay cannot cover.

Current entry points include:

- [test_router_e2e_with_mockers.py](../../../tests/router/test_router_e2e_with_mockers.py)
- [test_router_e2e_with_vllm.py](../../../tests/router/test_router_e2e_with_vllm.py)
- [test_router_e2e_with_trtllm.py](../../../tests/router/test_router_e2e_with_trtllm.py)
- [test_router_e2e_with_sglang.py](../../../tests/router/test_router_e2e_with_sglang.py)

Use this layer for changes involving:

- process boundaries
- request routing through the Dynamo frontend or router service
- worker registration and discovery
- event-plane transport and delivery
- backend integration behavior
- startup, recovery, or lifecycle flows

Typical command:

```bash
.venv/bin/python -m pytest tests/router/test_router_e2e_with_mockers.py
```

## Recommended Usage

When a router change is non-trivial or potentially breaking, consider the following default progression:

- Add or update Rust unit tests for the local logic.
- Add or update a bench-backed invariant test if the change affects replay ordering, indexer behavior, cache-event handling, or state-drain assumptions.
- Add or update a full `tests/router` E2E test if the change depends on real processes, transport, registration, or backend interaction.

Not every change needs all three layers. But if a change can break behavior outside a single module boundary, it usually deserves more than a unit test.
