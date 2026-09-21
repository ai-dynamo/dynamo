<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Continuum KV hint experiments

This directory contains the experiment-only policy and runners used to validate the Continuum-style session-aware KV hint prototype.

```text
experiments/continuum-kv-hints/
  policy/                           shared post-selection session KV hint policy crate
  unit-tests-and-microbenches/      focused correctness checks and microbenchmarks
  weka-small-qwen3-0.6b/            small-model WEKA lifecycle policy experiment
```

The policy resolves the selected worker's known session lineage into external block hashes. A final session request emits `kv.evict`; an optional fixed-retention configuration emits `kv.retain`. Both actions execute after the carrying request completes and include that request's blocks.

Run the focused checks from the Dynamo repository root:

```bash
cargo test -p dynamo-kv-hint-policy-example
cargo run -p dynamo-continuum-session-prefix-bench --example lineage_sanity
cargo bench -p dynamo-continuum-session-prefix-bench --bench session_prefix_index -- \
  --warm-up-time 1 --measurement-time 3 --sample-size 30
```

The boundary smoke requires the matching vLLM experiment branch and Dynamo Python environment. Build instructions are in [`BUILD.md`](./BUILD.md). Prior checks are documented under [`unit-tests-and-microbenches/`](./unit-tests-and-microbenches/), and the initial workload experiment is under [`weka-small-qwen3-0.6b/`](./weka-small-qwen3-0.6b/). Results are archived in `dynamo-workflows/dynamo/routing/agentic-kv-management/experiments/continuum-kv-hints/`.
