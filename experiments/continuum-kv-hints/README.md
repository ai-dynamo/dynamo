<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Continuum KV hint experiments

This directory contains the experiment-only policy and runners used to validate the Continuum-style session-aware KV hint prototype.

```text
experiments/continuum-kv-hints/
  policy/                             post-selection session KV hint policy crate
  session-prefix-indexer/             SessionPrefixIndexer CPU microbenchmark crate
  dynamo-vllm-boundary/smoke.py       in-process Dynamo-to-vLLM boundary smoke
  networked-e2e/                      Dynamo vLLM constrained-cache runners (legacy directory name)
  weka-correctness/                   reduced WEKA fixture and end-to-end correctness replay
```

The policy resolves the selected worker's known session lineage into external block hashes. A final session request emits `kv.evict`; an optional fixed-retention configuration emits `kv.retain`. Both actions execute after the carrying request completes and include that request's blocks.

Run the focused checks from the Dynamo repository root:

```bash
cargo test -p dynamo-kv-hint-policy-example
cargo run -p dynamo-continuum-session-prefix-bench --example lineage_sanity
cargo bench -p dynamo-continuum-session-prefix-bench --bench session_prefix_index -- \
  --warm-up-time 1 --measurement-time 3 --sample-size 30
```

The boundary smoke requires the matching vLLM experiment branch and Dynamo Python environment. Build instructions are in [`BUILD.md`](./BUILD.md). Results are archived in `dynamo-workflows/dynamo/routing/agentic-kv-management/experiments/continuum-kv-hints/`.
