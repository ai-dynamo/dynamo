<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Parser parity / conformance moved to frontend-crates

The cross-implementation parity harness and the HTML table generator that used to live here (`generate_parity_table.py`, the engine adapters, fixtures, and the `tests/parity/**` Python tests) have moved to the standalone **[ai-dynamo/frontend-crates](https://github.com/ai-dynamo/frontend-crates)** repo, under `conformance/`. Dynamo now consumes the published `dynamo-parsers` crate instead of an in-tree `lib/parsers`, so parser conformance is owned and run there. Nothing in this directory is generated or run from dynamo anymore.

See `conformance/README.md` in frontend-crates for the full layout. Quick reference:

## Rendering the tables (from a frontend-crates checkout)

- **v1 parity HTML** (the old Dynamo parity table — v1 parser code + v1 Dynamo-synced fixtures): `conformance/utils/render_table_v1.sh` — writes `conformance/utils/.stage/tests/parity/PARITY_v1.html`.
- **v2 conformance HTML** (bridge table: `TC batch (v1)` + reasoning tabs use v1 parser code; `TC batch-on-stream (v2)` + `TC stream (v2)` use the parser-v2 code): `conformance/utils/render_table_v2.sh` — writes `conformance/CONFORMANCE.html` (accepts `--output PATH`). No engines needed.

## Running the conformance tests (from a frontend-crates checkout)

Use the repo's pinned toolchain (Rust 1.93.1 via rustup):

```bash
# v1 batch tool-calling parity, all families:
cargo test --locked -p dynamo-conformance-fixtures-v2 --test parity_toolcalling
# v2 batch-on-stream and stream:
cargo test --locked -p dynamo-conformance-fixtures-v2 --test parity_toolcalling_batch_via_stream
cargo test --locked -p dynamo-conformance-fixtures-v2 --test parity_toolcalling_stream
# whole workspace (what frontend-crates CI runs):
cargo test --workspace
```

These run pure-Rust against the fixtures (no vLLM/SGLang engine launch). The dynamo-side parser tests that consume the crate still live in dynamo at `lib/llm/tests/` (`test_jail.rs`, `test_reasoning_parser.rs`, `test_streaming_tool_parsers.rs`, etc.).
