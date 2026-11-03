<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Parser parity / conformance moved to frontend-crates

The cross-implementation parity harness, fixture authoring and HTML table generator are owned by **[ai-dynamo/frontend-crates](https://github.com/ai-dynamo/frontend-crates)**, under `conformance/`. Dynamo consumes the parser crates and checks their external conformance integration through the standalone consumer in [`unified/`](unified/Cargo.toml).

## Verify the external Rust runner

The consumer pins both dependencies to commit `8095dc76a8620cec0b65845b8fef05c675375ab9` from [frontend-crates PR #532](https://github.com/ai-dynamo/frontend-crates/pull/532):

| Crate | Version |
| --- | --- |
| `dynamo-conformance-fixtures-v2` | `0.1.0` |
| `dynamo-parsers-v2` | `0.9.6` |

The runner is unpublished, so its version is paired with an immutable Git revision and a committed lockfile. The parser version matches Dynamo's root workspace pin. This independent Cargo workspace reads the authored corpus from Cargo's dependency checkout and keeps the serving dependency graph separate.

Run from the Dynamo root:

```bash
cargo test --locked --manifest-path tests/parity/unified/Cargo.toml --tests -- --nocapture
```

The native test runs eight existing Qwen3 cases covering reasoning, visible text, one and two tool calls, guided output, and prefilled reasoning. The runner compares authored GOLDEN, every valid UTF-8 two-chunk split, stream/batch output, and seven adjacent stream pairs. A second test checks explicit unsupported reporting for generation constraints. A separate test process registers a parser implemented here that returns tool syntax as visible text and requires the runner to report a GOLDEN mismatch with failure status.

The root Rust test job in `.github/workflows/pre-merge.yml` runs this consumer's formatting, Clippy and tests. These are selected integration cases; see frontend-crates' `conformance/README.md` for the complete corpus, capture and report workflows.
