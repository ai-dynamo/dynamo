<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Parser parity / conformance moved to frontend-crates

The cross-implementation parity harness, fixture authoring and HTML table generator are owned by **[ai-dynamo/frontend-crates](https://github.com/ai-dynamo/frontend-crates)**, under `conformance/`. Dynamo consumes the parser crates and checks their external conformance integration through the standalone consumer in [`unified/`](unified/Cargo.toml).

## Verify the external Rust runner

The consumer pins both dependencies to commit `cb366dae149aa72abcd9f26aafab07699beee26f` from [frontend-crates PR #532](https://github.com/ai-dynamo/frontend-crates/pull/532):

| Crate | Version |
| --- | --- |
| `dynamo-conformance-fixtures-v2` | `0.1.0` |
| `dynamo-parsers-v2` | `0.10.7` |

The runner and parser candidate are taken from the same immutable Git revision and a committed lockfile. Parser version `0.10.7` identifies this unpublished candidate; it does not change the root workspace parser pin. This independent Cargo workspace reads the authored corpus from Cargo dependency checkout and keeps the serving dependency graph separate.

Run the integration tests from the Dynamo root:

```bash
cargo test --locked --manifest-path tests/parity/unified/Cargo.toml --all-targets -- --nocapture
```

The native test runs eight existing Qwen3 cases covering reasoning, visible text, one and two tool calls, guided output, and prefilled reasoning. The runner compares authored GOLDEN, every valid UTF-8 two-chunk split, stream/batch output, and seven adjacent stream pairs. A second test checks explicit unsupported reporting for generation constraints. A separate test process registers a parser implemented here that returns tool syntax as visible text and requires the runner to report a GOLDEN mismatch with failure status.

The root Rust test job in `.github/workflows/pre-merge.yml` runs this consumer's formatting, Clippy and tests. These are selected integration cases; see frontend-crates' `conformance/README.md` for the complete corpus, capture and report workflows.

## Select families from the command line

Use the [command wrapper](unified/run.sh) from the Dynamo root:

```bash
./tests/parity/unified/run.sh list-v2-parsers
./tests/parity/unified/run.sh qwen3
./tests/parity/unified/run.sh glm
./tests/parity/unified/run.sh all
```

The wrapper builds and invokes the local `dynamo-conformance` binary using the pinned external runner. It works from another directory when invoked by its absolute path. The equivalent direct Cargo command is:

```bash
cargo run --release --locked --manifest-path tests/parity/unified/Cargo.toml --bin dynamo-conformance -- all
```

`list-v2-parsers` lists built-in Unified v2 families and aliases directly from the parser registry. `glm` selects `glm47`; registry aliases such as `qwen3_coder`, `nemotron3` and `harmony` select their canonical family's corpus. `all` runs every canonical built-in Unified family once and continues after a family error. It does not include legacy tool-only or reasoning-only parsers. Missing or unreadable corpora are errors, never silently skipped.

A family selection runs every active authored case with default split, stream/batch and stream isolation checks. This is broader than the eight-case Qwen3 integration test. The command prints counts per family, individual failures and unsupported cases, and a final family summary. Exit status is 0 for a passing run, help or listing; 1 for conformance failures or missing coverage; and 2 for invalid arguments. Unsupported cases are a separate verdict by default; `--fail-on-unsupported` makes them blocking. Cargo build failures retain Cargo's exit status.

With this pin, `all` selects 4,516 authored cases across 15 canonical Unified families: 4,373 pass, 10 are unsupported, and 133 GOLDEN failures occur across 14 case IDs. The isolation checks report zero failures; two families pass all selected cases and 13 fail. The failures include quoted control-marker cases and argument representation mismatches. The runner reports them without changing authored GOLDEN expectations, and `all` returns status 1. Passing the small integration tests certifies only their selected cases.

Structural-tag generation constraints are unsupported by the upstream runner. To fail on those explicit unsupported records as well as parser mismatches and corpus errors, use:

```bash
./tests/parity/unified/run.sh all --fail-on-unsupported
```

The compatibility spelling `--allow-unsupported` retains the default policy; it does not make unsupported cases pass. These commands use local fixtures and need no model server or GPU. This Rust family selector does not implement a Python parser adapter, external Rust adapter loading, or JUnit output. `cargo test` itself accepts test-name filters; family selection belongs to this command.
