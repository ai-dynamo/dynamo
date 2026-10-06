<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Protocol compatibility tooling

Assess Dynamo against a pinned native server: what differs, what changed, and what
needs action? Initial coverage is vLLM chat-completion and completion request
contracts, plus selected upstream behavioral-source changes. Static analysis does
not establish runtime parity. Start with the generated `report.md`.

The accepted [layered assessment design](../../lib/llm/docs/dynamo-vllm-protocol-assessment.md#design-decision-layered-compatibility-assessment)
separates contract comparison, behavioral conformance, and optional source
investigation. It lets developers complete contract work before independently
addressing behavior. Assessment v2 implements contract-only findings and gates;
source investigation is advisory and can be skipped with `--no-investigation`.
Behavioral conformance stays `not_assessed`: this command never launches servers
or substitutes source references for runtime tests. Native-schema export migration
and the separately scoped behavioral test workflow are not completed by this change.

## Commands

Run from the repository root with Python 3.12 and `PyYAML==6.0.2`:

```sh
python -m scripts.protocol_compatibility --help
python -m scripts.protocol_compatibility assess --help
python -m scripts.protocol_compatibility check-pins --help
python -m scripts.protocol_compatibility generate-inventory --help
python -m scripts.protocol_compatibility generate-release-fixtures --help
```

`assess --platform ...` selects configured pins and supports a first baseline,
Dynamo changes, version bumps, retained history and read-only upstream candidates.
`assess --upstream-commit ...` selects an explicit revision for historical Dynamo
commits without pins; use `--previous` to compare against a retained assessment.
Choose exactly one of `--platform` and `--upstream-commit`.

Read the [assessment guide](../../lib/llm/docs/dynamo-vllm-protocol-assessment.md)
for prerequisites, copyable commands, exact-revision examples, decision records,
coverage limits and gate semantics. Exit 0 means static gates passed, 1 means
action remains, and 2 means invalid inputs or a tool failure. No command imports
upstream engine modules or adopts candidate pins.

`generate-inventory` writes the compact Rust field vocabulary; detailed JSON is
optional through `--inventory-output`. Both support freshness checking via
`--check`. `generate-release-fixtures` retains pinned historical adapter excerpts
for the separate mixed-version test suite; it is not part of direct assessment
and does not prove N-2 runtime interoperability.

## Workflow parts

| Directory | Responsibility |
| --- | --- |
| `inputs/` | Validate source pins, choose revisions, validate retained report/decision/policy inputs |
| `extraction/` | Derive Python/Rust contract and handling facts, resolving reachable dependencies |
| `assessment/` | Compare contracts, track finding lifecycle, select upstream changes, apply decisions and gates |
| `reporting/` | Render the developer-readable assessment |
| `generation/` | Generate compact vocabulary, optional detailed inventory and historical release fixtures |
| `common/` | Shared contracts, repository paths, immutable Git reads and provenance |
| `tests/` | Mirror workflow groups; `integration/` exercises the public CLI and CI wiring |

`__main__.py` owns the public command interface. `assessment/workflow.py` connects
the stages and writes report artifacts. Extractors never import assessment policy
or renderers. The common layer does not import workflow stages. Do not add new
flat scripts or a second per-upstream-pair decision/reporting workflow.

Pins, reviewed decisions, support policy and generated Rust vocabulary remain
under `lib/llm/src/protocols/openai/compatibility/`. Reports and source snapshots go
to a caller-selected output directory, not into this package. Historical release
fixtures stay next to their backend tests. SGLang and TensorRT-LLM remain future
adapters, not implied coverage.

## Development

Run the complete source-only suite in the permitted validation environment:

```sh
python -m unittest discover -s scripts/protocol_compatibility/tests -t . -v
```

Tests use synthetic Git repositories, temporary outputs and no model or GPU.
Real-source assessment examples are separate evidence from runtime conformance.
When moving or adding modules, update imports, CI path filters/test discovery and
documentation. Provenance hashes implementation Python files recursively, excluding
tests; no hand-maintained helper list is needed. Regenerate generated artifacts
rather than editing their headers or contents by hand.
