<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Protocol Compatibility Tooling

Compare local, versioned NVIDIA Dynamo and framework OpenAPI 3.1/3.2 specifications for
`/v1/chat/completions` and `/v1/completions`. Inputs can be source exports, published
specifications, or previous server captures. The tool does not generate or fetch them
and requires no running server. Schema agreement does not prove runtime compatibility.
See the [comparison guide](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/openapi-request-contract-comparison.md)
for scope, provenance, normalization, aliases, and interpretation.

Use inputs that describe the full declared request/response contract for those endpoints,
not only one deployment's enabled features. Supplied scope and configuration labels are
retained, not used to filter differences. The tool cannot recover variants missing from
an input or attest its completeness.

The `assess` command fills dependency placeholders in Dynamo's native export, then
compares requests and responses separately. The guide explains the temporary
[async-openai annotation gap and removal condition](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/openapi-request-contract-comparison.md#schema-inputs-and-provenance).
Composition does not modify the running frontend's `/openapi.json` or `/docs`.

For the current source pin and SSE adapter recipe, see the
[upstream vLLM response assessment](../../lib/llm/docs/dynamo-vllm-response-assessment.md).

## Installation

From the repository root on a Linux validation host, install the pinned Python requirements and
[oasdiff 1.33.0](https://github.com/oasdiff/oasdiff/releases/tag/v1.33.0).
Verify the release checksum for your platform; CI pins the Linux amd64 checksum.

```bash
python -m pip install -r scripts/protocol_compatibility/requirements.txt
python -m scripts.protocol_compatibility assess --help
```

## CLI Examples

Prepare local Dynamo and framework JSON/YAML specifications and the
[pinned OpenAI schema](https://raw.githubusercontent.com/64bit/async-openai/d453592328fdbb3f59c1ea74f8e6cdf7f9140efb/openapi.yaml)
before comparison. How those files are generated or acquired is separate from this tool.
When a fresh Dynamo export is needed, run its file generator from the selected checkout:

```bash
cargo run --locked -p dynamo-llm --no-default-features --bin generate-frontend-openapi
```

This writes `docs/frontends/openapi.json` without a listener, model, or GPU. To compare it,
save the framework spec as `framework.yaml` and the pinned schema as `openai-d4535923.yaml`.
Replace `example-engine` with the framework's lowercase identifier and the comparator path
with its installed location. Use a new output directory:

```bash
python -m scripts.protocol_compatibility assess \
  --dynamo docs/frontends/openapi.json --framework framework.yaml \
  --framework-name example-engine \
  --openai openai-d4535923.yaml --oasdiff /absolute/path/to/oasdiff \
  --output-dir assessment
```

For optional scope, configuration, or provenance annotations, pass JSON object files with
`--dynamo-metadata` and `--framework-metadata`. Their contents are retained as unverified
`caller_metadata`; they do not restrict comparison or prove the inputs' source versions.

### Arguments

| Argument | Meaning |
| --- | --- |
| `--dynamo`, `--framework` | Required local OpenAPI JSON/YAML files |
| `--dynamo-metadata`, `--framework-metadata` | Optional JSON annotation files |
| `--framework-name` | Required lowercase slug, 1–64 characters, starting with a letter; allows digits, `_`, and `-`; cannot be `dynamo` |
| `--openai` | Required pinned OpenAI YAML; checksum checked against the manifest |
| `--manifest` | Optional composition manifest; defaults to `composition/async-openai-0.42.1.yaml` within this package |
| `--response-adapters` | Optional YAML with exact input hashes, source identities, server-sent event (SSE) interpretations, and bounded route mappings |
| `--oasdiff` | Required path to the pinned oasdiff 1.33.0 executable |
| `--output-dir` | Required output directory that must not already exist |

Review retained paths and annotations for secrets before publishing artifacts.

## Composition

The shared [manifest](composition/async-openai-0.42.1.yaml) records the expected
`dynamo-protocols` and `async-openai` versions, OpenAI source revision and checksum,
import mappings, and corrections reviewed against Rust serialization. When upgrading
a dependency, review these inputs and retest schema/Serde agreement and runtime
consumers. Never change a checksum to bypass a failed correction guard.

To compose without running the comparator, use the native export and pinned OpenAI
file from the CLI examples:

```bash
python -m scripts.protocol_compatibility.composition.responses \
  --raw docs/frontends/openapi.json \
  --openai openai-d4535923.yaml \
  --output-dir composed
```

The `composition.responses` command handles both request and response imports.
It requires a new output directory and writes
`openapi.composed.json` and `composition.yaml`. Keep the raw input and provenance
metadata separately. This command does not require the `oasdiff` executable.

### Composition Invariants

The composer:

- Creates a separate document. It preserves raw inputs, Dynamo endpoints, owned fields,
  and SSE envelopes. It never fills gaps from a framework's schema.
- Imports only definitions selected by placeholders and the components they reference.
  It gives imported components distinct names and rejects collisions, unmapped placeholders,
  external references, and unsupported reference constructs.
- Requires a `test` operation immediately before each JSON Patch correction.
  The test must check the affected value, or its parent for additions.
  Each correction must include a rationale based on Rust code.
- Redirects references to corrected definitions so nested uses receive the corrections.
  It preserves recursive references and traverses only schema locations.
  Literal examples and defaults remain data.

Checksums and patch guards verify input assumptions, not runtime behavior.
An explicit `serde_json::Value` is intentionally untyped. Record excluded Dynamo-specific
`nvext` fields without claiming compatibility.

## Outputs and Exit Codes

The `assess` command retains inputs under `dynamo/` and `framework/`. Each directory contains:

- `openapi.raw`: the exact input bytes, including JSON/YAML formatting.
- `acquisition.json`: the source path, input identity, SHA-256 checksum, read time
  (`captured_at`), and optional unverified `caller_metadata`.

Despite its name, `acquisition.json` records local file loading, not a server fetch.
Its timestamp does not identify when the schema was generated. To reuse an input, pass
its raw file to `--dynamo` or `--framework`. Supply provenance labels separately through
the corresponding metadata option.

Start with `assessment/report.md`. Retain the entire assessment directory:

| Output | Purpose |
| --- | --- |
| `report.md` | Human-readable findings, category counts, specific checks, and evidence links |
| `report.json` | Full machine-readable assessment: deltas, gaps, input metadata, comparator commands, and `finding_inventory` |
| `dynamo.composed.json` | Dynamo document after pinned imports and guarded corrections |
| `*.requests.json` | Original request projections |
| `*.requests.normalized.json`, `normalization.json` | Comparison inputs and equivalent-representation audit |
| `oasdiff.json` | Complete comparator delta on normalized inputs |
| `*.responses.json`, `*.responses.normalized.json` | Response projections and comparator inputs |
| `response-oasdiff.json`, `response-normalization.json` | Response deltas and equivalent-representation audit |
| `response-sources.json`, `response-adapters.yaml` | Source pointers and interpretations; adapter YAML is retained when supplied |
| `response-chunk-probes.json`, `chunk-probes/` | Separate selected-chunk comparisons when a source declares chunks only |
| `alias-matches.json` | Alias-name matches, parent constraints, and value-schema comparison results |

The `assess` command also saves the pinned OpenAI document, composition manifest, and coverage catalog.
Checksum mismatches and unresolved Dynamo imports stop the command with an error.
A later failure can leave partial output; use a new output directory for a retry.

The `assess` command returns these exit codes:

| Exit | Meaning |
| --- | --- |
| `0` | No reported declared differences or known gaps |
| `1` | Differences or coverage gaps require review |
| `2` | Invalid inputs or tooling failure |

No exit code proves runtime compatibility. The current coverage catalog contains
known gaps, so a completed assessment returns `1` even if the comparator finds no differences.

### Using the Findings

Use `report.md` to review findings; it omits repeated generic checks and empty category sections.
For programmatic use, read `report.json` and its `finding_inventory`. Each finding includes
its category, declarations, interpretation, suggested action, raw delta, and evidence
pointers. Raw deltas and provenance remain supporting evidence in the same report.
Agents and scripts do not need to parse Markdown.

Finding IDs identify locations and change kinds, not unchanged content or confirmed defects.
The accounting check assigns each raw delta terminal to one finding. Unresolved union or
`allOf` changes remain whole findings. Suggested actions are investigation guidance, not
approved fixes. See the guide's
[review workflow](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/openapi-request-contract-comparison.md#from-generated-findings-to-reviewed-actions)
for deciding and validating changes.

### Re-render Saved Evidence

To rebuild finding interpretations and Markdown from retained evidence, run:

```bash
python -m scripts.protocol_compatibility.reporting.openapi \
  --assessment-dir assessment --output-dir assessment-readable
```

The output directory must be new and outside the input directory. The renderer copies
the original assessment into `evidence/` and rebuilds `report.json` and
`report.md` with the current interpretation code. Categories and suggested checks can
change when that code changes; raw deltas remain unchanged. It does not modify the input
or rerun composition or comparison. Missing normalized projections appear as unavailable
declarations, not as absent fields. `render-source.json` records source hashes and that
comparison was not rerun.
Rendering exits `0` on success or `2` on failure; success is not a compatibility verdict.

## Source-qualified SSE Adapters

See the guide's [streaming comparison model](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/openapi-request-contract-comparison.md#streaming-responses)
for decoded-event semantics and limits. The
[`validate_sse_event` helper](assessment/responses.py) checks one parsed event's envelope
and encoded payload; it does not parse frames or check stream order or termination.

Pass `--response-adapters adapters.yaml` to select a non-native representation. Each source
requires the SHA-256 of the exact raw input bytes, a source identifier, and a revision.
The tool verifies the checksum, but not the source and revision labels.
This example selects one vLLM-style envelope. Replace the checksum and revision with those
of the local input. Add one stream entry per selected endpoint and status code.

```yaml
version: 1
sources:
  framework:
    sha256: REPLACE_WITH_RAW_INPUT_SHA256
    source: vllm-project/vllm
    revision: REPLACE_WITH_SOURCE_REVISION
    streams:
      - pointer: '#/paths/~1v1~1chat~1completions/post/responses/200/content/text~1event-stream/schema'
        interpretation: event-schema
```

Choose the interpretation that matches the source declaration:

| Interpretation | Source declaration |
| --- | --- |
| `event-schema` | OpenAPI 3.1 parsed event envelope under `schema` |
| `event-item-schema` | OpenAPI 3.2 parsed event under `itemSchema` |
| `decoded-chunk` | OpenAPI 3.1 Chat object with `choices`, not an envelope with `data` |

For framework inputs, pointers select schemas in the original document. The `dynamo` source
uses the same format, but its pointers apply after composition; its checksum still identifies
the raw export. The tool reports unsupported schema forms as coverage gaps. Invalid selectors,
duplicate keys, unknown interpretations, and checksum mismatches stop the command.

When either side uses `decoded-chunk`, the tool compares selected chunks separately.
For an event declaration, set `chunk_pointer` to the JSON `contentSchema` under
`properties/data`. The pointer can pass through `anyOf` data alternatives and `anyOf` or
`oneOf` event alternatives. It cannot traverse event-envelope references. The tool preserves
payload references and reports missing or unsupported chunk selections as coverage gaps.

For that pair, the full response comparison omits both SSE body schemas but keeps status
and media differences. The tool retains the full projections in `*.responses.json`.
A chunk comparison does not check the chunk-only source's envelopes, encoding, errors,
terminators, or extensions. The tool does not infer error schemas or termination behavior.

For the retained OpenAI snapshot's unprefixed routes, add this optional source field:

```yaml
path_mappings:
  /chat/completions: /v1/chat/completions
  /completions: /v1/completions
```

Only these two mappings are supported. They affect request and response comparison copies,
not raw inputs. Missing source operations and source/target collisions are errors.
The report retains original operation pointers and server context; it does not resolve
server URLs or assert deployed endpoint identity. The retained snapshot does not establish
the latest OpenAI contract. Its missing legacy-completion SSE declaration remains missing.

## Validation

Rust serialization and composed-schema checks share fixtures under
`lib/llm/tests/fixtures/openapi/`. Add a shared case when correcting a schema/Serde
disagreement, and keep each correction tied to its Rust-code rationale.

```bash
OASDIFF_BIN=/absolute/path/to/oasdiff \
  python -m unittest discover -s scripts/protocol_compatibility/tests -t . -v
cargo test --locked -p dynamo-llm --no-default-features \
  --test protocols
python -m scripts.protocol_compatibility.tests.composition.fidelity \
  --spec assessment/dynamo.composed.json
python -m scripts.protocol_compatibility.tests.composition.response_fidelity \
  --spec assessment/dynamo.composed.json
```

When `OASDIFF_BIN` is unset, tests that execute `oasdiff` skip. A skipped test does not
validate the comparator. The protocol-tooling workflow runs the real comparator on fixtures.
The pre-merge Rust workflow separately generates and uploads the native schema artifact.
Neither workflow composes that artifact or runs `assess` against a pinned framework schema. See the
[guide's validation boundaries](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/openapi-request-contract-comparison.md#maintenance-and-validation)
for what these checks establish.

## Module Map

| Module | Responsibility |
| --- | --- |
| [`inputs.py`](inputs.py) | Keep exact input bytes and metadata separate from derived comparison documents |
| [`composition/openai.py`](composition/openai.py) | Fill dependency placeholders from pinned imports and guarded corrections |
| [`assessment/openapi.py`](assessment/openapi.py) | Project requests and match aliases; supply shared schema helpers and the oasdiff runner |
| [`assessment/responses.py`](assessment/responses.py) | Project JSON bodies, decoded events, and selected chunks; identify unsupported response declarations |
| [`assessment/openapi_workflow.py`](assessment/openapi_workflow.py) | Run an assessment and retain its inputs, projections, comparator results, and reports |
| [`reporting/findings.py`](reporting/findings.py) | Classify retained evidence and build structured findings with raw-delta accounting |
| [`reporting/openapi.py`](reporting/openapi.py) | Write JSON/Markdown reports and rebuild interpretations from saved evidence without rerunning comparison |
