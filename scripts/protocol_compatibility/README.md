<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Protocol compatibility tooling

Compare Dynamo and framework OpenAPI 3.1/3.2 request and response contracts for
`/v1/chat/completions` and `/v1/completions`. Read the
[OpenAPI request-contract comparison guide](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/openapi-request-contract-comparison.md)
for scope, provenance, normalization, aliases, and interpretation. Schema agreement
does not establish actual acceptance or behavioral compatibility.

The `assess` command fills dependency placeholders in Dynamo's native export, then
compares requests and responses separately. The guide explains the temporary
[async-openai annotation gap and removal condition](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/openapi-request-contract-comparison.md#schema-inputs-and-provenance).
Composition does not modify the running frontend's `/openapi.json` or `/docs`.

## Installation

On a Linux validation host, install the pinned Python requirements and
[oasdiff 1.33.0](https://github.com/oasdiff/oasdiff/releases/tag/v1.33.0).
Verify the release checksum for your platform; CI pins the Linux amd64 checksum.

```bash
python -m pip install -r scripts/protocol_compatibility/requirements.txt
python -m scripts.protocol_compatibility acquire --help
python -m scripts.protocol_compatibility assess --help
```

## CLI Examples

Run these commands from the selected Dynamo checkout. Use new output directories for
`acquire` and `assess`.

1. Generate Dynamo's default schema. This command needs no GPU and starts no HTTP listener.

   ```bash
   cargo run --locked -p dynamo-llm --no-default-features --bin generate-frontend-openapi
   ```

2. Download the OpenAI schema pinned by the composition manifest.

   ```bash
   curl --fail --location \
     https://raw.githubusercontent.com/64bit/async-openai/d453592328fdbb3f59c1ea74f8e6cdf7f9140efb/openapi.yaml \
     --output openai-d4535923.yaml
   ```

3. Set `FRAMEWORK_NAME` to a lowercase identifier such as `example-engine`.
4. Set `FRAMEWORK_SPEC_URL` to the direct URL of the framework's published or served OpenAPI schema.
5. Capture the framework schema.

   ```bash
   python -m scripts.protocol_compatibility acquire \
     --server "$FRAMEWORK_NAME" --url "$FRAMEWORK_SPEC_URL" \
     --output-dir captures/framework
   ```

6. Compare the schemas. Replace `/absolute/path/to/oasdiff` with the installed executable's path.

   ```bash
   python -m scripts.protocol_compatibility assess \
     --dynamo docs/frontends/openapi.json --framework captures/framework \
     --framework-name "$FRAMEWORK_NAME" \
     --openai openai-d4535923.yaml --oasdiff /absolute/path/to/oasdiff \
     --output-dir assessment
   ```

Alternatively, pass JSON/YAML files directly as `--dynamo` and `--framework`.
To package a local file with acquisition metadata, use `acquire --file framework.yaml`
instead of `--url`. For optional Dynamo HTTP capture, use
`acquire --server dynamo --url http://127.0.0.1:8000/openapi.json` with a fresh
`--output-dir`. The `assess` command composes the Dynamo schema before comparison.

### Arguments

| Command | Argument | Meaning |
| --- | --- | --- |
| `acquire` | `--server` | Required identity: `dynamo` or a lowercase framework slug |
| `acquire` | `--url` / `--file` | Exactly one HTTP(S) URL or local JSON/YAML file |
| `acquire` | `--metadata` | Optional JSON annotations, retained as unverified `caller_metadata` |
| `assess` | `--dynamo`, `--framework` | Required files or checksummed capture directories |
| `assess` | `--framework-name` | Required non-Dynamo identity; must match capture metadata |
| `assess` | `--openai` | Required pinned OpenAI YAML; checksum checked against the manifest |
| `assess` | `--manifest` | Optional composition manifest; defaults to `composition/async-openai-0.42.1.yaml` within this package |
| `assess` | `--oasdiff` | Required path to the pinned oasdiff 1.33.0 executable |
| Both | `--output-dir` | Required output directory that must not already exist |

The `acquire` command requires HTTP(S) URLs that return HTTP 200 without redirects.
It rejects credentials, query strings, and fragments in URLs. It limits captures to
32 MiB, uses a 30-second socket timeout, and disables proxies. Review paths, URLs,
and annotations for secrets before publishing artifacts.

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
`openapi.composed.json` and `composition.yaml`. Keep the raw capture and acquisition
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

The `acquire` command writes exact JSON/YAML bytes to `openapi.raw` and observations to
`acquisition.json`. Legacy v1 capture directories with `openapi.raw.json` remain
readable; their deployment metadata is not reverified.

Start with `assessment/report.md`. Retain the entire assessment directory:

| Output | Purpose |
| --- | --- |
| `report.md`, `report.json` | Findings, coverage gaps, exclusions, framework identity, and input provenance |
| `dynamo.composed.json` | Dynamo document after pinned imports and guarded corrections |
| `*.requests.json` | Original request projections |
| `*.requests.normalized.json`, `normalization.json` | Comparison inputs and equivalent-representation audit |
| `oasdiff.json` | Complete comparator delta on normalized inputs |
| `*.responses.json`, `*.responses.normalized.json` | Response projections and comparator inputs |
| `response-oasdiff.json`, `response-normalization.json` | Response deltas and equivalent-representation audit |
| `alias-matches.json` | Alias-name matches, parent constraints, and value-schema comparison results |

The `assess` command also saves inputs, the composition manifest, and the coverage catalog.
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

## Response comparison boundaries

The report lists request differences first, then response differences, in the direction
framework → Dynamo. To assess compatibility, verify request acceptance, request semantics,
and whether Dynamo responses meet the client's expectations. Evaluate reverse-direction
comparisons and Dynamo-only additions separately before changing either contract.

Response comparison keeps status codes, media types and JSON body schemas. For
server-sent events (SSE), it requires OpenAPI 3.2 `text/event-stream.itemSchema`:
a parsed event object with string `data`. JSON data uses `contentMediaType: application/json`
and `contentSchema`; a literal marker uses `const: "[DONE]"`.

The pinned `oasdiff` version does not compare these media keywords directly.
The tool therefore creates an OpenAPI 3.1 copy for comparison:
`itemSchema` becomes an event `schema`, and data becomes an object tagged with
`encoding: json` plus its decoded `value`, or `encoding: literal` plus the marker.
This makes the embedded payload constraints visible to the comparator without
confusing the literal `[DONE]` marker with JSON containing the string `"[DONE]"`.
Raw server exports remain unchanged. Tests with the `oasdiff` executable verify that
changes inside `contentSchema` and to the marker produce differences. Equivalent inline
schemas and local references produce no differences.

The tool supports only these encodings. It reports encoded-string constraints,
unsupported keywords or media types, unresolved response references/imports, and missing
body schemas as coverage gaps. An error response with only a description is
not proof that the body is empty. Response headers, other endpoints, event ordering,
termination timing and cancellation are outside this comparison's coverage.
Generic JSON Schema validation does not validate the JSON encoded inside a string
against `contentSchema`. Tests against a running server must parse SSE, decode and validate
JSON data, and check the stream lifecycle. Schema comparison and payload fixtures alone
do not establish that behavior.

After parsing a data-bearing SSE event, call
`assessment.responses.validate_sse_event(document, item_schema, event)` to check
its envelope and encoded content. Supply a composed document with local payload
references and the endpoint's `itemSchema`. Keep `event["data"]` as a string.
The helper returns `None` on success or raises an exception for invalid or unsupported input.
It does not parse raw frames, count events, or establish a valid stream lifecycle.

## Validation

Rust serialization and composed-schema checks share fixtures under
`lib/llm/tests/fixtures/openapi/`. Add a shared case when correcting a schema/Serde
disagreement, and keep each correction tied to its Rust-code rationale.

```bash
OASDIFF_BIN=/absolute/path/to/oasdiff \
  python -m unittest discover -s scripts/protocol_compatibility/tests -t . -v
cargo test --locked -p dynamo-llm --no-default-features \
  --test openapi_request_schema --test openapi_request_fidelity
python -m scripts.protocol_compatibility.tests.composition.fidelity \
  --spec assessment/dynamo.composed.json
python -m scripts.protocol_compatibility.tests.composition.response_fidelity \
  --spec assessment/dynamo.composed.json
```

Tests that execute `oasdiff` skip when `OASDIFF_BIN` is unset. A skipped test does not
validate the comparator. CI supplies the verified binary for fixture tests and separately
generates the native schema artifact. CI does not compose that artifact or run `assess`
against a pinned framework schema. See the
[guide's validation boundaries](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/openapi-request-contract-comparison.md#maintenance-and-validation)
for what these checks establish.
