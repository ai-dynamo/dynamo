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

This package composes and compares schemas from the independent native export;
see the [composition guide](../../lib/llm/docs/protocol-openapi-composition.md).
Native schema export is an independent prerequisite. Offline composition lives
in this tooling package alongside independent request and response comparison.
Native `/openapi.json` still contains explicit dependency import slots;
composition does not modify the running frontend.

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

Run from the selected Dynamo checkout. Generate the default Dynamo input without a
GPU or listener:

```bash
cargo run --locked -p dynamo-llm --no-default-features --bin generate-frontend-openapi
```

Download the composition input pinned by the default manifest:

```bash
curl --fail --location \
  https://raw.githubusercontent.com/64bit/async-openai/884aff958c0461cce41e2d9e9b2fe4f29e76b740/openapi.yaml \
  --output openai-884aff95.yaml
```

Set `FRAMEWORK_NAME` to a lowercase identity such as `example-engine` and
`FRAMEWORK_SPEC_URL` to its direct published or live OpenAPI URL. Capture and compare:

```bash
python -m scripts.protocol_compatibility acquire \
  --server "$FRAMEWORK_NAME" --url "$FRAMEWORK_SPEC_URL" \
  --output-dir captures/framework

python -m scripts.protocol_compatibility assess \
  --dynamo docs/frontends/openapi.json --framework captures/framework \
  --framework-name "$FRAMEWORK_NAME" \
  --openai openai-884aff95.yaml --oasdiff /absolute/path/to/oasdiff \
  --output-dir assessment
```

Alternatively, pass JSON/YAML files directly as `--dynamo` and `--framework`.
To package a local file with acquisition metadata, use `acquire --file framework.yaml`
instead of `--url`. For optional Dynamo HTTP capture, use
`acquire --server dynamo --url http://127.0.0.1:8000/openapi.json` with a fresh
`--output-dir`. The generated Dynamo file still needs composition, performed by `assess`.

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

The framework identity does not select an adapter or schema exceptions. URLs must
be direct HTTP(S), with no credentials, query strings, fragments, or redirects.
Downloads are bounded to 32 MiB and 30 seconds; proxies are disabled. Review paths,
URLs, and annotations for secrets before publishing artifacts.

## Outputs and Exit Codes

Acquisition writes exact JSON/YAML bytes to `openapi.raw` and observations to
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

The directory also retains input and composition/coverage evidence. Invalid capture
checksums, pinned OpenAI checksum mismatches, and unresolved imports fail visibly.

| Exit | Meaning |
| --- | --- |
| `0` | No reported declared differences or known gaps |
| `1` | Differences or coverage gaps require review |
| `2` | Invalid inputs or tooling failure |

None of these statuses proves runtime parity. The current coverage catalog prevents
a complete-coverage result until its known limitations are resolved.

## Response comparison boundaries

The report lists request candidates first and response candidates separately.
Both compare the framework export to Dynamo; a structural delta is a candidate,
not a compatibility decision. To establish that a framework client can use Dynamo,
verify request acceptance and semantics and whether Dynamo responses meet that
client's expectations. Reverse-direction and additive differences need separate
interpretation, not automatic fixes.

Response comparison keeps status codes, media types and JSON body schemas. For
SSE it requires OpenAPI 3.2 `text/event-stream.itemSchema`: a parsed event object
with string `data`. A JSON data branch uses `contentMediaType: application/json`
and `contentSchema`; a literal marker uses `const: "[DONE]"`.

The pinned comparison engine does not consume these new media keywords directly.
The tooling therefore creates an explicitly comparison-only OpenAPI 3.1 copy:
`itemSchema` becomes an event `schema`, and data becomes an object tagged with
`encoding: json` plus its decoded `value`, or `encoding: literal` plus the marker.
This makes the embedded payload constraints visible to the comparator without
confusing the literal `[DONE]` marker with JSON containing the string `"[DONE]"`.
Raw server exports remain unchanged. Real-binary tests check that changes inside
`contentSchema` and to the marker produce deltas, while equivalent inline/local
event references do not.

Only these explicit data encodings are supported. Encoded-string constraints,
unsupported keywords or media types, unresolved references/imports, and missing
body schemas are reported as coverage gaps. A description-only error response is
not proof that the body is empty. Response headers, other endpoints, event ordering,
termination timing and cancellation are outside this comparison's coverage.
Generic JSON Schema validation does not assert JSON inside a `contentSchema`:
real-server tests must parse SSE, decode and validate JSON data explicitly, and
check stream lifecycle separately. Neither the comparison projection nor the
success-payload fixtures replace that E2E evidence.

After parsing a data-bearing SSE event, call
`assessment.responses.validate_sse_event(document, item_schema, event)` to check
its envelope and encoded content. Supply a composed document with local payload
references and the endpoint's `itemSchema`; `event["data"]` must still be a string.
The helper returns `None` on success and raises on invalid or unsupported input.
It does not parse raw frames, count events, or establish a valid stream lifecycle.

## Validation

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

Real-comparator tests explicitly skip without `OASDIFF_BIN`; that skip is not
validation evidence. CI supplies the verified binary. See the
[guide's validation boundaries](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/openapi-request-contract-comparison.md#maintenance-and-validation)
for what these checks establish.
