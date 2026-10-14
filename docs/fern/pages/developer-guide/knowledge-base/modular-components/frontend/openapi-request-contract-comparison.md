---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: OpenAPI Request-Contract Comparison
subtitle: Compare declared request and response schemas without treating agreement as support
---

Compare NVIDIA Dynamo's declared request and response schemas with an OpenAPI 3.1 or 3.2
framework contract. The `assess` command resolves dependency schemas, normalizes equivalent
representations, and reports request and response differences. It also reports coverage
gaps: contracts that the tool cannot fully check.

[Protocol Field Handling](protocol-field-handling.md) remains authoritative for semantic
ownership, the complete field lifecycle, and the two acceptance layers.

> [!IMPORTANT]
> Schema agreement does not prove request acceptance, backend effects, response behavior,
> error behavior, or streaming lifecycle compatibility.

For installation, CLI arguments, runnable examples, output files, and exit codes, use the
[tooling README](https://github.com/ai-dynamo/dynamo/blob/main/scripts/protocol_compatibility/README.md).

## Comparison Scope

Both inputs must contain JSON POST request schemas for `/v1/chat/completions` and
`/v1/completions`. If a required request schema is missing or contains an unresolved
or external reference, `assess` stops with an error.

The tool compares responses separately by status code, media type, and payload schema.
Server-sent event (SSE) comparison accepts native OpenAPI 3.2 `itemSchema` with explicit
data encoding. Source-qualified adapters also support 3.1 event envelopes or decoded
Chat chunks under `schema`. Chunk-only declarations leave other event layers unverified.
Headers, other endpoints, ordering, cancellation, and termination behavior are outside
this comparison.

The framework name labels inputs and reports; it does not select an adapter or schema
correction. To select an adapter, supply the exact input hash, source and revision labels,
and schema pointers. Keep framework versions, model, configuration, deployment, and cleanup
instructions in the framework's recipe.

## Schema Inputs and Provenance

Use `generate-frontend-openapi` as the default Dynamo source. It builds the frontend router
without a listener and writes `docs/frontends/openapi.json`; no model, GPU, or deployment is
required. File export includes all standard endpoints and optional chat dispatch events.
HTTP export includes only registered endpoints and enabled dispatch events. Capture HTTP
when you need a deployment's configured contract. Match endpoints and configuration,
and exclude credentials from retained evidence. A capture does not test inference.

The native export contains `x-dynamo-schema-import` placeholders because the pinned
`async-openai` 0.42.1 types lack the annotations needed to generate their schemas.
The composer fills these placeholders from an OpenAI document with a pinned checksum.
It applies corrections reviewed against Dynamo's Rust types and guarded by input checks.
See the [composition invariants](https://github.com/ai-dynamo/dynamo/blob/main/scripts/protocol_compatibility/README.md#composition-invariants).

`async-openai` 0.42.2 includes the annotations. Frontend-crates maintainers must enable
those schemas and replace the placeholder types. Track that integration in
[frontend-crates issue #351](https://github.com/ai-dynamo/frontend-crates/issues/351).
After Dynamo exports those schemas through updated dependencies, remove their import mappings.
Remove composition only when no placeholders remain for the compared schemas.

For each assessment, retain:

- The source revision and any local patch
- `Cargo.lock` and the exact generation command, including feature flags
- The raw schema bytes and checksum
- The composition manifest, comparator version, and coverage catalog

For a framework, prefer a revision-pinned generated, checked-in, or published JSON/YAML
document. The optional acquisition command retains exact file or HTTP(S) bytes with their
checksum and origin.

These inputs answer different questions:

| Question | Evidence |
| --- | --- |
| Can the comparison be repeated? | Exact schema bytes, composition/coverage inputs, comparator version and settings |
| Is a generated spec current with its source? | Regeneration from the recorded source and comparison with the saved spec |
| Does a deployment expose or implement that contract? | Deployment inspection and served spec; behavioral tests for actual acceptance and effects |

When a schema needs correction, update its authoritative definition and regenerate it.
Acquisition metadata and checksums support repeatability. They do not verify the source
build, dependencies, configuration, or agreement with runtime behavior. Use dependency
documents that match the versions in the composition manifest.

## Native Response Schemas

Both completion endpoints declare HTTP 200 `application/json` and `text/event-stream`
responses. Chat uses separate types for non-streaming responses and streamed chunks;
legacy completions use one type for both. The exporter derives Dynamo-owned schemas from
Rust types. Native definitions and guarded import corrections describe these serialization rules:

- Chat fields that the serializer always emits are required even when their value is null.
  These include choice `finish_reason` and `logprobs`, and message `content` and `refusal`.
- Chat messages and chunks use the configured reasoning key (`reasoning_content` by
  default, or `reasoning`). The separate `reasoning_dispatch` payload always uses
  `reasoning_content`. The offline generator uses the default reasoning key.
- Client-facing chunks exclude internal `llm_metrics`, `tool_call_completion`, and
  streaming `prompt_logprobs`. Non-streaming responses still include `prompt_logprobs`.
- Chat usage omits absent detail members; legacy usage emits null detail members.
  Separate import paths preserve that distinction.
- Legacy completions always emit nullable `usage` and `system_fingerprint`, but omit
  absent choice `logprobs` and `finish_reason`.

Derived schemas for optional fields can allow null even when serializers omit absent fields.
The export does not establish runtime numeric limits or cover every conditional value and
response variant. Non-stream error body schemas remain a coverage gap. See the
[response-schema tests](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/tests/protocols/openapi_response_schema.rs)
for export assertions and serialization fixtures.

## Additional Handling Beyond the Comparator

The tool uses pinned `oasdiff` for structural comparison, with the normalization, alias
matching, and streaming projection below. These operations do not change runtime behavior.

### Equivalent representations versus contract differences

The tool treats `anyOf: [{type: string}, {type: null}]` and
`type: [string, null]` as equivalent, regardless of order. It records each rewrite and
retains both original and normalized projections. Only bare string/null alternatives are
normalized; constraints, defaults, references, `oneOf`, other unions, and literal data
remain significant.

### Input aliases

Dynamo's `x-dynamo-input-aliases` extension lists alternative input names.
It documents Serde's rejection of multiple spellings of one field in the same object.
The extension does not add a standard JSON Schema validation rule or an output alias.

The tool matches declared aliases at the same location in a request and compares their
value schemas. Ambiguous alias ownership produces a coverage gap. A match establishes
declared name coverage only. It does not prove runtime acceptance, requiredness, union
behavior, or whole-contract compatibility. The tool retains the complete request differences.

### Streaming responses

Native OpenAPI 3.2 `itemSchema` describes parsed SSE events with string `data`.
`contentSchema` describes JSON chunks and structured errors decoded from that string;
literal `[DONE]` is a separate alternative. The chat schema includes enabled named dispatch events.
Use behavioral tests to check comment-only annotations, keep-alives, and the order of
usage and finish events.

For comparison, the tool creates schema copies that describe decoded event data.
Because oasdiff does not compare `itemSchema` or `contentSchema` directly, the copies
replace event `data` with `{encoding, value}`. JSON uses `encoding: json` and the decoded
payload schema; literals use `encoding: literal` and the marker. This distinguishes
literal `[DONE]` from JSON containing that string. Raw exports remain unchanged.
Unsupported encodings or string constraints become coverage gaps.

Generic JSON Schema validation does not check JSON encoded in a string against
`contentSchema`. Runtime tests must parse SSE, decode and validate data, and check
the stream lifecycle. A description-only error response does not establish its body schema.

For a source that declares chunks or 3.1 event envelopes, select an explicit adapter.
The adapter interprets only the selected schema pointers. See the
[adapter configuration](https://github.com/ai-dynamo/dynamo/blob/main/scripts/protocol_compatibility/README.md#source-qualified-sse-adapters).

When a source declares only chunks, the tool compares selected chunks separately and
records gaps for envelopes, encoding, errors, terminators, and extensions. Chunk agreement
does not establish event equivalence. The tool does not import Chat error envelopes from
Responses API documentation.

Event projections retain configured `tool_call_dispatch` and `reasoning_dispatch` alternatives.
Their presence alone does not prove client compatibility or malformed SSE. Use behavioral
tests to check ordering, error-path termination, and client handling.

The native exporter derives components with Utoipa 5, then adds OpenAPI 3.2 fields to
the serialized JSON. Do not deserialize that document back into Utoipa 5: its typed model
cannot preserve `itemSchema`. The bundled Swagger UI has limited `itemSchema` rendering
([upstream issue](https://github.com/swagger-api/swagger-ui/issues/11065)); inspect
`/openapi.json` for the full declaration.

## Reading the Result

The tool runs oasdiff 1.33.0 without external reference fetching. It enables `allOf`
flattening only when the corresponding projection has no coverage gaps. Original,
composed, and comparison documents remain separate.

The comparison direction is framework → Dynamo. At the compared location, an addition
exists only in Dynamo and a deletion exists only in the framework. A structural change,
such as moving a property into `allOf`, does not by itself prove that the field was removed.
Direction alone does not determine compatibility.

Start with `report.md`. Each finding shows both declarations, its location, interpretation,
evidence links, and a suggested check. `findings.json` retains location IDs and accounts for
every raw delta terminal. An unresolved union or `allOf` change remains a structural group;
an accounting pass is not a compatibility pass.

| Outcome | Meaning |
| --- | --- |
| Declared difference | Candidate for source-contract investigation, not proof of runtime incompatibility |
| Coverage gap | The export or comparison cannot establish the contract; a gap prevents complete coverage |
| Intentional exclusion | Deliberately omitted from the comparison; compatibility remains unverified |

For each candidate, check the authoritative source contract. A missing `default` annotation
does not establish a runtime default of null. A different union layout does not by itself
establish different accepted values. Use the
[field-handling development checklist](protocol-field-handling.md#development-checklist) to
collect acceptance, delivery, effect, response, and error evidence.

### From Generated Findings to Reviewed Actions

The tool inventories declarations; the reviewer decides what they mean for a client workflow:

1. Freeze the endpoints, transports, client workflow, revisions, and configuration.
   Distinguish static exports from captures of a running server.
2. Reconcile each candidate with earlier reports by location ID, delta, and declaration
   content. A stable ID does not prove unchanged content; disappearance does not prove a fix.
3. Assess request acceptance and effects separately from response compatibility.
   Extra accepted input and extra possible output have different client consequences.
4. Attach evidence with its original revisions and configuration. Distinguish declaration
   defects, runtime defects, intentional policy, and unresolved coverage. Do not present
   historical observations as new runtime results.
5. Prioritize by client impact, affected workflows, evidence confidence, and the next
   check's investigation value. Category order is navigation, not severity. Group
   candidates that share a root cause; keep uncertainty separate from impact.
6. Assign an owner and a concrete next action. Verify whether a related change is proposed,
   implemented, tested, approved, or merged, and whether the compared input includes it.

Keep proposed protocol changes separate from approved work. To regenerate only the report,
use the [saved-evidence renderer](https://github.com/ai-dynamo/dynamo/blob/main/scripts/protocol_compatibility/README.md#re-render-saved-evidence).

## Maintenance and Validation

When dependencies or framework configuration change:

- Select matching revision-pinned inputs.
- Review import mappings, guarded corrections, and coverage gaps.
- Never change a checksum to bypass a failed guard.
- Remove a coverage gap only with a fix and a test for the newly represented shape.
- Retain a fresh direct Dynamo-versus-framework assessment; an upstream-only diff is supporting
  context, not a replacement.

Rust, composed-schema, and Python tests cover export fidelity, normalization, aliases, SSE
projection, and the pinned comparator. They are not real-server or stream-lifecycle tests. See
the [tooling README](https://github.com/ai-dynamo/dynamo/blob/main/scripts/protocol_compatibility/README.md)
for validation commands and artifact details.
