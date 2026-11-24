---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: OpenAPI Request-Contract Comparison
subtitle: Compare declared request and response schemas without treating agreement as support
---

NVIDIA Dynamo provides a
[protocol compatibility CLI](https://github.com/ai-dynamo/dynamo/tree/main/scripts/protocol_compatibility)
to compare its declared request and response schemas with a framework's OpenAPI 3.1 or 3.2
specification. Its `assess` command reads local JSON/YAML files, fills missing dependency
schemas, normalizes equivalent representations, and reports differences and coverage gaps:
parts of the contract it cannot fully check.

Specification generation and acquisition are separate from comparison. A local file may
come from a source exporter, a published specification, or a previous live-server capture.
The tool does not fetch schemas or require running servers.

[Protocol Field Handling](protocol-field-handling.md) remains authoritative for semantic
ownership, the complete field lifecycle, and the two acceptance layers.

> [!IMPORTANT]
> Schema agreement does not prove request acceptance, backend effects, response behavior,
> error behavior, or streaming lifecycle compatibility.

For installation, CLI arguments, runnable examples, output files, and exit codes, use the
[tooling README](https://github.com/ai-dynamo/dynamo/blob/main/scripts/protocol_compatibility/README.md).

## Comparison Scope

Compare the full declared request/response contracts for the supported endpoints, not only
the features enabled in one deployment. The tool limits comparison to those endpoints.
Do not remove differences from the inputs to manufacture agreement.

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
The production HTTP export includes only registered endpoints and enabled dispatch events.
A file captured from that endpoint is a valid input, but its configured scope may omit
variants. Record that scope; comparison cannot recover missing variants or establish that
the input covers the full contract. Capturing a schema does not test inference.

Dynamo's pinned `async-openai` 0.42.1 dependency lacks the annotations needed to generate
complete schemas. The tool fills missing definitions from a checksum-pinned OpenAI
specification and applies corrections checked against Dynamo's Rust types.
See the [composition invariants](https://github.com/ai-dynamo/dynamo/blob/main/scripts/protocol_compatibility/README.md#composition-invariants).

Newer upstream annotations are available, but Dynamo still needs the integration tracked in
[frontend-crates issue #351](https://github.com/ai-dynamo/frontend-crates/issues/351).
Composition can be removed once native exports provide all definitions needed for comparison.

For each assessment, retain:

- The source revision and any local patch
- `Cargo.lock` and the exact generation command, including feature flags
- The raw schema bytes and checksum
- The composition manifest, comparator version, and coverage catalog

For a framework, prefer a revision-pinned generated, checked-in, or published JSON/YAML
document saved locally. Assessment retains exact input bytes and checksums. Optional
per-input metadata records scope, configuration, and provenance without verification or
filtering the comparison.

These inputs answer different questions:

| Question | Evidence |
| --- | --- |
| Can the comparison be repeated? | Exact schema bytes, composition/coverage inputs, comparator version and settings |
| Is a generated spec current with its source? | Regeneration from the recorded source and comparison with the saved spec |
| Does a deployment expose or implement that contract? | Deployment inspection and served spec; behavioral tests for actual acceptance and effects |

When a schema needs correction, update its authoritative definition and regenerate it.
Input metadata and checksums support repeatability. They do not verify the source
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

### Equivalent Representations Versus Contract Differences

The tool treats `anyOf: [{type: string}, {type: null}]` and
`type: [string, null]` as equivalent, regardless of order. It records each rewrite and
retains both original and normalized projections. Only bare string/null alternatives are
normalized; constraints, defaults, references, `oneOf`, other unions, and literal data
remain significant.

### Input Aliases

Dynamo's `x-dynamo-input-aliases` extension lists alternative input names.
It documents Serde's rejection of multiple spellings of one field in the same object.
The extension does not add a standard JSON Schema validation rule or an output alias.

The tool matches declared aliases at the same location in a request and compares their
value schemas. Ambiguous alias ownership produces a coverage gap. A match establishes
declared name coverage only. It does not prove runtime acceptance, requiredness, union
behavior, or whole-contract compatibility. The tool retains the complete request differences.

### Streaming Responses

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

Read `report.md` for a concise review with declarations, specific checks, and evidence links.
Use `report.json` for the full structured findings, raw deltas, and provenance; agents do not
need to parse Markdown. Missing declaration evidence is marked unavailable. Unresolved
union or `allOf` changes remain whole findings.

The report defines categories in a suggested review order, not a severity ranking.
Declared aliases classify spelling-only differences as `alias_evidence`; value constraints
remain separate. Keep the full assessment directory so evidence pointers remain usable.

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

The tool reports declaration evidence, not a fix specification. A human or agent still needs
the source checkout and intended contract to decide whether to change Dynamo, correct an
export, or retain an intentional difference:

1. Freeze the endpoints, transports, client workflow, revisions, and configuration.
   Distinguish static exports from captures of a running server.
2. Reconcile each candidate with earlier reports by location ID, delta, and declaration
   content. A stable ID does not prove unchanged content; disappearance does not prove a fix.
3. Assess request acceptance and effects separately from response compatibility.
   Extra accepted input and extra possible output have different client consequences.
4. Attach evidence with its original revisions and configuration. Distinguish declaration
   defects, runtime defects, intentional policy, and unresolved coverage. Do not present
   historical observations as new runtime results.
5. Prioritize by client impact, affected workflows, evidence confidence, and the value of
   the next check. Group candidates that share a root cause; keep uncertainty separate
   from impact.
6. Assign an owner and a concrete next action. Resolve unclear intended behavior or scope
   with the responsible maintainer. An existing contract and authorized task can settle
   routine fixes without a separate decision for every finding.

For an approved change, add a focused regression test and regenerate affected schemas before
rerunning comparison. Runtime changes also need behavioral validation. Check whether related
work is proposed, implemented, tested, approved, or merged, and whether the input includes it.

The [saved-evidence renderer](https://github.com/ai-dynamo/dynamo/blob/main/scripts/protocol_compatibility/README.md#re-render-saved-evidence)
rebuilds interpretations and presentation with the current code. It preserves raw deltas and
does not rerun comparison; a revised report is not evidence that a fix passed.

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
