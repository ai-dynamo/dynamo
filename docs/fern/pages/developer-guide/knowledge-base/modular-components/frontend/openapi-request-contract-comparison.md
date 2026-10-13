---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: OpenAPI Request-Contract Comparison
subtitle: Compare declared request and response schemas without treating agreement as support
---

Compare NVIDIA Dynamo's declared request and response schemas with an OpenAPI 3.1 or 3.2
framework contract. The workflow composes dependency-owned schemas, normalizes selected
equivalent representations, and reports request differences, response differences, and
coverage gaps.

[Protocol Field Handling](protocol-field-handling.md) remains authoritative for semantic
ownership, the complete field lifecycle, and the two acceptance layers.

> [!IMPORTANT]
> Schema agreement does not prove request acceptance, backend effects, response behavior,
> error behavior, or streaming lifecycle compatibility.

For installation, CLI arguments, runnable examples, output files, and exit codes, use the
[tooling README](https://github.com/ai-dynamo/dynamo/blob/main/scripts/protocol_compatibility/README.md).

## Comparison Scope

Both inputs must contain JSON POST request schemas for `/v1/chat/completions` and
`/v1/completions`. Missing endpoints and unresolved or remote references fail visibly
rather than producing an empty success.

Responses are compared separately by status code, media type, and payload schema.
SSE comparison requires OpenAPI 3.2 `itemSchema` and explicit data encoding; older
payload-only streaming declarations produce coverage gaps. Headers, other endpoints,
ordering, cancellation, and termination behavior are outside this comparison.

The workflow is framework-neutral within this bounded surface. A framework name identifies
provenance and reports; it does not select an adapter or schema correction. Framework recipes
own version, model, and configuration pins as well as deployment and cleanup procedures.

## Schema Inputs and Provenance

Use `generate-frontend-openapi` as the default Dynamo source. It builds the frontend router
without a listener and writes `docs/frontends/openapi.json`; no model, GPU, or deployment is
required.

The raw export contains explicit `x-dynamo-schema-import` slots for some upstream types.
Assessment resolves them from a checksum-pinned OpenAI document and applies guarded
corrections reviewed against Dynamo's Rust types. The
[composition invariants](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/docs/protocol-openapi-composition.md)
describe this dependency boundary and its implementation guards.

Retain:

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
| Is a generated spec current with its source? | Regeneration from the recorded source and a freshness diff |
| Does a deployment expose or implement that contract? | Deployment inspection and served spec; behavioral tests for actual acceptance and effects |

Change authoritative definitions and regenerate published schemas; do not hand-edit generated
output. Acquisition metadata and checksums support repeatability but do not attest the source
build, dependencies, configuration, or runtime fidelity. Select dependency documents matching
the versions recorded in the composition manifest.

### Optional Deployment Inspection

Use HTTP capture when the question is what a deployment exposes, and compare matching endpoint
and configuration scopes. Capture verifies the served declaration, not inference behavior.
Keep framework launch, recapture, and cleanup instructions in its recipe. Exclude credentials
and full environment dumps from retained evidence.

## Additional Handling Beyond the Comparator

Pinned `oasdiff` provides the structural comparison. The workflow adds the bounded
normalization, alias matching, and streaming projection described below; none redefines
runtime support.

### Equivalent representations versus contract differences

The workflow treats `anyOf: [{type: string}, {type: null}]` and
`type: [string, null]` as equivalent, regardless of order. It records each rewrite and
retains both original and normalized projections. Only bare string/null alternatives are
canonicalized; constraints, defaults, references, `oneOf`, other unions, and literal data
remain significant.

### Input aliases

Dynamo's `x-dynamo-input-aliases` extension declares alternative accepted names and rejects
multiple spellings in the same object. It describes Serde input behavior, not a standard JSON
Schema rule or output alias.

The assessor matches declared aliases at the same request instance path and compares their
value schemas. Ambiguous ownership remains a coverage gap. A match establishes declared name
coverage only; it does not prove runtime acceptance, requiredness, union behavior, or
whole-contract compatibility. The complete request delta remains unchanged.

### Streaming responses

For comparison, the response projection turns OpenAPI 3.2 `itemSchema` events into explicit
decoded-data alternatives. JSON data retains its `contentSchema`; literal data such as
`[DONE]` stays distinct. These comparison-only objects are not server payloads, and raw exports
remain unchanged. Missing or unsupported event representations become coverage gaps.

## Reading the Result

Assessment creates independent request and response projections with their reachable schema
components. Raw exports and the composed Dynamo document remain separate. The comparator is
pinned to oasdiff 1.33.0 with external reference fetching disabled; it flattens `allOf` only
when the coverage scan finds no unsupported constructs.

The direction is framework → Dynamo: additions are Dynamo-only and deletions are
framework-only. Direction alone does not classify compatibility. Start with `report.md`, then
inspect the complete delta and normalization and alias audits.

| Outcome | Meaning |
| --- | --- |
| Declared difference | Candidate for source-contract investigation, not proof of runtime incompatibility |
| Coverage gap | The export or comparison cannot establish the contract; a gap prevents complete coverage |
| Intentional exclusion | Outside the comparison claim, not a declaration of compatibility |

Resolve findings against the authoritative source contract, then use the
[field-handling development checklist](protocol-field-handling.md#development-checklist) to
collect acceptance, delivery, effect, response, and error evidence.

## Maintenance and Validation

When dependencies or framework configuration changes:

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
