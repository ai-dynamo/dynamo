---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: OpenAPI Request-Contract Comparison
subtitle: Compare declared request schemas without treating schema agreement as support
---

The OpenAPI workflow compares NVIDIA Dynamo's declared request schemas with a selected
framework server's declarations. It is a practical companion to
[Protocol Field Handling](protocol-field-handling.md), which remains authoritative for
the source contract, semantic ownership, complete field lifecycle, and the two acceptance layers.

This workflow implements only part of
[source-contract conformance](protocol-field-handling.md#verify-source-contract-and-behavioral-conformance):
declared request-schema comparison. It does not prove actual request acceptance,
forwarding or backend effects, response or error conformance, streaming behavior, or
behavioral compatibility. A schema match does not redefine what it means to support a field.

For installation, CLI arguments, runnable examples, output files, and exit codes, use the
[tooling README](https://github.com/ai-dynamo/dynamo/blob/main/scripts/protocol_compatibility/README.md).

## Comparison Scope

Both inputs must be OpenAPI 3.1 documents containing JSON POST request schemas for
`/v1/chat/completions` and `/v1/completions`. Chat completion is primary; completion is
included for compatibility. Missing endpoints, unresolved references, or remote schema
references fail visibly rather than producing an empty success.

The workflow is framework-neutral within this bounded surface, not a comparator for arbitrary
APIs. The framework name is a provenance and reporting identity, not an adapter selector or a
schema-correction switch. Framework recipes own concrete version/model/configuration pins,
deployment inspection, recapture, and cleanup; comparison does not manage server lifecycles.

## Schema Inputs and Provenance

Use `generate-frontend-openapi` as the default Dynamo schema source. It builds the actual
frontend router without starting a listener and writes `docs/frontends/openapi.json`.
Pass that file directly to the assessor. No model, GPU, or deployment is required.

The generated file is a raw export. Some upstream types are represented by explicit
`x-dynamo-schema-import` slots rather than complete schemas. Assessment resolves them
using a pinned OpenAI document and guarded corrections reviewed against Dynamo's Rust
types. Generation does not replace composition. The
[composition invariants](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/docs/protocol-openapi-composition.md)
describe the dependency boundary and implementation guards.

Retain the source revision and any local patch, `Cargo.lock`, exact generation command
including feature flags, and the output checksum. For a framework, use a revision-pinned
generated, checked-in, or published JSON/YAML spec when available. The optional acquisition
command retains exact file or HTTP(S) bytes and records their checksum and origin.

These inputs answer different questions:

| Question | Evidence |
| --- | --- |
| Can the comparison be repeated? | Exact schema bytes, composition/coverage inputs, comparator version and settings |
| Is a generated spec current with its source? | Regeneration from the recorded source and a freshness diff |
| Does a deployment expose or implement that contract? | Deployment inspection and served spec; behavioral tests for actual acceptance and effects |

A published or checked-in spec supports repeatable comparison without launching its server.
If publishing a generated spec, change its authoritative definitions and regenerate it;
do not hand-edit the output. Review the generated diff and add freshness validation to the
publishing workflow. The comparison workflow itself introduces neither a full checked-in
spec snapshot nor a freshness gate.

Acquisition observes bytes, URL/path, timestamp, and HTTP status where applicable. Caller
annotations are retained as unverified metadata. Neither those annotations nor checksums
attest the source build, dependencies, configuration, or runtime fidelity. The composition
manifest states the dependency versions its mapping was reviewed against; selecting a
matching document remains the caller's responsibility.

### Optional Deployment Inspection

Use Dynamo HTTP capture when the question is what a particular deployment exposes.
The helper's optional `--serve IP:PORT` mode exposes the actual router with in-memory
discovery and no workers; it checks HTTP schema export, not inference.

The helper enables all standard endpoint families. A deployment can enable a subset or
add routes, so compare matching endpoint and configuration scope. Even with matching
configuration, file mode additionally documents `/docs` and `/openapi.json`: those
routes are registered after the HTTP-served document is generated. The two assessed request
contracts match; full-document byte equality is not the comparison criterion.

Keep concrete framework inputs and deployment/recapture procedures in the framework's
recipe, including immutable image/source/model pins, startup configuration, selected
inspection evidence, and cleanup. Avoid credentials or full environment dumps in evidence.

## Additional Handling Beyond the Comparator

Pinned `oasdiff` provides the underlying comparison. The normalization and alias handling
below are workflow-specific additions on top of it, not new definitions of support.

### Equivalent representations versus contract differences

Before invoking `oasdiff`, the workflow treats
`anyOf: [{type: string}, {type: null}]` and `type: [string, null]` as equivalent,
in either branch/type order. This is a
comparison-only normalization, not a change to either server's exported spec.
Original `*.requests.json` files are retained alongside
`*.requests.normalized.json`; `normalization.json` and the report record each
rewrite. `oasdiff.json` is the comparator result on the normalized inputs.

Only bare string/null alternatives are canonicalized. Constraints and defaults
remain significant; constrained branches, references inside unions, `oneOf`, and
other type unions are not simplified. Literal values inside examples/defaults
are never rewritten. Equivalent encoding does not establish runtime parity.

### Input aliases

Dynamo exports `x-dynamo-input-aliases`
on `chat_template_args` (alias `chat_template_kwargs`) and assistant-message
`reasoning_content` (alias `reasoning`). This custom extension lists alternative
input names and, by definition, rejects multiple spellings of the same field in
one object, even when their values are equal. No spelling takes precedence and
no separate conflict-policy marker is needed. It describes Serde input behavior,
not a standard JSON Schema validation rule or output alias; ordinary validators
do not enforce it. Rust tests check both spellings and rejection of equal and
unequal values when both occur together.

Older captures carrying the redundant `x-dynamo-alias-conflict: reject` marker
remain supported. A contradictory legacy marker is rejected as invalid metadata.

On top of the complete request comparison, the assessor matches explicitly
declared aliases at the same request instance path, including nested message
fields. Ambiguous ownership or repeated backend
union declarations remain coverage gaps. `alias-matches.json` records the match,
parent constraints, and separate value-schema comparisons using pinned oasdiff.
Array-item path segments are represented by JSON `null`, distinct from any key.

A name match establishes declared input-name coverage, not actual acceptance or
whole-contract compatibility. Value constraints are still compared; requiredness, union context,
and simultaneous-name behavior require separate review. The complete request
delta is retained unchanged. The human-readable top-level summary classifies
matched spellings separately from genuinely missing names; it does not silently
rename properties or clear the report's incomplete-coverage status. Older captures
without these extensions cannot establish alias coverage.

## Reading the Result

Assessment projects only the two POST JSON request bodies and their reference closure.
Raw exports and the composed Dynamo document remain separate retained inputs. Scoped
validation does not certify unrelated routes or the full API document.

The comparator is pinned to oasdiff 1.33.0 with external reference fetching disabled.
It flattens `allOf` only when the coverage scan finds no unsupported constructs; otherwise
those constructs stay in the inputs and appear as gaps. This is not a general JSON Schema
equivalence prover.

The comparison direction is framework → Dynamo: additions are Dynamo-only, deletions are
framework-only. Start with `report.md`, then inspect the complete delta and the normalization
and alias audits. Review three outcomes separately:

- Declared differences are candidates for source-contract investigation, not proof of
  runtime incompatibility.
- Coverage gaps identify what the export or comparison cannot establish. The current seven
  catalogued gaps prevent a complete-coverage result, even if no new differences are found.
- Intentional exclusions, such as the contents of Dynamo-specific `nvext`, are outside the
  comparison claim, not declarations of compatibility.

The report does not replace the
[field-handling development checklist](protocol-field-handling.md#development-checklist).
Resolve findings against the authoritative source contract, then collect the missing acceptance,
delivery, effect, response, and error evidence for the claimed path.

## Maintenance and Validation

When dependencies or framework versions/configuration change, select matching revision-pinned
inputs and review import mappings, guarded corrections, and coverage gaps. Never update a
checksum just to bypass a failed guard. Remove a fidelity gap only with a fix and a test of
the newly represented input shape. Retain a fresh direct Dynamo-versus-framework assessment;
an upstream-only version diff is supporting context, not a replacement.

Python tests exercise the pipeline with synthetic framework identities and the real pinned
comparator. Shared Rust and composed-schema tests currently cover 48 aligned cases and six
known-gap witnesses; they are not exhaustive acceptance tests. The tooling README lists the
commands.

The root pre-merge Rust job runs schema generation after its tests using its default features.
Generation failure fails that job. Its 14-day `frontend-openapi-<sha>` artifact retains the
raw spec, revision, lockfile, command, and checksums. This checks generation, not composed
contract completeness, deployment correspondence, spec freshness, or behavioral conformance.
