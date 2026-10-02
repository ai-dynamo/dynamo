<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Enhance frontend protocol to framework compatibility

| Attribute | Value |
|---|---|
| Status | Draft |
| Scope | `/v1/chat/completions` (primary) and `/v1/completions` (compatibility), internal request protocol, and backend adapters |
| Compatibility window | Current Dynamo release plus the two previous releases |
| Last updated | 2026-10-02 |

**Table of contents**

<!-- Regenerate with: uvx --from md-toc md_toc -p -s 6 github -l 2 lib/llm/docs/frontend-protocol-framework-compatibility.md -->
<!--TOC-->

- [Summary](#summary)
- [Context](#context)
- [How frontend compatibility works today](#how-frontend-compatibility-works-today)
- [Compatibility targets and endpoint scope](#compatibility-targets-and-endpoint-scope)
- [Goals](#goals)
- [Protocol documentation deliverables](#protocol-documentation-deliverables)
- [Upstream protocol drift detection and triage](#upstream-protocol-drift-detection-and-triage)
- [Non-goals](#non-goals)
- [Design principles](#design-principles)
- [Field placement policy](#field-placement-policy)
- [What `nvext.extra_fields` means](#what-nvextextra_fields-means)
- [Proposed architecture](#proposed-architecture)
- [Detailed case: PR #13957 (`prompt_logprobs`)](#detailed-case-pr-13957-prompt_logprobs)
- [Error model](#error-model)
- [Versioning and mixed deployments](#versioning-and-mixed-deployments)
- [Observability](#observability)
- [Test strategy](#test-strategy)
- [Definition of done](#definition-of-done)
- [Implementation plan](#implementation-plan)
- [Alternatives considered](#alternatives-considered)
- [Open questions](#open-questions)
- [Appendix A: observed compatibility incidents](#appendix-a-observed-compatibility-incidents)
- [Appendix B: `nvext` placement consensus and recommendation](#appendix-b-nvext-placement-consensus-and-recommendation)
- [Appendix C: Field placement, semantic ownership, and transport](#appendix-c-field-placement-semantic-ownership-and-transport)
- [Appendix D: Current frontend compatibility mechanism](#appendix-d-current-frontend-compatibility-mechanism)
- [References](#references)

<!--TOC-->

## Summary

Dynamo's frontend should be a faithful, fail-closed adapter between public inference APIs and
backend frameworks. A request field that Dynamo claims to support must either retain its meaning
through preprocessing, routing, transport, and backend lowering, or be rejected with an actionable
client error. Returning success after silently dropping a semantic directive is not acceptable.

This document proposes a common policy for deciding whether a field belongs in the canonical
top-level API, a namespaced backend extension envelope, or `nvext`. It also defines compatibility,
validation, observability, and test requirements. The examples in Appendix A show that the problem
has occurred in request admission, preprocessing, backend transport, and response projection.

The compatibility targets are the native HTTP servers provided by vLLM, SGLang, and TensorRT-LLM:
`vllm serve`, SGLang's server, and `trtllm-serve`, respectively. The primary endpoint is
`/v1/chat/completions`. `/v1/completions` is included to preserve compatibility for clients and
models that still rely on the legacy completions API, not as an equal surface for new feature work.

## Context

Dynamo accepts OpenAI-shaped requests but integrates with several frameworks whose APIs evolve at
different rates. A field can currently fail in several ways:

- a strict request schema rejects it before a backend can inspect it;
- an "ignore unsupported fields" mode accepts the request but drops the field;
- one frontend implementation honors it while another ignores it;
- preprocessing or stream conversion loses it between the HTTP boundary and the worker;
- the backend computes a result, but the frontend exposes it only through a Dynamo-specific path;
- HTTP and native transports carry different subsets of the request.

These failures are hard to diagnose because the response can still look plausible. For example,
dropping a modality payload can produce a valid text-only answer, and dropping
`continue_final_message` can cause the model to start a new turn instead of continuing the final
assistant prefix.

## How frontend compatibility works today

Dynamo owns a multi-stage compatibility path rather than proxying requests to the target servers.
The complete current mechanism, its existing guarantees, and the gaps addressed by this proposal
are described in
[Appendix D: Current frontend compatibility mechanism](#appendix-d-current-frontend-compatibility-mechanism).

## Compatibility targets and endpoint scope

Dynamo SHOULD behave like the selected framework's native server for every capability it declares
as supported.

### Target servers

- vLLM through `vllm serve`;
- SGLang through its native HTTP server;
- TensorRT-LLM through `trtllm-serve`.

Each profile is versioned independently. Dynamo does not need to expose the union of every option
from all three servers through one undifferentiated schema; it must accurately advertise, preserve,
or reject fields for the selected target profile.

### Endpoints in scope

- **Primary — `/v1/chat/completions`:** This is the main compatibility and feature surface. New
  protocol work, conformance coverage, and behavior-parity decisions MUST focus here first.
- **Compatibility — `/v1/completions`:** Preserve compatibility with the corresponding
  native-server behavior for supported models and existing clients. Add or change behavior only
  when required for that compatibility; do not treat this endpoint as an independent
  feature-expansion surface.

The Responses API and standalone image, video, and speech endpoints are outside the current
implementation and acceptance scope. Incidents from those endpoints may still appear in Appendix A
because they demonstrate protocol failure modes that inform the placement policy. Multimodal inputs
to the two in-scope endpoints remain in scope.

## Goals

### Contract and scope

- Define versioned compatibility profiles for `vllm serve`, the SGLang server, and
  `trtllm-serve`.
- Scope each profile to the exact target-server version or commit shipped by or explicitly
  supported with the corresponding Dynamo release. Versions outside that declared set are
  `Unverified`, not implicitly compatible.
- Establish `/v1/chat/completions` as the primary behavior-parity target and
  `/v1/completions` as a compatibility-only target.
- Define one placement policy for standard fields, framework-native fields, opaque backend
  extensions, and Dynamo-only metadata, using the independent dimensions described in
  [Appendix C](#appendix-c-field-placement-semantic-ownership-and-transport).

### Correctness and version compatibility

- Preserve the semantic value and wire location of every field declared as supported.
- Keep HTTP, preprocessing, routing, internal transport, and backend lowering behavior aligned.
- Reject unsupported or unsafe combinations before dispatch instead of silently degrading them.
- Support mixed-version frontend and worker deployments across the current release and two prior
  releases. This N-2 window applies to Dynamo frontend-to-worker compatibility; it does not imply
  support for N-2 target-server versions.

### Documentation, verification, and observability

- Publish a clear user-facing protocol specification and a developer-facing change procedure that
  define the invariants Dynamo must preserve and how contributors prove those invariants remain
  satisfied.
- Publish a compatibility reference for each target native server that identifies matching
  behavior and explains every known difference, including whether it is upstream drift awaiting
  Dynamo alignment, a Dynamo implementation gap, or an intentional divergence.
- Publish compatibility behavior as human-readable documentation and machine-readable profiles,
  expose runtime compatibility decisions through bounded metrics, and verify compatibility claims
  with conformance tests.

### Continuous upstream alignment

- Establish an owned, recurring process that detects protocol changes in each target native
  server, raises a reviewable compatibility diff, and tracks every change through alignment or an
  explicit compatibility decision.

## Protocol documentation deliverables

This effort MUST produce three complementary documentation layers:

### User protocol guide

The user protocol guide is the authoritative shared contract for the in-scope endpoints, including
field names and locations, types and defaults, validation and error behavior, streaming and
non-streaming response shapes, Dynamo extensions, and supported legacy aliases. It must link to the
per-target references for version-specific compatibility rather than duplicating their differences.

### Per-target compatibility references

Create one guide each for `vllm serve`, the SGLang server, and `trtllm-serve`. Each guide owns the
version-scoped delta between the shared Dynamo contract and that target server, compared by
endpoint, server version, request field, response field, and streaming mode. Each entry must use one
of these statuses:

- **Compatible:** Dynamo matches the versioned native-server behavior and has conformance evidence.
- **Upstream drift:** Dynamo matched the previously verified upstream baseline, but a later
  upstream version changed and Dynamo alignment is pending.
- **Dynamo gap:** Dynamo differs from a target version it claims to support, and the mismatch is not
  newly introduced by a later upstream change.
- **Intentional divergence:** Dynamo deliberately behaves differently for a documented reason.
- **Unsupported:** Dynamo does not claim the capability and rejects it explicitly.
- **Unverified:** parity has not yet been established; compatibility must not be implied.

Every non-compatible entry in a per-target reference must state the affected target and Dynamo
versions, native-server behavior, Dynamo behavior, rationale, user impact, last verification date,
and available test evidence. It must state a workaround or explicitly say that none is available.
Include a tracking issue when follow-up work remains; otherwise record the approved no-action or
intentional-divergence decision. An intentional divergence must explain why matching the native
server would be incorrect or harmful for Dynamo; it must not be used as a label for an
unprioritized implementation gap.

### Developer protocol-change guide

The developer protocol-change guide defines the procedure for adding or changing a protocol field.
It must require contributors to classify the field, update the affected target profiles and public
schema, preserve it across preprocessing and transport, implement backend lowering and response
projection, add positive and negative conformance tests, evaluate mixed-version behavior, and
update user documentation in the same change. It must also define how upstream changes are
detected, triaged, assigned, resolved, and reflected in the compatibility references.

### Cross-document invariants

The user protocol guide owns shared field definitions; per-target references own only
version-scoped deltas and statuses. The invariant is that those documents, machine-readable
compatibility profiles, generated schema, implementation, and conformance tests describe the same
behavior. A change is not complete when any of these layers disagree. Where practical, generated
documentation and CI checks should make this invariant enforceable rather than relying only on
reviewer memory.

### Current documentation baseline

Current Dynamo documentation provides parts of this contract but not yet one authoritative protocol
specification:

- [Frontend Configuration — HTTP endpoints](../../../docs/fern/pages/reference/components/frontend-configuration.mdx#http-endpoints)
  lists the exposed routes, runtime OpenAPI and Swagger endpoints, and frontend controls.
- [NVIDIA Request Extensions (`nvext`)](../../../docs/fern/pages/developer-guide/additional-resources/nvidia-request-extensions-nvext.md)
  documents Dynamo-specific request controls and optional response metadata.
- [Compatibility](../../../docs/fern/pages/reference/general/compatibility.mdx#mixed-version-compatibility)
  documents the existing N-2 frontend-to-worker guarantee and backend feature support.

These resources are the initial baseline. The new guides should consolidate or cross-link their
protocol requirements instead of duplicating them into independently maintained sources of truth.

## Upstream protocol drift detection and triage

Compatibility cannot rely only on contributors noticing upstream changes manually. Each target
server MUST have a version-aware drift check that runs on a documented schedule and whenever Dynamo
changes the supported upstream version or commit. The exact schedule is an operational choice, but
the cadence, owner, and triage response expectation must be explicit in the developer guide.

The check should compare the last verified upstream baseline with the candidate version using the
strongest available evidence:

- published or generated request and response schemas;
- upstream protocol models and server route declarations;
- release notes and protocol-related source changes; and
- runtime probes for defaults, validation, errors, response placement, and streaming behavior that
  declarations cannot establish.

The comparison must cover additions, removals, renames, type and requiredness changes, defaults,
accepted values, validation rules, error shapes, response fields, and streaming behavior for the
in-scope endpoints. Static declaration differences are candidates for investigation, not proof of
runtime behavior when the native server transforms or ignores a field.

Every run must produce a reviewable artifact recording:

- the target server, previous verified version or commit, candidate version or commit, sources
  examined, and verification date;
- the structural diff and any runtime evidence;
- one compatibility status for each material difference;
- the responsible owner, tracking issue or explicit no-action decision, and expected next step; and
- required updates to profiles, tests, schemas, and per-target compatibility references.

Triage must verify that each reported change is real rather than extraction noise, classify it
using the compatibility-reference statuses, and then do one of the following:

1. align Dynamo and add conformance coverage;
2. record a temporary upstream-drift or Dynamo-gap entry linked to tracked work;
3. document and approve an intentional divergence with its rationale and user impact;
4. mark the capability unsupported with explicit rejection behavior; or
5. mark it unverified until runtime evidence is available.

The mechanism must not automatically adopt an upstream behavior change. Resolution requires review
because a new native-server behavior can affect other target profiles, mixed-version deployments,
routing, caching, security, or the public Dynamo contract. A no-change result should still advance
the recorded last-verified version and date so users can judge the freshness of the reference.

### Framework-version bump integration and initial rollout

Any change that bumps a supported target-server version or commit MUST run the drift check against
the previous pin. The bump is not complete until its report is reviewed and every material
difference is reflected in the compatibility profile and per-target reference, resolved in the
same change, or linked to tracked follow-up work.

The initial rollout target is `vllm serve`. Initial delivery MUST include:

- a working vLLM drift check connected to the repository's vLLM version-bump workflow;
- a checked-in example report generated by that mechanism from a real comparison, identifying the
  full previous and candidate upstream vLLM commit SHAs and the exact full Dynamo commit SHA used
  for the run;
- the tool or extractor revision, inputs, structural diff, runtime evidence when required, and
  classification of every material result; and
- a reusable interface and documented expansion path for SGLang and TensorRT-LLM.

The example must use immutable commits rather than illustrative placeholders. It is evidence that
the mechanism executes as designed, not merely an example of the intended report format.

## Non-goals

- Supporting every option exposed by every target-server version.
- Expanding compatibility coverage to the Responses API or standalone media-generation endpoints.
- Adding semantics that are absent from the selected target-server contract.
- Blindly forwarding arbitrary JSON into an engine.
- Guaranteeing long-term stability for explicitly experimental backend extensions.
- Replacing typed canonical fields with a generic property bag.

## Design principles

### Preserve meaning or fail closed

For each supported `{endpoint, frontend processor, transport, target server, target version}`
profile, Dynamo MUST do one of the following:

1. preserve the field and its semantics end to end; or
2. reject the request with a 4xx error that identifies the field, selected profile, and supported
   alternatives.

Dynamo MUST NOT return success after discarding a semantic field. Compatibility switches that
ignore unknown OpenAI fields may remain as migration aids, but fields with known semantic impact
must be guarded before they can enter such a drop path.

### Prefer typed canonical fields

A field should be typed at the public boundary when it is part of the advertised OpenAI-compatible
contract, is used by more than one backend, affects frontend behavior, or requires validation before
dispatch. Typed fields provide schema generation, collision detection, validation, and a stable
place for documentation.

### Namespace opaque backend extensions

Framework-specific fields that Dynamo does not interpret may be captured at the public edge, but
they should cross internal boundaries inside a versioned, namespaced envelope. This prevents an
extension from colliding with canonical fields and makes its ownership explicit.

### Keep Dynamo metadata in `nvext`

Routing controls, internal diagnostics, and optional Dynamo response metadata belong in `nvext`.
An established framework-compatible response field should not be available only through `nvext`.

## Field placement policy

| Field class | Public representation | Internal representation | Unsupported behavior |
|---|---|---|---|
| Standard or cross-backend semantic field | Typed top-level field | Canonical typed field | 4xx validation error |
| Stable framework-compatible field | Typed top-level field, scoped in the compatibility matrix | Canonical field or typed backend lowering | 4xx capability error |
| Experimental backend-owned field | Captured from `extra_body` or an explicit extension object | Versioned `backend_extensions.<framework>` map | 4xx if the selected path cannot carry it |
| Dynamo routing or request metadata | Typed `nvext` field | Typed Dynamo extension | 4xx for invalid values |
| Optional Dynamo response metadata | Requested through `nvext.extra_fields` | Response projection selection | Omit unless requested |
| Unsafe, ambiguous, or unrepresentable field | Not admitted | Not forwarded | 4xx with a reason |

When a field exists in both a new top-level location and a legacy extension location, the top-level
field is authoritative. The legacy location may be read during a documented compatibility window,
but a conflict must not be resolved silently.

## What `nvext.extra_fields` means

`nvext.extra_fields` is an opt-in selector for optional metadata in the response's `nvext` object.
It is not a request passthrough map and does not authorize arbitrary fields.

For example:

```json
{
  "model": "example-model",
  "messages": [{"role": "user", "content": "Hello"}],
  "nvext": {
    "extra_fields": ["worker_id", "timing", "prompt_logprobs"]
  }
}
```

The currently documented selectors are `worker_id`, `timing`, `routed_experts`, `engine_data`,
`stop_reason`, `detailed_finish_reason`, `prompt_token_ids`, `completion_token_ids`, and
`prompt_logprobs`. Each selector controls a known response field. The requested data is emitted only
when it exists and the corresponding path supports it.

`prompt_logprobs` illustrates the placement rule. It may remain available as optional Dynamo
metadata for existing clients, but a top-level framework-compatible request should receive its
established top-level response without requiring an unrelated `nvext` opt-in.

## Proposed architecture

```text
Client JSON
    |
    v
Public endpoint parser
  - typed canonical fields
  - controlled extension capture
    |
    v
Validation and capability policy
  - endpoint/backend/version/transport profile
  - safety and combination checks
    |
    v
Canonical request
  - frontend-owned typed data
  - backend_extensions.<framework>
    |
    +--> preprocessing and routing fingerprint
    |
    v
Versioned frontend-to-worker transport
    |
    v
Backend adapter
  - typed lowering
  - namespaced opaque extension install
    |
    v
Engine

Engine output follows the reverse path through typed normalization and response projection.
```

### Compatibility profile

Compatibility should be described by a profile keyed by:

- public endpoint;
- frontend processor implementation;
- transport type and protocol version;
- target native server and the exact supported versions or commits;
- deployment mode, including aggregated or disaggregated serving.

Each field entry should declare its public location, internal representation, validation rules,
backend support, response location, and fallback behavior. Profiles should be machine-readable so
they can drive documentation and conformance tests rather than becoming a second manual allowlist.

> **Owner tracking note:** The necessity of a distinct machine-readable compatibility profile is
> not yet fully justified. Keep the proposal for now, but before committing to an implementation,
> validate what it provides beyond existing source types, generated schemas, conformance-test
> metadata, and documentation. Prefer deriving the profile from an existing source of truth if a
> separately authored profile would create another synchronization obligation.

### Extension envelope

The internal request should reserve a structure equivalent to:

```json
{
  "backend_extensions": {
    "schema_version": 1,
    "vllm": {
      "field_name": "opaque JSON value"
    }
  }
}
```

The actual transport type may be strongly typed rather than raw JSON. The required properties are
namespacing, an explicit schema version, bounded size and depth, collision protection, and a clear
owner. Adapters must reject an envelope for a backend that cannot install it.

### Routing and caching

An opaque field may change tokenization, modality expansion, model output, or cache identity. A
field cannot be generally supported until its routing and caching implications are classified.

- Fields that affect preprocessing must be available before routing.
- Fields that affect cache identity must contribute a stable fingerprint or disable the affected
  cache optimization.
- If Dynamo cannot safely derive the required disaggregated handoff, that deployment mode must
  reject the field.

### Security boundaries

Extension forwarding must enforce size and nesting limits and must not expose arbitrary filesystem,
plugin-loading, command-execution, or network-fetch controls. Raw extension payloads should not be
logged by default because they may contain user data. Admission should use allowlists or explicit
backend capability declarations for security-sensitive fields.

## Detailed case: PR #13957 (`prompt_logprobs`)

[PR #13957](https://github.com/ai-dynamo/dynamo/pull/13957) is an additive response-contract fix
for non-streaming `/v1/chat/completions`. It is open as of 2026-09-30.

### Public API before the PR

The request already accepted the top-level vLLM-compatible field:

```json
{
  "model": "example-model",
  "messages": [{"role": "user", "content": "1 + 1 = ?"}],
  "prompt_logprobs": 2,
  "stream": false
}
```

The worker computed prompt log probabilities, but the normal response omitted them. A client had to
make a second, Dynamo-specific request declaration:

```json
{
  "prompt_logprobs": 2,
  "nvext": {
    "extra_fields": ["prompt_logprobs"]
  }
}
```

Only then did the response contain `nvext.prompt_logprobs`. A vLLM-compatible client that sent only
the top-level request field received HTTP 200 with no corresponding response field.
`choices[].logprobs` was not a substitute: it describes generated completion tokens, while
`prompt_logprobs` describes tokens in the input prompt.

### Public API after the PR

The request schema is unchanged. For a non-streaming request where `prompt_logprobs` is present,
the response gains an optional root field next to `choices` and `usage`:

```json
{
  "model": "example-model",
  "choices": [],
  "usage": {
    "prompt_tokens": 2,
    "completion_tokens": 1,
    "total_tokens": 3
  },
  "prompt_logprobs": [
    null,
    {
      "17": {
        "logprob": -0.25,
        "rank": 1,
        "decoded_token": " hello"
      }
    }
  ]
}
```

The exact token IDs and values are backend output. The first item may be `null` because the first
prompt token has no preceding-token probability.

The compatibility behavior is:

- top-level `prompt_logprobs` in the request controls the new root response field;
- an explicit value of `0` remains distinct from an omitted field;
- `nvext.extra_fields: ["prompt_logprobs"]` continues to produce
  `nvext.prompt_logprobs` for existing clients;
- requesting both locations produces both response locations;
- streaming Server-Sent Event chunks do not serialize the request-sized root payload repeatedly;
- an absent backend payload leaves the optional response field absent, while a malformed backend
  payload is treated as an error rather than silently discarded.

### Internal protocol changes

The frontend produces client-visible streaming deltas even for a non-streaming request, then folds
those deltas into a unary response. The PR therefore adds an internal-only `prompt_logprobs` slot to
the stream response wrapper and a public optional slot to the unary wrapper.

The payload is:

1. decoded from backend `engine_data.prompt_logprobs` when requested;
2. held until a terminal delta;
3. carried through tool-call parsing, legacy parser jail paths, request tracing, and stream-to-unary
   conversion;
4. collected by the unary aggregator;
5. serialized at the non-streaming response root.

The internal stream slot is marked not to serialize, so it acts as transport metadata rather than a
new SSE wire field. The payload uses shared ownership to avoid deep-copying request-sized logprob
data across parser templates and fan-out paths.

### Why this is a compatibility example

The backend capability and request parsing already existed. The failure was in response projection:
Dynamo translated a framework-compatible top-level field into a Dynamo-only opt-in response. The
fix restores the established framework contract while retaining the legacy extension during the
compatibility window.

## Error model

Compatibility errors should use a stable machine-readable shape with:

- the rejected field and endpoint;
- the selected backend and deployment mode;
- whether the failure occurred at admission, preprocessing, transport, or backend capability
  validation;
- supported alternatives or a compatible profile when one exists.

Errors should distinguish an unknown field, a known-but-unsupported field, an invalid value, an
unsafe combination, and a mixed-version transport limitation.

## Versioning and mixed deployments

Readers should be tolerant and writers conservative across the current Dynamo release and the two
previous releases. The current representation is authoritative. Any compatibility shim must name:

- the old and new representations;
- the releases in which the shim is required;
- precedence when both are present;
- its removal condition and target release.

When a newer frontend cannot safely lower a field for an older worker, it should return a targeted
unsupported-feature error. It must not send a partial request that changes meaning.

## Observability

The frontend should record, without logging raw user payloads:

- admitted canonical and backend-extension fields;
- rejections by field, profile, and reason;
- use of legacy aliases or compatibility shims;
- fields captured by a migration-only ignore mode;
- backend payload decode failures;
- response fields requested but unavailable.

Unknown-field and ignored-field metrics should be bounded by a controlled field-name vocabulary to
avoid unbounded label cardinality.

## Test strategy

Every supported field needs tests at the following boundaries:

1. **JSON contract:** omission, `null`, `0`, `false`, empty collections, unknown keys, and conflicts.
2. **Preprocessing:** the semantic value survives chat templates, tokenization, media resolution, and
   parser transformations.
3. **Transport:** current-to-current plus current-to-N-1 and current-to-N-2 behavior.
4. **Backend lowering:** the exact typed value or opaque semantic JSON reaches the intended engine
   option.
5. **Response projection:** standard, framework-compatible, and `nvext` locations are correct for
   streaming and non-streaming responses.
6. **Negative capability:** unsupported backends and deployment modes fail closed.
7. **End to end:** compare representative requests and responses from `vllm serve`, the SGLang
   server, and `trtllm-serve` with the corresponding Dynamo-mediated profiles and assert semantic
   parity.

The compatibility profile should generate a matrix test so adding a backend or transport cannot
silently narrow support. Chat-completions coverage is required before the corresponding behavior is
considered complete. Completions coverage is required only for behavior Dynamo declares compatible
with the native server. Each claim in a per-target compatibility reference must link to conformance
evidence or be marked `Unverified`. The drift-detection mechanism itself needs fixtures that prove
it detects representative field additions, removals, type/default changes, and response-shape
changes without treating formatting-only changes as protocol drift.

## Definition of done

The requirements below describe initial delivery unless explicitly labeled as the full operating
model. Drift automation is staged: vLLM is the initial working integration, while recurring checks
for all three target servers remain the completed operating model.

- A documented field-classification policy is implemented at the public and internal boundaries.
- Versioned target profiles exist for `vllm serve`, the SGLang server, and `trtllm-serve`.
- Each profile names the exact target-server versions or commits supported by the corresponding
  Dynamo release; all other versions are marked `Unverified`.
- `/v1/chat/completions` has conformance coverage for every behavior declared as supported by each
  target profile.
- `/v1/completions` preserves the declared native-server compatibility subset without becoming an
  independent feature-expansion requirement.
- Known semantic fields cannot enter a silent-drop path.
- A versioned, namespaced backend-extension envelope is available where opaque passthrough is
  justified.
- HTTP and native transports pass the same conformance suite for their declared profiles.
- N-2 mixed-version tests cover every new internal field and compatibility shim.
- Error responses identify unsupported fields and the selected compatibility profile.
- Metrics expose rejects, legacy fallbacks, and attempted drops without recording payload contents.
- The user protocol guide owns the shared declared contract, including top-level fields, backend
  extensions, and `nvext.extra_fields` as distinct mechanisms, without duplicating target-specific
  deltas.
- Separate compatibility references exist for `vllm serve`, the SGLang server, and
  `trtllm-serve`; every known difference is classified, explained, version-scoped, and connected to
  test evidence and either an issue when follow-up work remains or an explicit no-action decision.
- Each target profile records its last verified upstream version or commit and verification date.
- The initial `vllm serve` integration runs on vLLM version bumps and produces reviewable drift
  artifacts with documented ownership, cadence, triage expectations, and issue escalation.
- A checked-in example generated by the working vLLM mechanism records full previous and candidate
  upstream commit SHAs, the exact Dynamo commit SHA, inputs, evidence, and classified results.
- **Full operating model:** recurring and version-change-triggered drift checks produce reviewable
  artifacts for vLLM, SGLang, and TensorRT-LLM.
- The developer protocol-change guide defines the required classification, implementation,
  compatibility, documentation, and conformance-test procedure.
- CI detects drift among machine-readable compatibility profiles, generated schemas, conformance
  tests, and generated protocol documentation where those artifacts can be derived mechanically.
- The incidents in Appendix A have regression coverage at the boundary where each field was lost.

## Implementation plan

### Phase 1: inventory and policy

- Inventory existing top-level fields, passthrough maps, environment-controlled ignore behavior,
  and `nvext` fields.
- Inventory the protocol contract currently distributed across frontend configuration, runtime
  OpenAPI, `nvext`, compatibility, and endpoint-specific documentation.
- Inventory known parity differences for each target server and classify them as upstream drift,
  Dynamo gaps, intentional divergences, unsupported behavior, or unverified behavior.
- Classify each field using the placement table.
- Publish the initial capability matrix for `vllm serve`, the SGLang server, and `trtllm-serve`,
  split by chat-completions and completions endpoints, and name owners for each target profile.

### Phase 2: fail-closed admission

- Replace silent drops for known semantic fields with targeted capability errors.
- Add conflict and precedence validation for top-level and legacy representations.
- Add metrics for rejected and legacy fields.

### Phase 3: internal extension envelope

- Introduce the versioned namespaced envelope.
- Migrate existing ad hoc passthrough paths without changing their public wire contracts.
- Add N-2 reader/writer fixtures and removal metadata for shims.

### Phase 4: conformance and promotion

- Generate cross-backend contract tests from compatibility profiles.
- Compare selected behavior with the three target native-server APIs, prioritizing
  `/v1/chat/completions` and using `/v1/completions` only for the declared compatibility subset.
- Publish the user protocol guide, the three per-target compatibility references, and the developer
  protocol-change guide; cross-link the existing resources and add mechanical drift checks for
  generated artifacts.
- Promote widely supported extension fields to typed canonical fields when their semantics stabilize.

### Phase 5: continuous upstream alignment

- Establish a reproducible protocol baseline for each supported target-server version.
- Implement declaration-level diffs plus runtime probes for behavior not established by schemas.
- Integrate the first working check with the vLLM version-bump workflow so a bump report is a
  required review artifact.
- Check in a real example report produced by the vLLM mechanism with exact previous and candidate
  upstream commit SHAs and the exact Dynamo commit SHA used for the run.
- Publish each result as a reviewable artifact and route material differences to the target-profile
  owner for classification and tracking.
- Update compatibility profiles, references, schemas, and tests when a difference is resolved, and
  retain the rationale for intentional divergences and unsupported behavior.
- Generalize the mechanism to SGLang and TensorRT-LLM, then run it for all three targets on the
  documented schedule and whenever their supported upstream pin changes.

## Alternatives considered

### Accept every unknown top-level field

This maximizes short-term compatibility but loses schema quality and creates collision, security,
routing, and caching risks. It also cannot guarantee that a backend actually consumed the value.

### Add every framework field as a Dynamo typed field immediately

This is safe for mature cross-backend semantics but does not scale with rapidly evolving backend
APIs. It would couple frontend releases to every experimental engine option.

### Put every non-OpenAI field in `nvext`

This keeps the canonical schema small but breaks drop-in framework compatibility and forces existing
SDK clients to learn a Dynamo-specific request and response shape. `nvext` should remain the home of
Dynamo-owned behavior, not a catch-all for established framework contracts.

## Open questions

- Which framework-version source should populate compatibility profiles at runtime?
- Should experimental extension admission require an explicit server-side allowlist, a client-side
  opt-in, or both?
- How should an opaque extension participate in routing fingerprints when Dynamo cannot interpret
  its values?
- Which profiles should be exposed through discovery APIs versus static documentation?
- What is the deprecation period for fields duplicated between top-level and `nvext` locations?

## Appendix A: observed compatibility incidents

### A.1 Root-level `thinking_token_budget`

**Occurrence.** The vLLM-compatible control existed through legacy
`nvext.max_thinking_tokens`, but clients expected the framework-facing
`thinking_token_budget` field at the request root. Chat completions and Responses did not share one
canonical typed representation.

**Resolution.** [PR #12624](https://github.com/ai-dynamo/dynamo/pull/12624), merged on 2026-09-19,
added typed root-level support to both endpoints, forwarded it to the backend sampling parameter,
and retained the legacy `nvext` fallback. The root field takes precedence when both are supplied.

**Lesson.** A stable framework-compatible field belongs in the typed public contract; `nvext` can
provide a bounded migration path but should not be its only representation.

### A.2 `continue_final_message` in the Rust chat preprocessor

**Occurrence.** The Python frontend honored `continue_final_message` and
`add_generation_prompt`, while the Rust preprocessor ignored them. Requests succeeded, but an
assistant prefix was closed and the model began a new turn.

**Resolution.** [PR #13841](https://github.com/ai-dynamo/dynamo/pull/13841), merged on 2026-08-28,
added the fields to the Rust request model, rejected incompatible combinations, required a final
assistant message for continuation, and rendered the prompt with the intended continuation
semantics.

**Lesson.** Accepting a field is insufficient; every frontend implementation and preprocessing path
must apply the same meaning.

### A.3 Media `extra_body` passthrough

**Occurrence.** OpenAI clients merge `extra_body` values into the request root, but Dynamo's image,
video, and speech request structs silently discarded unknown fields. Backend-specific controls such
as image sizing, audio-generation toggles, and speech emotion could not reach the worker.

**Resolution.** [PR #13817](https://github.com/ai-dynamo/dynamo/pull/13817), merged on 2026-09-03,
captured unknown public-edge fields, moved them under the explicit internal
`extra_args.media_passthrough` namespace, and taught the media handlers to apply recognized sampling
parameters or pass the remainder to engine-specific arguments.

**Lesson.** Controlled edge capture plus internal namespacing can support a fast-moving framework
surface without allowing extensions to collide with canonical protocol fields.

### A.4 Chat `prompt_logprobs` response placement

**Occurrence.** The top-level request field reached vLLM and the worker computed the data, but the
Rust frontend returned it only when the client separately requested
`nvext.extra_fields: ["prompt_logprobs"]`. Otherwise the request returned HTTP 200 without the
expected root response field.

**Resolution.** [PR #13957](https://github.com/ai-dynamo/dynamo/pull/13957), open as of 2026-09-30,
adds optional root-level prompt log probabilities to non-streaming chat responses, preserves the
payload through internal stream aggregation, prevents it from leaking into SSE chunks, and keeps the
legacy `nvext` projection.

**Lesson.** Response placement is part of compatibility. Computing the correct value does not help
framework clients if Dynamo exposes it only through a private extension.

### A.5 Completion `multi_modal_data`

**Occurrence.** A custom vLLM modality payload sent as an unknown field was rejected by default or
dropped when unsupported-field ignoring was enabled. Neither outcome allowed the backend's
registered multimodal processor to receive the payload, and silent dropping could yield a plausible
but incorrect text-only result.

**Proposed resolution.** [PR #15144](https://github.com/ai-dynamo/dynamo/pull/15144), open as of
2026-09-30, adds a typed JSON-safe request map and a separately named frontend-to-worker field. The
vLLM adapter installs it for supported modes, while unsupported backends and handoff modes reject it.

**Lesson.** Opaque data can be supported safely when ownership is explicit and every path that
cannot preserve it fails closed.

## Appendix B: `nvext` placement consensus and recommendation

### B.1 Status of the decision

The available design documents, implementation history, and internal discussions do not establish
a ratified rule that either removes `nvext` or places every non-OpenAI field inside it. They instead
show a durable middle ground:

> Use top-level fields for OpenAI or upstream-server compatibility. Keep `nvext` for Dynamo-owned
> behavior and metadata. Do not maintain the same semantic field permanently in both places.

An earlier Dynamo-NIM API-alignment proposal sought to move all `nvext` fields to the root and
eventually remove the envelope. The corresponding discussion subsequently narrowed that plan:

- move sampling, guided-decoding, and other framework-facing fields to the request root;
- allow a bounded compatibility period when an existing client still uses the legacy `nvext`
  location;
- make the top-level value authoritative during that transition; and
- decide the fate of remaining Dynamo-specific `nvext` fields separately.

The full removal did not become the Dynamo contract. Current code and documentation instead use
`nvext` for routing, preprocessing, scheduling, cache isolation, and opt-in response metadata. At
the same time, [`CommonExt`](../src/protocols/openai/common_ext.rs) flattens commonly supported
framework extensions such as `top_k`, `min_p`, `prompt_logprobs`, and
`continue_final_message` into the public request root.

More recent compatibility discussions reinforce this division: callers should be able to send the
same top-level fields accepted by the selected vLLM, SGLang, or TensorRT-LLM server profile, while
Dynamo may organize and validate those fields in framework-specific internal structures. Mapping
must remain intentional; accepting a field does not establish support unless all declared semantic
owners handle it with the documented behavior.

### B.2 What `extra_body` changes—and what it does not

`extra_body` is an OpenAI client SDK argument for adding properties not declared by the typed SDK
method. The SDK merges those properties into the top-level JSON request. It does not create a
standard wire-level `extra_body` extension namespace.

For example:

```python
client.chat.completions.create(
    model="example-model",
    messages=[...],
    extra_body={
        "top_k": 20,
        "nvext": {"agent_hints": {"priority": 5}},
    },
)
```

produces a request shaped like:

```json
{
  "model": "example-model",
  "messages": [],
  "top_k": 20,
  "nvext": {
    "agent_hints": {
      "priority": 5
    }
  }
}
```

The server must still classify, validate, and lower `top_k` and `nvext`. Consequently,
`extra_body` improves client ergonomics but does not answer whether a field belongs at the root, in
`nvext`, or in a backend-specific namespace.

### B.3 Current placement consensus

The strongest points of agreement are:

1. **Framework compatibility belongs at the root.** If a declared vLLM, SGLang, or TensorRT-LLM
   compatibility profile exposes a field at the root, placing it only under `nvext` breaks drop-in
   clients, benchmarking tools, and framework-native request builders.
2. **Dynamo ownership belongs in `nvext`.** Routing controls, scheduling hints, cache and
   preprocessing controls owned by Dynamo, and optional Dynamo-produced response metadata benefit
   from an explicit collision-resistant namespace.
3. **Duplicates are transitional.** A field may be read from both locations during migration, but
   the root is authoritative, conflicts must be diagnosed, and the legacy location must have a
   removal plan.
4. **Response placement is part of compatibility.** If an upstream server returns a value at the
   root, Dynamo should not require an unrelated `nvext.extra_fields` opt-in to obtain it.
5. **Unknown acceptance is not passthrough.** Ignoring an unsupported field and returning HTTP 200
   is not equivalent to forwarding it. It can silently change model behavior.

The unresolved area is arbitrary backend passthrough. There is no demonstrated consensus that all
unknown root fields should automatically reach whichever backend happens to receive the request.
That behavior would be especially ambiguous in a multi-backend deployment or across a sidecar
transport that needs an explicit representation for the field.

### B.4 Recommended field-placement rule

| Field category | Public wire location |
|---|---|
| OpenAI-standard field | Typed top-level field |
| Field exposed at the root by the selected vLLM, SGLang, or TensorRT-LLM compatibility profile | Top-level field; type it when stable, or admit it through a profile-scoped capture rule when its payload remains opaque to Dynamo |
| Stable field shared by multiple serving frameworks | Typed top-level field, normally through `CommonExt` |
| Dynamo routing, scheduling, preprocessing, or cache control | Typed `nvext` field, or an `x-dynamo-*` header when supplied by a trusted gateway |
| Optional Dynamo-generated response metadata | Response `nvext`, selected through `nvext.extra_fields` |
| Opaque backend response metadata | `nvext.engine_data`, including backend identity and a schema version when pools can be heterogeneous |
| Experimental request parameter owned by one backend | Versioned `backend_extensions.<framework>` envelope or a declared backend-specific compatibility profile |
| Completely unknown field | Reject by default; do not silently drop or broadcast it |

Keeping `nvext` is preferable to renaming it solely for aesthetics. It is already a public contract,
and renaming it to `dynext` would create migration cost without resolving field ownership. The
documentation should nevertheless describe it consistently as the Dynamo/NVIDIA platform
extension namespace rather than a general bucket for anything absent from the OpenAI schema.

### B.5 Implications for current compatibility work

- **`prompt_logprobs`:** Keep the request at the root and expose the established root-level
  non-streaming response. A legacy `nvext.prompt_logprobs` projection may remain during a bounded
  compatibility window. The prompt-sized payload should remain absent from ordinary SSE chunks.
- **`thinking_token_budget` and `continue_final_message`:** Keep them at the root because they are
  upstream-facing request semantics rather than Dynamo controls.
- **`multi_modal_data`:** Put it at the root only for a compatibility profile whose native server
  exposes that exact field and whose Dynamo path preserves its semantics. A single custom
  processor's ability to consume the value is not sufficient to make it universal.
- **Media generation arguments:** Capture only through an explicit typed or namespaced extension
  path. Do not treat general unknown-field acceptance as backend delivery.
- **Unsupported-field handling:** Replace global accept-and-drop behavior with profile-aware
  admission. Every admitted semantic field must reach all of its declared semantic owners;
  unsupported paths must reject it before dispatch.

The practical review question for every new field is therefore:

> Does the field earn a root-level location through protocol compatibility, or an `nvext` location
> through Dynamo ownership?

A backend's ability to consume an arbitrary value, by itself, satisfies neither condition. This
question decides public placement only; Appendix C defines the additional semantic-ownership and
transport questions required for complete support.

## Appendix C: Field placement, semantic ownership, and transport

The goal "define one placement policy for standard fields, framework-native fields, opaque backend
extensions, and Dynamo-only metadata" is shorthand. Those terms do not form four mutually
exclusive classes. In particular, **framework-native** describes where a field's public contract
comes from, while **opaque** describes how Dynamo represents or transports the value. A
framework-native field can therefore be opaque to Dynamo.

### C.1 Three independent questions

Every field should be specified along three dimensions.

1. **Public ownership and location:** Is the field part of the OpenAI contract, exposed by a target
   native server, owned by Dynamo, or unsupported? This determines whether it appears at the
   request or response root, under `nvext`, or nowhere.
2. **Semantic ownership:** Which component must behave differently because of the field?
   - **Dynamo-handled:** Dynamo changes preprocessing, routing, aggregation, streaming, validation,
     or response serialization.
   - **Backend-handled:** The engine implements the functional behavior; Dynamo preserves and
     delivers the value.
   - **Jointly handled:** Dynamo and the backend implement distinct parts of the end-to-end
     behavior.
3. **Internal representation and transport:** Is the value carried as a canonical typed field, a
   typed backend-specific field, a namespaced opaque value, a translated backend parameter, or not
   forwarded because Dynamo handles it completely?

These dimensions must not be collapsed. Top-level public placement does not imply that Dynamo owns
the semantics. A typed representation does not imply that the frontend uses the value itself. An
opaque representation does not require a Dynamo-specific public namespace when the target native
server exposes the field at the root.

### C.2 What it means for Dynamo to recognize a field

A field is recognized when Dynamo makes an explicit compatibility commitment for it. Recognition
means that Dynamo:

- accepts the field at its documented wire location and preserves meaningful values such as
  `false`, `0`, `null`, and empty collections;
- identifies the endpoints, target servers, versions, transports, and deployment modes that
  support it;
- delivers it to every declared semantic owner in the required representation;
- preserves and exposes resulting response data at the documented location; and
- rejects unsupported paths instead of silently discarding the field.

Recognition does not, by itself, mean that the Dynamo frontend implements the field's functional
semantics. A recognized field may be Dynamo-handled, backend-handled, or jointly handled.

### C.3 Examples

| Field | Public ownership and location | Semantic ownership | Internal representation |
|---|---|---|---|
| `continue_final_message` | Target-server-compatible top-level field | Dynamo-handled: changes chat-template preprocessing | Typed; the original field does not need to reach the engine after preprocessing |
| `temperature` | OpenAI-compatible top-level field | Backend-handled: changes engine sampling | Typed and lowered to the backend sampling parameters |
| `prompt_logprobs` | Target-server-compatible top-level request and response field | Jointly handled: the backend computes the values; Dynamo preserves, aggregates, and serializes them according to the streaming contract | Typed across the supported request, transport, and response paths |
| Custom processor option | Target-server or plugin-specific top-level field, admitted only by a declared profile | Normally backend-handled | Namespaced opaque value such as `backend_extensions.vllm`; it may later be promoted to a typed field |
| `nvext.extra_fields` | Dynamo-owned field under `nvext` | Dynamo-handled: selects optional Dynamo response metadata | Typed Dynamo extension |

For jointly handled fields, the specification must divide responsibility explicitly. Both
components must not independently apply the same transformation. For example, with
`prompt_logprobs`, the backend computes the payload while Dynamo owns aggregation and public
response placement.

### C.4 Placement consequences

- Use the public root when OpenAI or the selected native-server profile defines the field there,
  regardless of whether Dynamo carries it internally as a typed or opaque value.
- Use `nvext` for behavior or metadata owned by Dynamo, not as the default home for every field
  absent from the OpenAI schema.
- Choose the internal representation according to semantic ownership, validation needs, stability,
  and collision risk. Shared or Dynamo-handled semantics usually justify a typed field;
  profile-specific payloads that Dynamo does not interpret may remain namespaced and opaque.
- Reject completely unknown fields by default. Root-level admission must be tied to a declared
  target profile rather than becoming unrestricted passthrough.
- Promote an opaque field to a typed representation when its schema and semantics stabilize and
  Dynamo can validate and test its end-to-end contract.

### C.5 Per-field specification template

Compatibility profiles should record the dimensions directly. The following YAML is illustrative,
non-normative pseudocode; it does not require YAML files or runtime YAML loading:

```yaml
prompt_logprobs:
  public_location: top_level
  semantic_owners:
    - dynamo
    - vllm
  dynamo_behavior:
    - validate_request
    - aggregate_result
    - serialize_non_streaming_response
  backend_behavior:
    - compute_prompt_logprobs
  internal_transport: typed
  streaming_behavior: omit_from_ordinary_chunks
  unsupported_behavior: reject
```

The general review question is therefore not merely "typed or opaque?" It is:

> Where does the public contract place this field, who owns each part of its semantics, how does
> Dynamo transport it, and which profiles can preserve that complete behavior?

## Appendix D: Current frontend compatibility mechanism

Dynamo does not normally forward an OpenAI request to `vllm serve`, the SGLang server, or
`trtllm-serve`. It provides its own HTTP frontend, converts the request into Dynamo's canonical
frontend-to-worker protocol, and calls an engine adapter directly. Compatibility with a native
server is therefore the combined result of several Dynamo-owned layers rather than a property of
one protocol model or proxy boundary.

### D.1 Current request-to-response path

```text
Client JSON
  -> public request model and validation
  -> chat rendering, tokenization, and request normalization
  -> Dynamo PreprocessedRequest
  -> routing and frontend-to-worker transport
  -> vLLM, SGLang, or TensorRT-LLM adapter
  -> LLMEngineOutput / BackendOutput
  -> streaming delta or non-streaming aggregation
  -> client-facing OpenAI-shaped response
```

The layers currently divide responsibility as follows:

1. **Public request admission.** The chat and completion request wrappers combine the shared
   OpenAI-shaped protocol model with root-level [`CommonExt`](../src/protocols/openai/common_ext.rs),
   an optional typed [`nvext`](../src/protocols/common/extensions.rs), and several explicitly
   declared request fields. A flattened `unsupported_fields` map captures anything not represented
   by those models. The concrete chat wrapper is
   [`NvCreateChatCompletionRequest`](../src/protocols/openai/chat_completions.rs); completions use
   the analogous wrapper.
2. **Validation and normalization.** The frontend validates field types, combinations, endpoint
   restrictions, and unsupported fields before dispatch. Unknown fields are rejected by default.
   A small root-level pass-through allowlist is recognized explicitly, while
   `DYN_IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS` can make other unknown fields be accepted and dropped.
   These policies currently live in [`validate.rs`](../src/protocols/openai/validate.rs). Some
   values are also normalized at this boundary; for example, chat reasoning controls are resolved
   into one set of chat-template arguments.
3. **Chat processing and preprocessing.** Every chat request is rendered and tokenized before it
   reaches a worker. The default `dynamo` chat processor implements framework-independent behavior
   in Rust. The `vllm` and `sglang` processor modes reuse those frameworks' local libraries for
   rendering and parsing when Dynamo lacks a model-specific implementation; they do not send the
   request to the frameworks' HTTP servers. The current options are documented in
   [Chat processors](../../../docs/fern/pages/use-cases/tool-calling-and-reasoning/chat-processors.mdx).
4. **Canonical internal request and transport.** Preprocessing converts the admitted public
   request into [`PreprocessedRequest`](../src/protocols/common/preprocessor.rs), which carries
   token IDs or embeddings, multimodal data, sampling and output options, stop conditions, routing
   data, and selected extension payloads. This serialized type is the frontend-to-worker contract
   covered by Dynamo's N-2 mixed-version compatibility policy. Fields consumed only by the
   frontend are not necessarily transported; fields needed by a worker must be represented here
   or in a deliberately opaque payload.
5. **Backend lowering.** The selected vLLM, SGLang, or TensorRT-LLM adapter translates the canonical
   request into that engine's native in-process request or sampling objects. Each adapter decides
   which canonical fields it can honor and performs framework-specific conversion. Consequently,
   accepting and transporting a field is necessary but is not sufficient to establish native-
   server parity.
6. **Output normalization and public response projection.** Engine adapters emit common
   [`LLMEngineOutput` and `BackendOutput`](../src/protocols/common/llm_backend.rs) values. Dynamo's
   response builders then map those values into chat or completion streaming deltas, or aggregate
   the stream into a non-streaming response. Standard response fields, Dynamo `nvext` data, and
   opaque engine data take different projection paths, so a value produced by an engine is not
   client-visible unless the appropriate response path preserves and serializes it.

Compatibility today is therefore partly declarative and partly behavioral. Rust and Python types
describe many accepted fields and internal payloads, but defaults, validation order, chat-template
behavior, backend support, aggregation, and streaming placement are implemented in executable
code. The runtime OpenAPI schema describes much of the public shape; it does not by itself prove
that Dynamo matches any particular version of a native server end to end.

### D.2 What the current mechanism already provides

- A typed OpenAI-shaped request and response surface for the two in-scope endpoints.
- Central validation that rejects most unknown fields instead of silently sending arbitrary data
  into a worker.
- A common preprocessed request and normalized engine output that let one frontend serve multiple
  backends without exposing every backend's internal API directly.
- Selective typed and opaque extension paths for capabilities outside the standard OpenAI schema.
- An N-2 compatibility policy for the Dynamo frontend-to-worker wire contract.
- Framework-local chat processor fallbacks that can reuse vLLM or SGLang rendering and parser
  behavior while retaining Dynamo routing and serving architecture.

These are useful building blocks, but none of them is a versioned claim that the complete Dynamo
HTTP behavior matches a particular `vllm serve`, SGLang server, or `trtllm-serve` release.

### D.3 Missing pieces addressed by this proposal

1. **No version-scoped compatibility contract.** Support is distributed across protocol structs,
   validation functions, processor implementations, backend adapters, tests, and documentation.
   There is no single profile that says which field and behavior Dynamo supports for a specific
   target-server version, endpoint, processor mode, and streaming mode.
2. **No enforced end-to-end field lifecycle.** Adding a field currently requires coordinated
   edits at every layer, but the repository does not mechanically require proof that admission,
   normalization, transport, backend lowering, and response projection all agree. A field can be
   accepted at one layer and lost or interpreted differently at another.
3. **Extension ownership and placement are inconsistent.** Typed root fields, `nvext`, the
   unsupported-field pass-through allowlist, and generic opaque payloads evolved for different
   cases. They do not yet implement one policy that separates a public framework-compatible field,
   a Dynamo-owned control, and a backend-opaque extension.
4. **Unknown-field handling is not target-aware.** The current choice is primarily global: reject
   an unknown field, explicitly allowlist it, or configure the frontend to accept and drop it. It
   is not derived from the selected target profile and cannot explain that a field belongs to a
   newer upstream version or only to another backend.
5. **Processor parity is not one uniform behavior.** The default Dynamo processor and the vLLM or
   SGLang fallback processors may render templates or parse tools and reasoning differently.
   TensorRT-LLM does not yet have an equivalent fallback path. These variations are useful, but
   their compatibility boundaries are not captured systematically.
6. **Response compatibility can drift independently from requests.** An engine may compute a
   value that is omitted, placed only under `nvext`, repeated in streaming chunks, or lost during
   non-streaming aggregation. Request-schema parity cannot detect these response-placement and
   lifecycle differences.
7. **Internal wire compatibility is not native-server compatibility.** The N-2 guarantee protects
   supported Dynamo frontends and workers from each other. It does not establish parity with the
   public API, defaults, validation, or streaming behavior of an upstream server version.
8. **Documentation and OpenAPI do not expose the full behavioral contract.** They can show an
   accepted field shape, but not all target-specific defaults, deliberate divergences, unsupported
   combinations, processor dependencies, or whether parity was verified by runtime observation.
9. **Upstream changes are detected reactively.** Framework version bumps do not yet produce a
   required protocol diff, runtime probe result, compatibility decision, and documentation update.

This proposal keeps the current execution pipeline. It adds a contract and verification layer
around it: versioned target profiles, one field-placement and ownership policy, target-aware
validation, explicit opaque-extension rules, end-to-end conformance tests, per-target compatibility
guides, and a framework-version-bump drift check. Together, those additions turn compatibility
from an emergent property of several implementations into a maintained and reviewable invariant.

## References

- [Dynamo NVIDIA request extensions (`nvext`)](../../../docs/fern/pages/developer-guide/additional-resources/nvidia-request-extensions-nvext.md)
- [Issue #13941: chat prompt logprobs response mismatch](https://github.com/ai-dynamo/dynamo/issues/13941)
- [PR #12624: root-level thinking token budget](https://github.com/ai-dynamo/dynamo/pull/12624)
- [PR #13817: media `extra_body` passthrough](https://github.com/ai-dynamo/dynamo/pull/13817)
- [PR #13841: honor `continue_final_message`](https://github.com/ai-dynamo/dynamo/pull/13841)
- [PR #13957: return chat prompt logprobs](https://github.com/ai-dynamo/dynamo/pull/13957)
- [PR #15144: forward backend multimodal data](https://github.com/ai-dynamo/dynamo/pull/15144)
- [PR #2380: add root-level common extensions](https://github.com/ai-dynamo/dynamo/pull/2380)
- [PR #2404: add structured-output parameters at the root](https://github.com/ai-dynamo/dynamo/pull/2404)
- [PR #4372: add opt-in `nvext` response metadata](https://github.com/ai-dynamo/dynamo/pull/4372)
- [PR #8099: add opaque `nvext.engine_data`](https://github.com/ai-dynamo/dynamo/pull/8099)
- [OpenAI Python SDK: undocumented request parameters](https://github.com/openai/openai-python#undocumented-request-params)
