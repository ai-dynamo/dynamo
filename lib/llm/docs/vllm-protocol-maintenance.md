<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Maintaining vLLM protocol compatibility

The protocol inventory answers **which native-server fields Dynamo must recognize
as meaningful**, not which fields Dynamo supports. A declaration, an unchanged
source file, or a green inventory check is not evidence of runtime parity.

This guide covers runtime compatibility maintenance. The
[consolidated assessment guide](https://github.com/ai-dynamo/dynamo/blob/codex/vllm-protocol-tooling/lib/llm/docs/dynamo-vllm-protocol-assessment.md)
owns source-assessment commands, decisions, gates and CI operation. The complete
contract, backend transport, response projection, and conformance requirements
remain in the [compatibility design](https://github.com/ai-dynamo/dynamo/blob/da45b41777ec507974d3a6a41053a9e908e530d3/lib/llm/docs/frontend-protocol-framework-compatibility.md).
The review branches implement a bounded subset; complete profile/schema alignment,
scheduled CI execution and broad native conformance remain continuation work.
For client-facing behavior and known differences, see the
[vLLM serve compatibility reference](vllm-serve-compatibility.md). Its claims are
scoped to the recorded server and Dynamo revisions; inventory membership alone
must not be presented as compatibility.

## Sources of truth

- [`container/context.yaml`](../../../container/context.yaml) selects configured
  framework image versions. Check every platform, not just CUDA.
- [`vllm_pins.json`](../src/protocols/openai/compatibility/vllm_pins.json) maps those
  versions to immutable source commits and platforms. It does not select a second
  set of framework versions.
- [`vllm_inventory.json`](../src/protocols/openai/compatibility/vllm_inventory.json)
  is generated from those commits. It records types, defaults, declaring classes,
  source paths, and statically resolvable input aliases for the two endpoints.
- [`vllm_fields.rs`](../src/protocols/openai/compatibility/vllm_fields.rs) is the
  derived bounded vocabulary used by Rust admission. Do not edit either generated
  file; change the extractor or pins and regenerate.

Admission checks this vocabulary only after typed parsing and explicit passthrough
classification. An unhandled known field is rejected even when
`DYN_IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS=1`. This includes values such as `null`,
`false`, `0`, and empty collections. A field from the other in-scope endpoint is
also meaningful: sending a chat-only directive to completions does not make it a
harmless unknown. This inventory must never be used as permission to forward all
native fields to a worker.

## Shared and endpoint-owned request fields

`CommonExt` is a Rust composition helper for non-standard fields shared by chat
and text completion requests. Its fields are flattened into the public request
root; it is not a separate JSON namespace or a promise of universal backend
support. Put endpoint-specific fields directly on their endpoint request type.

`add_generation_prompt` and `continue_final_message` now belong to
`NvCreateChatCompletionRequest`. Their public names and root-level JSON placement
are unchanged. Rust prompt rendering and the internal unified request read these
chat-owned values. On `/v1/completions`, either key is rejected by presence,
including `null` and `false`, before the generic unsupported-field ignore policy.
That is a Dynamo fail-closed rule, not a claim that native vLLM rejects the same
misplaced fields.

When changing ownership, check all of these boundaries:

- Request deserialization and serialization keep the public wire shape stable.
- Internal conversion and the actual renderer/processor retain the value.
- The generated OpenAPI **endpoint request schema**, following component
  references, contains the field only where it is supported.
- The other endpoint rejects meaningful misplaced controls without echoing their
  values or allowing migration ignore mode to erase them.

The `test_common_ext` integration tests cover root JSON and derived schemas;
the `chat_only` Rust unit-test filter covers unified conversion, completion error
mapping, and inline endpoint schemas with shared OpenAPI component references. The latter uses the same
generator as `/openapi.json`; it is not a standalone live HTTP fetch. Rendering
and native-server comparisons are separate evidence, not implied by schema tests.

### Configurable route paths and schema identity

Chat/completion router constructors attach their logical endpoint to `RouteDoc`
using `with_completion_endpoint`. The OpenAPI generator uses that identity for
the POST request schema, error schema, summary, and description, while retaining
the actual configured path and its operation ID. Do not infer chat versus text
completion from a URL suffix: a text-completion handler can be mounted at a path
whose name ends in `chat/completions`.

When adding or changing one of these routes, preserve the constructor's endpoint
metadata and test the served `/openapi.json` against the actual handler. Plain
`RouteDoc::new` entries and other HTTP methods do not acquire completion schemas
merely by using a familiar path. Route equality and hashing deliberately remain
method/path-only so documentation metadata cannot bypass duplicate-registration
checks. This metadata describes the HTTP handler, not the selected backend or
the completeness of its compatibility profile.

The custom-route regression fetches the served schema and exercises invalid-count
responses at default and custom paths. It does not establish complete base-request
schemas, all error categories, or a native-server conformance claim.

## Framework-version bump procedure

The runtime admission wrapper in
[`admission.rs`](../src/protocols/openai/compatibility/admission.rs) is distinct
from the generated vocabulary. It currently selects only one inspected rule
for exact vLLM 0.29.0/0.30.0 releases, using the selected WorkerSet's advertised
engine version. It is not a full support profile. A version bump must review
these selectors and their source-pin test as well as regenerate the inventory;
adding a release to the inventory alone must not grant runtime compatibility.
Unknown/local releases remain Unverified. Missing engine-version capability
retains the legacy unidentified path, not an implicit native-server match.

The wrapper now derives a serializable admission descriptor from
[`profile.rs`](../src/protocols/openai/compatibility/profile.rs). Its rule is the
same rule used for rejection, not a separately authored compatibility allowlist.
The endpoint comes from the request type, and the selected pipeline provides
processor, internal RPC representation, and advertised worker role. This makes
error attribution precise without changing model-card checksums or writing
frontend policy into engine runtime metadata.

Do not infer more than those facts:

- Token-input completions use the Rust processor even when chat uses a factory.
- The built-in factory registration explicitly identifies `vllm` or `sglang`;
  `external_factory` means a custom callback. This is frontend-owned registration
  metadata, not an inference from a callback name, worker card, or environment.
  Identity alone grants no capabilities or conformance claims.
- Text-input paths use backend preprocessing and OpenAI-shaped RPC; they do not
  inherit token-path verification.
- Missing worker-role metadata means unknown deployment topology for compatibility
  purposes, even where legacy routing uses an aggregated fallback.
- The existing whole RPC message is unversioned. The extension envelope's schema
  version does not version that message or identify TCP versus another request
  plane. Those dimensions need separate authoritative facts and evidence.
- A native request-validator rule can apply across downstream paths without
  establishing end-to-end support on any of them. In particular, rejecting positive
  streaming prompt counts does not prove unary prompt-logprob preservation.

The descriptor is exposed through the
[registered pipeline catalog](#registered-pipeline-catalog), not a complete
per-field conformance catalog. It does not replace the remaining work to connect
all field locations, lowering, fallback policies, and runtime evidence to profiles.

For the source-assessment steps, follow the
[framework-version bump procedure](https://github.com/ai-dynamo/dynamo/blob/codex/vllm-protocol-tooling/lib/llm/docs/dynamo-vllm-protocol-assessment.md#framework-version-bump-or-dynamo-only-change).
It compares Dynamo directly with each selected native revision, including existing
gaps, and retains selected upstream behavioral-change signals. One assessment
registry records scoped findings and decisions; the experimental per-upstream-pair
decision schema and `--require-triage` CLI are retired.

Run the commands from the consolidated B1 tooling checkout. While these PRs remain
stacked, use `--dynamo-repo /path/to/runtime-checkout` and immutable commits to
assess this branch's runtime code; an older checkout may still contain retired
scripts. This does not require rebasing the runtime PRs merely to run assessment.

Review `report.md`, not a full generated inventory diff. A version bump must update
the runtime image configuration, immutable source pins and compact vocabulary,
review affected runtime selectors, disposition new or changed findings, and run
targeted native-server/Dynamo probes for behavior that source analysis cannot
establish. An accepted static decision is not proof of runtime parity or release
approval. Generated inventories remain optional diagnostic artifacts.

### Report provenance

The consolidated report records exact Dynamo/native revisions, selected scope,
input hashes and hashes of the nested implementation modules. Tool hashes follow
the package structure automatically, excluding tests. Historical source snapshots,
reports and evidence remain attributable to their original revisions; do not
reinterpret experimental upstream-only decisions as current direct-assessment
approvals. See the assessment guide for decision applicability and invalidation.

## Release-pinned adapter boundary checks

The generated fixtures under
[`protocol_releases`](../../../components/src/dynamo/vllm/tests/fixtures/protocol_releases)
retain exact sampling-builder and logprob-parser excerpts from Dynamo 1.4.0 and
1.5.0, their source hashes, and each release's configured CUDA vLLM image tag.
They also retain the Rust preprocessor's literal legacy passthrough vocabulary.
Regenerate from the pinned Git objects, never by editing the JSON:

```sh
python -m scripts.protocol_compatibility generate-release-fixtures --repo /path/to/runtime-checkout
python -m scripts.protocol_compatibility generate-release-fixtures --repo /path/to/runtime-checkout --check
```

The generator supports `--check` for fixture freshness. Runtime checks are a
separate matrix: run
`components/src/dynamo/vllm/tests/test_vllm_protocol_release_boundary.py` with
vLLM 0.26.0 (Dynamo 1.4), 0.28.0 (Dynamo 1.5), and the current 0.30.0 baseline.
Use immutable image digests in retained evidence. The tests deliberately skip
nonmatching engine versions; one green invocation does not cover all releases.
Repository pytest configuration also requires the benchmark, asyncio, and timeout
plugins. No model, GPU, or historical Dynamo binding is required by these tests.

The older-reader checks execute trusted, checked-in function excerpts with real,
version-matched `SamplingParams`. The current-reader check replays fields from the
old writer's extracted vocabulary through the production extension applicator;
it does not execute the old Rust writer or the complete current worker. The
fixture generator does not execute historical code. Removal follows the N-2
window: remove 1.4 coverage when it leaves that window, then 1.5, alongside their
associated compatibility shims.

Important boundaries established by this matrix:

- Both older adapters consume the legacy map and ignore a new-envelope-only
  payload. Preserve the legacy representation while those readers are supported.
- The 1.4 frontend's passthrough list lacks `logprob_token_ids`; 1.5 includes it.
  A field accepted by an old worker is not necessarily emitted by its frontend.
- With nonempty explicit token selection, the 1.4 builder retains the requested
  logprob count; 1.5 clears it. Assigning both attributes is not proof of successful
  engine verification or generation on 1.4.
- Old adapters retain an empty token-selection list; the current applicator
  normalizes it to `None`. This is another reason not to equate payload assignment
  with native-server behavior.
- Old builders let the legacy map overwrite canonical sampling values. The
  current reader intentionally rejects contradictory representations.

These excerpt tests do not establish the historical registration path or prove
HTTP, discovery, guided-decoding, RL, or disaggregated N-2 operation. Inspecting
the complete pinned sources shows that 1.5 publishes the generic
`vllm_inference_v1_generate` marker through `engine_generate.py`, called from
`main.py`; 1.4 lacks that publisher. Neither has the new versioned capability.
Do not infer vLLM from absent metadata or a user-chosen endpoint name.

### Explicit legacy identification boundary

[`legacy_vllm.rs`](../src/protocols/common/legacy_vllm.rs) and
[`legacy_vllm.py`](../../../components/src/dynamo/common/legacy_vllm.py)
implement an immutable declaration resolver for older workers without universal
engine identity metadata. The dynamic HTTP frontend accepts declarations through
`--legacy-vllm-targets`, with Rust (`--dyn-chat-processor dynamo`) or Python vLLM
chat processing. Omission leaves automatic capability resolution unchanged.

For example, an operator who has independently verified a Dynamo 1.5.0 vLLM
deployment may declare its exact serving scope:

```bash
python -m dynamo.frontend --legacy-vllm-targets '[{
  "namespace": "release-15",
  "component": "worker",
  "endpoint": "generate",
  "model": "model-a",
  "worker_type": "aggregated",
  "dynamo_release": "1.5.0"
}]'
```

This is a JSON argument, not a file path or a public inference-request field.
It has no environment-variable fallback. Interactive, gRPC, non-dynamic engine,
and SGLang chat-processor configurations reject nonempty declarations.

Each declaration requires exact namespace, component, endpoint, model, worker
role, and Dynamo release. Only the pinned 1.4.0 and 1.5.0 adapters are accepted;
prefix/wildcard scopes, duplicate scopes, unknown keys, missing roles, encoder
roles, and unrecognized releases are rejected. Resolution also requires token
input. The declaration is frontend-owned policy, not proof of the deployed binary
or native HTTP conformance. An operator must keep it aligned with deployment
generations; reusing a scope for another backend requires removing the declaration.

The extension writers accept an explicitly resolved release separately from
runtime facts. The dispatch validator still requires a live selected-worker
runtime snapshot. A present v1 capability is authoritative, including malformed,
null, or field-restricted advertisements: declarations cannot turn a rejection
into permission. Without a v1 advertisement, explicit 1.4 policy permits only
allowed-token and bad-word IDs; explicit 1.5 policy additionally permits token
selection. The 1.4 worker's ability to assign a selection attribute is insufficient
because it retains a conflicting logprob count. No synthetic capability is
written into the model card or runtime metadata.

The implementation carries this policy separately from advertised runtime facts:

1. `FrontendConfig` passes JSON explicitly to `EntrypointArgs`. Rust validates it
   again and carries the immutable policy in `LocalModel`, outside its model card.
   The HTTP model manager accepts it once, before discovery starts.
2. Discovery resolves against the selected WorkerSet's card and exact endpoint.
   Rust preprocessing and routing receive the resolved policy; the Python engine
   factory resolves the same contract using the selected instance and card.
   MDC checksums, cohort selection and admission receivers are unchanged.
   For token-input pipelines, the shared admission wrapper checks the sampling
   extension contract before either processor creates a stream. This is necessary
   because an exception raised inside a lazy Python generator can otherwise arrive
   after HTTP 200 and SSE headers have already been committed. Ordinary requests
   without extension directives do not require a new capability advertisement.
3. Committed prefill bindings resolve their own declaration from their own card.
   A decode declaration does not authorize prefill. The legacy explicit-endpoint
   constructor, which has no committed card, receives no declaration.
4. Immediately before dispatch, both builtin and KV routes validate the actual
   selected worker's live capability. The declaration cannot bypass a missing or
   closed watch, and a present capability remains authoritative.

The identity source is operator-declared, not worker-advertised or runtime-verified.
The partial catalog reports the resulting transport policy, not proof of the
operator's declaration or deployed binary. Do not interpret this option as a
claim of complete N-2 native-serving parity. The
[server reference](vllm-serve-compatibility.md#rolling-upgrade-behavior) records the tested
old-release discovery/HTTP subset. The complete age-direction matrix, replacement
scenarios, and disaggregated execution still need conformance coverage; boundary
tests and extracted release-adapter tests do not substitute for those checks.

Frontend-authored extension errors contain reviewed field names and static reasons,
not request values. Python exposes these through the explicit `InvalidArgument`
exception. Do not make generic `HttpError` or arbitrary backend diagnostic text
public to recover the detail: those messages intentionally undergo sanitization.

## Registering frontend processor identity

The built-in frontend passes its selected factory and
`chat_engine_factory_identity` together to `EntrypointArgs`. The binding accepts
only `vllm`, `sglang`, or `custom`, requires a callable factory on a dynamic
frontend, and stores the identity with the callback. Omission means `custom`;
a callback named after a framework is not identified by inspection. No environment
variable is read or written for this metadata after configuration parsing.

Rust transports this pair as `ChatEngineFactory` through HTTP setup and discovery.
The pipeline descriptor derives chat processor identity from that registration.
Token-input completions remain `rust`, and text-input pipelines remain `backend`,
even when a chat factory was registered. Custom callbacks produce
`external_factory`, preserving explicit uncertainty.

For embedded Python callers, `EntrypointArgs.chat_engine_factory_identity` is a
read-only view of the registered identity (or `None` without a factory). Register
a custom callback without an identity unless it is actually the named built-in
processor. Embedded Rust callers can convert a `ChatEngineFactoryCallback` with
`.into()` for custom identity, or construct `ChatEngineFactory::new` with an
explicit `ChatProcessorIdentity` at the owning registration boundary.

The registrar is trusted to describe the implementation it installs; this is not
code attestation. This metadata does not alter worker capabilities, selected MDC
checksums, cohort admission, or per-hop validation. It must not be used to grant
backend fields or mark an untested pipeline compatible. Implementation revision,
upstream pins, transport, topology, and runtime evidence remain separate dimensions
of a conformance claim. Recognizing the SGLang registration does not implement
SGLang protocol parity in this vLLM iteration.

## Retaining native conformance evidence

Run the native HTTP comparison with `-o tmp_path_retention_policy=all` and a
fresh `--basetemp` directory when collecting evidence. This repository otherwise
retains only failed temporary test directories: a passing processor's capture
can disappear, and a later case may reuse the same numbered directory. Preserve
passing and failing responses, service logs, the command, immutable image/model
pins, and the source-state digest. Identify the processor from the test command
and frontend logs rather than assuming a numbered directory identifies it.

For example, in an already prepared remote validation environment:

```sh
python -m pytest tests/frontend/test_vllm_native_protocol_http.py \
  -o tmp_path_retention_policy=all --basetemp=/path/to/new-run/pytest
```

A passing native matrix only establishes the current tested subset
on its recorded version/model/mode. Failures in another configuration remain failures;
do not exclude them or label the entire target compatible. See the
[server reference](vllm-serve-compatibility.md) for current gaps.

### Actual mixed-release HTTP checks

`tests/frontend/test_vllm_mixed_release_http.py` is an opt-in host-Docker harness.
Set `DYNAMO_PROTOCOL_RELEASE_MATRIX` to a JSON file containing `current` and
`legacy` entries with `image` (immutable digest), `python_source`, `wheel`,
`wheel_sha256`, `runtime_version`, `engine_version`, and `source_commit`, plus
`model_cache`. These are paths on the Docker host. Export the complete historical
Python source from the specified commit and pair it with the released runtime
wheel; substituting current bindings does not test the old release boundary.
Verify historical source blobs and the current dirty-source digest before running.
The harness checks wheel hashes and installed versions inside each container.

Run and profile the age directions separately: the profiling helper has a
300-second outer deadline, independent of pytest's per-test timeout. For example:

```sh
python tests/utils/profile_pytest.py --no-find-min-vram \
  --csv /path/to/new-run/vram.csv \
  tests/frontend/test_vllm_mixed_release_http.py -k new-frontend \
  --models-dir=/path/to/model-cache \
  --basetemp=/path/to/new-run/pytest -o tmp_path_retention_policy=all
```

Use `-k old-frontend` and a separate output directory for the reverse direction.
The current harness selects physical GPU 0; run it exclusively on that GPU, not
under a parallel GPU allocator. Container names are unique and normal teardown
stops them explicitly. An outer process kill can bypass teardown: inspect the
recorded container names and stop only surviving containers owned by that run.
Do not turn a timed-out or failed matrix into a pass by omitting its evidence.

## CI ownership and schedule

The consolidated B1 `protocol-assessment.yml` implements PR, manual and weekly
candidate checks with read-only repository permissions. It does not automatically
adopt pins or implement upstream changes. Its
[ownership and escalation contract](https://github.com/ai-dynamo/dynamo/blob/codex/vllm-protocol-tooling/lib/llm/docs/dynamo-vllm-protocol-assessment.md#ci-periodic-ownership-and-escalation)
is the single reference for triggers, platform selection, gates and artifact
retention. The older deferred `protocol-compatibility.yml` draft is historical,
not a second workflow to activate. Hosted scheduling and required checks are not
established merely by authoring a workflow file.

### Recorded real version-bump exercise

The earlier upstream-only exercise detected 95 source-change candidates for
vLLM 0.29.0 to 0.30.0 and exited 1 awaiting triage. Its immutable reports remain
historical evidence, not a Dynamo/native parity assessment or approved decisions.
Use the consolidated guide's
[real-source version-bump rehearsal](https://github.com/ai-dynamo/dynamo/blob/codex/vllm-protocol-tooling/lib/llm/docs/dynamo-vllm-protocol-assessment.md#reproducible-real-source-version-bump-rehearsal)
for current commands and exact Dynamo/native commits. Keep source assessment,
synthetic tooling regressions and actual runtime conformance evidence distinct.

## Registered pipeline catalog

`GET /v1/models/{model}/compatibility` is a Dynamo-specific read-only diagnostic
subresource. It follows the configured models-path prefix and accepts model IDs
containing slashes. An exact registered model name takes precedence over the
suffix: a model named `foo/compatibility` remains retrievable; its own catalog is
at `/v1/models/foo/compatibility/compatibility`. Unknown models return 404.
The service readiness gate still applies, but a committed model need not be
individually ready to expose its catalog. Consult its `/ready` subresource for
readiness rather than treating a catalog response as permission to send traffic.
An alias reports its own registered name and attached pipeline descriptors. It
does not fall back to a similarly named primary after its membership is removed.
Exact-name precedence also applies to unready, non-displayable models: retrieving
such a literal name returns its readiness error, not a sibling's catalog.

OpenAPI describes model retrieval and both diagnostic subresources with a required
string `model_id` path parameter. Axum's `{*model_id}` wildcard is an internal
routing detail, not an OpenAPI parameter name. The configured models prefix applies
to these documented paths as well as the actual router.

The version-1 response explicitly reports:

- `scope: registered_pipeline_admission`: committed pipelines, including unready
  ones, not a routing reservation or a statement that every endpoint is mounted;
- `coverage_complete: false` and `unlisted_fields: not_catalogued`: omitted fields
  are not classified by this partial catalog, rather than implicitly supported
  or rejected;
- `end_to_end_conformance: unverified`: these descriptors do not establish native
  runtime parity for the entire pipeline;
- `profiles`: distinct endpoint/pipeline admission descriptors, with the exact
  inspected upstream commit selected by the engine-version declaration, and the
  three reviewed sampling fields' transport decisions.

Sampling transport is `v1_with_legacy_copy`, `legacy_only`, `rejected`, or
`outside_this_admission_rule` for non-token RPC paths. It describes capability
permission for a well-typed field, not acceptance of every value or combination.
Type, resource-bound, endpoint-combination, and live selected-worker checks still
apply. The catalog uses the same `SamplingTarget` resolver and per-field check
as lowering; a malformed/restrictive v1 declaration never gains permission from
a legacy fallback. No separate profile YAML or support allowlist is maintained.

Discovery captures the descriptor with the actual wrapped engine. Committed
membership publication controls visibility; removal/replacement removes old
entries, and LoRA views retain the descriptor of the underlying wrapped pipeline.
Equivalent entries are deduplicated in stable order without exposing namespaces,
worker identifiers, checksums, or weights. Different cohorts are listed separately,
not combined into a union of capabilities. Embedded/custom engines without this
registration have `admission: null` and an empty sampling-field list; their
placeholder model cards are not used to invent a native-server identity.

This addresses the design's profile-necessity question for runtime diagnostics:
request types and OpenAPI describe possible JSON shapes, but cannot identify the
processor, selected target rule, and capability precedence captured in a live
pipeline. The HTTP response projects those existing sources rather than adding a
second policy. It intentionally does **not** claim that the full machine-readable
conformance-profile requirement is complete. Live dispatch eligibility, physical
transport, all field semantics, and evidence-linked native parity remain separate
work. The catalog itself has a generated OpenAPI response schema.

The native HTTP regression captures both this diagnostic and `/openapi.json`
after a real worker becomes ready, then runs the existing response comparisons.
Raw diagnostics and inference responses are retained before assertions so a
catalog regression does not erase independent inference evidence. The oracle's
CPU-only tests reject incorrect processor identity, upstream pins, deployment
mode, extension placement, capability transport, and invented whole-RPC versions.
Keep those metadata checks separate from response conformance: a correct
descriptor alone cannot establish that a directive survives execution.

## Adding or supporting a native field

### Compatibility decision metrics

The frontend exports `dynamo_frontend_protocol_decisions_total` through `/metrics`
(or the configured sanitized frontend prefix). It counts **field decisions at a
named boundary**, not unique requests, dispatches, successful generations, or
conformance results. A request may contribute several decisions or be validated
more than once.

| Stage and reason | Meaning |
|---|---|
| `validation / unhandled_native_field` | A known native directive has no handler on this request path and was rejected |
| `validation / unknown_field` | An unknown key was rejected with migration ignore disabled |
| `validation / wrong_endpoint` | A chat-only generation control was rejected on completions; each present key is counted |
| `validation / migration_ignore` | An unknown key was allowed to be ignored after this validation boundary passed |
| `admission / prompt_stream_combination` | The selected profile rejected the prompt-logprob/streaming combination |
| `admission / prompt_admission_passed` | The effective prompt-logprob request passed this admission check; this does not prove downstream preservation |
| `admission / extension_contract_passed` | An effective sampling-extension field passed the selected cohort's lowering contract |
| `admission / extension_bundle_rejected` | The extension bundle failed lowering; `field=sampling_extensions` avoids guessing which member caused it |
| `admission / legacy_extension_envelope` | A field was admitted through the legacy-only sampling envelope |
| `admission / dual_write_extension_envelope` | A field was admitted with the v1 envelope plus its N-2 legacy copy |
| `response_decode / malformed_backend_payload` | A Rust chat/completion converter failed to decode requested prompt-logprob data; the error remains a response error |

Labels are `endpoint`, `target`, `processor`, `transport`, `deployment`, `field`,
`stage`, `decision`, and `reason`. Known field labels come from the generated
vocabulary or the fixed extension vocabulary; arbitrary names collapse into
`unknown`. Other labels come from enums and the selected admission descriptor.
No request value, model name, worker address, request ID, or arbitrary advertised
version becomes a label. The `transport` label identifies the internal RPC
representation, not the physical request plane.

Before target selection, target and pipeline labels are `unresolved`; the two
in-scope typed request validators still provide their endpoint. After selection,
`unidentified` means the selected card lacks the target metadata, while
`vllm/unverified` means its metadata cannot select a reviewed release rule. These
are not implicit support claims. Legacy envelope observations do not upgrade an
unidentified card into a verified native profile.

Response-decoding observations also use unresolved profile context: these
converters know their endpoint, not the authoritative selected-worker profile.
They inspect prompt data only when root `prompt_logprobs` or its `nvext`
projection is requested. Absent/null data is not a decoding failure, since it
may arrive on another chunk. These counters do not diagnose unavailable data
at end of request or instrument the Python processor's decoder.

The counter is process-wide, following the existing terminal-failure counter's
prefix initialization and service registry behavior. An early observation can
initialize the environment/default prefix before an embedded caller constructs
a service; subsequent prefix changes do not rename an existing collector.

Current coverage is deliberately explicit: generic unsupported-key validation,
chat-only generation controls on completions, prompt-logprob admission, the three
reviewed sampling extensions, and requested prompt-data decoding in the Rust
response converters. Other typed-field validators, aliases, live per-hop failures,
other backend decode failures, and unavailable response fields still need observations.
Do not use absence of a counter increase to claim those paths were accepted or
compatible. Preserve this distinction when extending coverage.

### Required field-change procedure

Frontend-local compatibility errors can attach
[`CompatibilityRejection`](../src/protocols/openai/compatibility/rejection.rs)
to the existing canonical error. The HTTP renderer adds its structured details
only to canonical invalid-request HTTP 400 responses; it does not turn an
internal failure into a client error. Keep field identity, stage, profile and
alternatives bounded and independent of request values or backend diagnostics.
Test both error-chain wrapping and actual endpoint serialization. Early request
validation must leave the profile null; it must not guess a selected pipeline.
Use `known_native_field` to obtain static identities from the generated inventory.
For several unsupported native fields, choose the primary field alphabetically
and retain the complete recognized list in the safe message. Unknown caller keys
must not become structured compatibility identities.

Coverage currently includes `prompt_logprobs`, unhandled known native fields,
the three reviewed sampling fields' token-array validation, token selection
without output logprobs, and their capability failures at profile admission.
`SamplingCapabilityFailure` preserves static causal field identity through shared
lowering; admission attaches its actual profile without parsing diagnostic text.
Aggregate envelope failures, other validators, live per-hop errors, and
worker-originated errors remain outside this annotation. Extending coverage does
not by itself establish native error parity or cross-process transport.

Inventory membership alone never completes implementation. Classify whether the
field affects frontend behavior, backend execution, response projection, cache
identity, or more than one of these. Then update the typed public schema or
controlled capture policy, capability checks, preprocessing, versioned transport,
backend lowering, and projection as applicable.

Tests must cover omission versus explicit null/zero/false/empty values; conflicts
and aliases; unsupported versions/deployments; and both upgrade directions in
Dynamo's N-2 window for every new internal representation. Establish chat-completion
conformance first. Add completion behavior only where needed for compatibility.
Update the user contract and per-version reference with the implementation; do
not remove an admission rejection until the supported path preserves semantics.

### Avoid false-positive native comparisons

Treat prompt and generated-token probability normalization separately. Native
vLLM's prompt helper replaces only negative infinity; applying a blanket
`max(value, -9999)` would corrupt valid finite prompt values. The vLLM worker
must normalize before JSON transport, where infinity otherwise becomes null.
`test_vllm_prompt_logprob_decode.py` compares with the installed native helper,
including a value below the sentinel, and separately tests the deliberate
fail-closed policy for NaN/positive infinity. Rust projection fixtures retain
both below-sentinel and normalized values through aggregation. Do not "repair"
an old worker's `-1e30` after transport; it is indistinguishable from a finite
value. Recheck this helper on upstream version bumps.

Generated-token probabilities have a different rule: native chat projection
floors sampled and alternative values at `-9999`. The worker's
`_json_safe_logprob` only replaces negative infinity and rejects NaN/positive
infinity; it preserves finite internal values. Rust/Python public projection
owns the generated-token floor. Keep those responsibilities separate so that
shared worker normalization cannot accidentally floor prompt probabilities.
`test_vllm_generated_logprob_contract.py` compares the worker-to-Python-projection
boundary with the installed native chat helper. The combined regression in
`protocol_prompt_logprob_projection.rs` verifies distinct prompt/generated rules
through both endpoint aggregators. Recheck both native helpers on version bumps.

For every claimed supported value, require a native positive control under a
configuration that enables the feature. Two equal error statuses do not prove
that either server preserved the request's meaning. Check the error field and
the HTTP-versus-SSE boundary for negative controls, and verify payload contents
and placement for successful responses.

For example, native `prompt_logprobs=-1` passes public schema validation for a
non-streaming request but can still fail the engine's `max_logprobs` limit.
The separate native prompt-count test uses both a restrictive limit and an
unrestricted limit, and requires every vocabulary entry in the latter response.
It deliberately makes no Dynamo parity claim. Keep a bounded prompt and retain
compressed raw responses when testing this potentially large payload.

Implementing the Dynamo side must cover the signed public value, Rust and Python
preprocessing, engine-limit admission, and all selected worker hops. The existing
Generate adapter maps `-1` to `u32::MAX` in unsigned internal output options,
and the current vLLM handler reverses that mapping. Inspected Dynamo 1.4/1.5
handlers instead assign the unsigned value directly to SamplingParams. Their
ability to parse the message is not evidence that they understand the sentinel.
Do not enable public admission based solely on the current decode worker or
change the N-2 output-options wire type without a compatibility strategy.
Any alternative lowering to a finite full-vocabulary count must use verified
engine vocabulary size, not assume it equals tokenizer vocabulary size.

The implemented public chat/completion path now uses a signed count while
preserving `OutputOptions.prompt_logprobs: Option<u32>` on the internal wire.
Only `-1` maps to `u32::MAX`; invalid negative values and the reserved positive
sentinel are rejected. Workers publish `runtime_data.vllm_prompt_logprobs`
with schema version 1, `wire_count: "u32_max"`, their effective `max_logprobs`,
and model-config `vocab_size`. This capability is separate from sampling-field
and generic Generate capabilities. Frontend admission, Rust/Python conversion,
and live selected-hop validation use it; the Python frontend also uses the
worker's limit when constructing its native request validator.

The compatibility endpoint exposes
`full_vocab_prompt_logprobs_unary_admitted` from that same admission policy.
This is a registered-pipeline rule, not a live all-worker conformance claim.
Keep its distinction from runtime evidence: current aggregated HTTP tests
cover both processors. Pairwise current/1.4/1.5 tests establish pre-stream
rejection for `-1` and preservation of omitted/null ordinary controls in both
upgrade directions. They do not establish full-vocabulary output through old
releases, concurrent mixed-worker cohorts, or actual disaggregated execution.
Route-level mocked lifecycle tests cover lost capabilities and closed watches
but cannot substitute for those deployment tests.

The mixed-release harness uses separately pinned runtime wheels and Python
sources; verify both rather than substituting current bindings into an old
worker image. Its full-vocabulary tests use unrestricted native positive
controls and retain compressed responses. Profile one parameterized deployment
case per `profile_pytest.py` invocation: its single-pass timeout is 300 seconds
for the whole invocation, independent of the per-test timeout. A hard profiler
timeout can bypass the test's Docker cleanup; inspect and stop only the exact
task-owned containers before retrying. Do not classify that interrupted run as
successful or leak-free.

## Extending to another target server

The consolidated tooling currently implements only vLLM. Add a target-specific
extractor under `scripts/protocol_compatibility/extraction/` that returns the
normalized contracts and explicit coverage diagnostics defined in `common/`. Preserve the separation between static candidates and runtime evidence.
Add fixtures for declarations, aliases, validators, response/stream changes, and
irrelevant formatting. Add that framework's configured-version checks, native
conformance probes, ownership, and reference before advertising support.

SGLang and TensorRT-LLM are design targets, not implemented extractors in this
iteration. Their native protocols must not be inferred from the vLLM inventory.
