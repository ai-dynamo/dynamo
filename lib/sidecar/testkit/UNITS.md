<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Isolated unit coverage (DIS-2942)

The unit increment in [#15089](https://github.com/ai-dynamo/dynamo/pull/15089)
builds on the shared foundation in [#14879](https://github.com/ai-dynamo/dynamo/pull/14879),
refreshed onto main `4a0547f8ba2675f14d50e48d6aec53b1bc3cf3e3`.
Its pinned versions remain vLLM 0.29.0
(`98dff2a81d747d1dba01a47f939f48c3526d4206`) and `vllm-proto` 0.3.0.
The source matrix used older Dynamo snapshots and vLLM 0.28.0. Current native
Control discovery, health, metadata, multimodal forwarding, LoRA, Encode, RL
administration, priority and rank selection are supported and retain their
existing assertions. The DEP and its five tabs remain read-only; departures are
recorded separately in [DEVIATIONS.md](DEVIATIONS.md).

Tests live in `tests/unit/` and are included as `#[cfg(test)]` child modules by
their production owners. They use no sockets, processes, engines or model
downloads. Common production code runs once; shared unit scenarios are
instantiated for vLLM first. SGLang unit setup and enrollment are follow-up
work. The existing shared wire suite still instantiates both backends, both retained Mocker suites
remain, and E2E allocation is unchanged. Additional wire, process and native
integration belong to [#15091](https://github.com/ai-dynamo/dynamo/pull/15091), as
mapped in [COVERAGE.md](COVERAGE.md).

## Shared and backend-specific cases

Test definitions live in two case files. Small source-group macros include each
group under its private production owner; they do not implement conversion or
model another backend's protocol.

```text
src/
  fixtures.rs                 # Common inputs for unit and integration tests
  assert.rs                   # Common output assertions
tests/
  unit/
    shared.rs                 # Common-code and cross-backend test definitions
    vllm.rs                   # Only vLLM-specific test definitions
    support/
      mod.rs                  # Per-test lane macro
      vllm.rs                 # Plain native setup and production calls
  support/
    fixtures/
      vllm.rs                 # Native input builders for both test suites
    ...                       # Existing integration server support
  conformance.rs              # Existing shared CPU integration scenarios
```

| Source | Responsibility |
| --- | --- |
| `src/fixtures.rs` | Minimal canonical request plus the existing integration input builders. |
| `tests/support/fixtures/vllm.rs` | Rich requests and native model, response, media and handoff builders used by unit and retained wire tests. |
| `tests/unit/shared.rs` | Complete common-code and cross-backend scenarios: inputs, production calls, assertions and lane declarations. |
| `tests/unit/vllm.rs` | Native request, response, configuration, JSON, LoRA and candidate assertions. No shared-test wrappers. |
| `tests/unit/support/mod.rs` | The `sidecar_test!` lane declaration macro. |
| `tests/unit/support/vllm.rs` | Model, worker, request and response setup functions grouped for inclusion beneath private production owners. No test definitions or scenario assertions. |

The production owners' test-only hooks enroll shared and native groups
separately. Compiled module names use `unit_common_*`, `unit_shared_*` and
`unit_native_*`, so registration and runner reporting preserve the distinction.
Common production code is tested once. No empty SGLang or TensorRT-LLM files are
created before those backends are enrolled.

Share a complete scenario when its contract is common and a plain function or
small amount of native setup connects it to production code. Prefer existing
Dynamo input/output types. If sharing needs a second request/response model,
capability matrix or substantial backend-dependent behavior, keep the test
native. A shared fixture alone does not make a test shared. The number of shared
cases follows these decisions; it is not a quota.

The ten shared backend scenarios are:

| Group | Shared scenarios |
| --- | --- |
| Requests | Oversized logprob rejection, selected LoRA forwarding, prefill/decode rank selection and fallback. |
| Responses | Prompt-logprob opt-in/positions/values and terminal-reason preservation through real conversion helpers. |
| Model | Model identity/limits, absent optional limits and logical block size/per-rank capacity. |
| Worker | Worker options/model identity and generation/cleanup before startup. |

Private production functions remain private. Native setup calls the real
functions rather than reproducing their transformations. The vLLM crate has a
test-only dependency on testkit to reuse its common fixtures; normal sidecar
builds gain no dependency on testing infrastructure.

| Native exception | Why it remains backend-specific |
| --- | --- |
| Native sampling, structured output, priority and cache controls | Exact native fields and support rules differ. Direct assertions avoid a common observation type and capability matrix. |
| Legacy `vllm_tito` envelopes, KV aliases and port normalization | These keys, compatibility precedence and wire representations are vLLM contracts. |
| Protobuf Struct integer bounds and nonfinite values | The double-number encoding has precision limits absent from SGLang's typed integers and JSON strings. |
| Image-only Encode, media identifiers and encoder responses | SGLang sidecar has no corresponding Encode service. |
| LoRA loading/inventory, RL capabilities and draft updates | These management schemas differ from shared worker configuration. |
| Discovery fallback, missing model identity, topology ownership and startup compatibility | Shared cases check resulting identity, limits and block capacity; field validation, fallback and native compatibility remain local. |
| Streaming conversion, malformed messages, logprob metadata and handoff timing | vLLM has a directly callable response state; SGLang conversion is coupled to its live stream. Do not imitate that loop or invent a synthetic response interface just to share its tests. Common response checks are shared only where the actual production helper is callable. |

## Per-test lanes and runner selection

Each test declares the earliest lane in which it runs:

```rust
sidecar_test! {
    lane: pre_merge;
    #[test]
    fn preserves_request_fields() {
        // Scenario assertions.
    }
}
```

`#[tokio::test]` and result-returning tests use the same declaration. The macro
encodes one lane in the compiled test name; invalid or missing lane declarations
in governed unit modules fail compilation or inventory validation.

| Selected lane | Test declarations included |
| --- | --- |
| `pre-merge` | `pre_merge` |
| `post-merge` | `pre_merge`, `post_merge` |
| `nightly` or `all` | `pre_merge`, `post_merge`, `nightly` |

`--suite unit` selects isolated units, `--framework vllm` selects the activated
backend plus common tests, and `--lane` selects this cumulative lane set.
`--framework all` currently has the same backend enrollment. Explicit
`--framework sglang` fails until its unit setup is implemented; it does not
fall back to vLLM. The compatibility aliases are `--level pre-merge` for
`--suite unit --lane pre-merge`, and `--level unit` or `--level all` for
`--suite unit --lane all`. Conflicting `--level` and `--lane` values fail.

The runner inventories compiled test binaries, classifies common/shared/native
cases, validates every governed lane marker, and executes exact selected names.
Export records all unit lanes in versioned `tests.json`, even when an earlier
lane was requested. Running exported artifacts recollects each binary and
checks it against the stored inventory before selection. Empty, duplicate,
missing or mismatched inventories, failures, and ignored governed tests fail.
The reorganized suite retains **81 test registrations: 11 common, 10 shared
vLLM instances and 60 native cases**, all `pre_merge`. The shared backend cases
comprise three request, two response, three model and two worker scenarios.
Compiled collection and execution confirmed these counts; they track coverage
rather than an acceptance quota.

The PR workflow selects pre-merge. Push-triggered execution of
`pre-merge.yml` selects post-merge, and nightly Rust coverage includes all lanes.
The workspace CI entry point validates the unit inventory, runs other packages
with their original Cargo arguments, then runs all targets of registered unit
owners with lane filters. This preserves their legacy socket and executable
tests without forwarding libtest arguments to unrelated custom benchmarks.
The isolated unit selection does not claim integration execution.

## Enabling SGLang in a follow-up

Add `unit/support/sglang.rs` for the native setup needed by existing shared
scenarios, and `unit/sglang.rs` only for backend-specific cases. Include shared
groups through SGLang's test-only production hooks and enroll the backend in the
runner. Then `--framework sglang` selects its shared and native tests plus common
tests; `--framework all` includes both backend instances with common code tested
once. Reuse `src/fixtures.rs` and add native builders under
`tests/support/fixtures/` only where necessary.

Consolidate existing SGLang model, rank-fallback and other matching assertions
before removing an old test. Sharing must use actual production functions, not
copied converters, fake production behavior or a dummy setup that makes the flag
succeed. In particular, live response-stream conversion stays at the integration
boundary unless a real callable production helper provides the unit boundary.
SGLang worker construction currently performs bootstrap I/O; exposing a private
in-memory construction seam is separate follow-up work.

Account separately for SGLang's typed gRPC and opaque HTTP paths. Typed gRPC
rejects seed, nonzero priority and some guide forms, whereas the HTTP envelope
forwards native sampling fields and uses a different priority representation.
Special-token policy, cache controls, invalid top-k, orphan media identifiers and
conflicting guides require backend-specific decisions and checks; they are not
reasons to add capability branches to every shared unit. These are unexecuted
parity observations, not demonstrated user-facing failures. Early bootstrap
handoff also needs its own protocol assertions rather than vLLM's
completed-prefill handoff expectation.

## R01–R32 mapping

Each row credits retained assertions before additions. File links identify the
executable owner; deferred scenarios do not count as executed coverage.

| ID | Retained coverage, additions or justified disposition | Owner |
| --- | --- | --- |
| R01 | Move existing endpoint authority, scheme, IPv6 and HTTP checks unchanged. | [Common endpoints][endpoints] |
| R02 | Move common defaults, overrides and zero rejection; defer new SGLang clamp cases. | [Common arguments][args] |
| R03 | Retain discovery/config checks; add exact WorkerConfig fields, model identity, parser and Encode rejection through production `from_discovered`. | [Shared worker/model cases][shared-worker], [native configuration][config]; retained bootstrap sockets |
| R04 | Execute the actual retry policy with virtual time: first failure/retry, all pool slots, one absolute deadline, bounded attempt/sleep and retained peer cause. | [Common transport][transport]; existing connection sockets |
| R05 | Move and extend the status table to all 17 codes, categories, RPC/code/message text and both tonic versions. | [Common errors][errors] |
| R06 | Move exact nondefault native field assertions from the aggregate socket test; add optional/zero/false sentinels, deduplicated stop IDs and top-k boundaries. Keep gRPC metadata and outputs in wire tests. | [Requests][requests] |
| R07 | Retain existing logprob/media/Encode rejections; add supported/default versus unsupported generation, thinking, media, overrides and cache-input cases. Preserve no-RPC admission checks. | [Requests][requests]; [retained socket tests][legacy] |
| R08 | Assert exact guide types/payloads and conflict, backend and whitespace rejection. | [Requests][requests] |
| R09 | Exercise real ResponseState empty messages, delta tokens, empty engine text, terminal and usage. Actual generate-loop termination stays in wire tests. | [Responses][responses]; [streaming wire][streaming] |
| R10 | Check typed malformed sequence/count/finish/logprob/prompt metadata failures. Real stream termination after conversion failure belongs to the additional wire coverage in #15091. | [Responses][responses]; [error wire][wire-errors] |
| R11 | Keep open/read/EOF and postterminal consumption at the real generation boundary; do not copy its loop into units. | [Error wire][wire-errors], [streaming wire][streaming] |
| R12 | Keep cancellation and remote release at wire level, including decode's first-token safeguard. | [Cancellation wire][cancellation]; [retained decode sockets][legacy] |
| R13 | Add vLLM unstarted generation and idempotent cleanup; remove only its matching wire subsection after replacement validation. Preserve SGLang's original subsection and both backends' active cleanup. Post-cleanup admission additions belong to #15091. | [Shared worker cases][shared-worker], [lifecycle wire][lifecycle] |
| R14 | Concurrent identity and cancellation isolation use the shared wire scenario, without a duplicate backend unit loop. | [Cancellation wire][cancellation] |
| R15 | Retain selected-only logprobs; add exact multi-token selected/top IDs, ranks, values, opt-in and alignment. | [Responses][responses] |
| R16 | Retain prompt-metadata timing; add exact first-position null, selected/alternate payload and terminal-only opt-in. | [Shared responses][shared-responses], [native responses][responses] |
| R17 | Add user string/token, system EOS and hidden-token overlap table; fix exposed system-only stop reasons. | [Responses][responses] |
| R18 | vLLM releases GenerateStream; it has no targeted Abort RPC to test. Explicit-RPC fixtures for other backends are deferred; R12 remains required. | [Cancellation wire][cancellation] |
| R19 | Retain opaque/repeated handoff sockets; add decode precedence, port normalization, malformed/missing payload, prefill suppression/usage and failed-prefill no-handoff. | [Requests][requests], [responses][responses]; retained handoff sockets; process/native additions in #15091 |
| R20 | Assert current canonical prefixed cache identity, no model-checksum fallback, bypass precedence and redundant-input consistency. | [Requests][requests] |
| R21 | Move existing JSON cases; add signed integer limits, fractions and nested nonfinite incoming values. | [JSON conversion][json] |
| R22 | Retain negative-infinity normalization; add NaN, positive infinity and finite-underflow association checks. | [Responses][responses] |
| R23 | Defer new SGLang JSON discovery fixtures; cover supported vLLM native identity, aliases, optional metadata and startup compatibility. | [Model config][config] |
| R24 | Retain vLLM local ranks and capacity checks; consume authoritative effective block size with legacy fallback, without reproducing engine arithmetic. Stock 0.29 producer limitation remains below. | [Shared model cases][shared-model], [native configuration/ranks][config] |
| R25 | Defer new SGLang health fixtures. Isolate vLLM incompatible local configuration; retain existing wire checks; additional process registration sequencing belongs to #15091. | [Worker config][worker]; integration |
| R26 | SGLang bootstrap-address policy is not a vLLM contract. Cover vLLM opaque port handling in R19 and roles in R03. | New SGLang cases deferred |
| R27 | SGLang room/rendezvous protocol is not a vLLM contract. Cover failed-prefill/success handoff in R19. | New SGLang cases deferred |
| R28 | Refresh stale unsupported claim: check vLLM LoRA name and role-specific DP/prefill ranks, fallback, load schema and inventory identity; retain administration sockets. | [Shared requests][shared-requests], [native requests/LoRA][lora], [legacy sockets][legacy] |
| R29 | vLLM's request schema has no SGLang trace-header field. Preserve shared tracing coverage; do not invent a native field. | New SGLang cases deferred |
| R30 | Preserve existing SGLang released-protobuf tag test and CI. vLLM uses published `vllm-proto` 0.3.0. | Existing SGLang activation retained; no new backend unit cases |
| R31 | TRT mandatory max-tokens adaptation is not vLLM behavior. vLLM absent/zero sentinel forwarding is R06. | New TRT cases deferred |
| R32 | vLLM GenerateResponse has no TRT cached-token-count field; do not estimate engine cache usage. | New TRT cases deferred |

## Preserved and consolidated assertions

The original unit increment moved these 21 pure definitions from
`vllm/src/tests.rs`. The mapping below follows their assertions through shared
extraction; renamed or split scenarios are not lost coverage. Socket tests remain
in [that file][legacy]. Rich/native builders are in
[the vLLM fixtures](tests/support/fixtures/vllm.rs), used by both layers.

| Original test | New owner |
| --- | --- |
| `engine_config_advertises_supported_capabilities` | Common identity/limits in [shared model cases][shared-model]; native capability assertions in [model config][config]. |
| `rl_worker_metadata_identifies_zero_parallelism_dimensions` | [Model config][config] |
| `discovery_rejects_zero_data_parallelism` | [Model config][config] |
| `startup_compatibility_rejects_parallelism_change` | [Model config][config] |
| `discovery_rejects_incompatible_model_metadata` | [Model config][config] |
| `discovery_rejects_nonzero_dp_start_without_local_size` | [Model config][config] |
| `engine_config_normalizes_total_kv_blocks_per_dp_rank` | `logical_block_size_and_per_rank_capacity_are_registered` in [shared model cases][shared-model]. |
| `engine_config_handles_zero_and_inexact_aggregate_kv_capacity` | [Model config][config] |
| `oversized_logprob_counts_are_rejected` | Same scenario in [shared requests][shared-requests]. |
| `skip_special_tokens_is_forwarded_without_compatibility_envelope` | Direct native special-token assertions in [requests][requests]. |
| `compatibility_envelope_preserves_typed_controls` | [Requests][requests] |
| `native_sampling_is_rejected_instead_of_silently_discarded` | [Requests][requests] |
| `prefill_uses_canonical_controls_without_decode_sampling_json` | [Requests][requests] |
| `released_envelope_hydrates_kv_transfer_with_canonical_precedence` | [Requests][requests] |
| `canonical_dynamo_priority_is_converted_for_vllm` | Native priority scenario, including signed-minimum input and literal wire expectations, in [requests][requests]. |
| `unsafe_media_uuids_are_rejected` | [Requests][requests] |
| `encode_requests_reject_non_image_media` | [Requests][requests] |
| `encode_response_enforces_terminal_contract` | [Responses][responses] |
| `prompt_logprobs_are_retained_for_the_terminal_chunk` | Shared prompt opt-in, positions and values; native early-frame metadata checks in [responses][responses]. |
| `negative_infinity_logprobs_are_normalized` | [Responses][responses] |
| `zero_output_logprobs_omits_top_logprobs` | Native zero/absent/top-candidate assertions in [responses][responses]. |

Existing inline common argument/endpoint/error tests and vLLM JSON, rank and
[candidate extraction](tests/unit/vllm.rs) tests also moved to
their owning isolated modules. The broad aggregate socket test's exact request
field assertions moved into
`representative_request_preserves_all_supported_native_fields` at the previous
boundary. Its current replacements are direct native sampling, stopping and
extension-field assertions; execution evidence is recorded below.
Registration, transport DP metadata, tokens/text/logprobs and usage assertions remain at their wire boundary. No existing SGLang/TRT or
Python/E2E test is migrated or removed by this increment. The four retained
Mocker tests for each of vLLM and SGLang remain; the five additional vLLM wire
replacements are reserved for #15091, not this unit boundary.

The remaining mixed tests are split by obligation:

| Previous combined assertion | Current ownership |
| --- | --- |
| Stops, top-k, selected adapter and rank hints | Shared LoRA selection and rank scenarios; native stopping, top-k and Encode rank assertions. |
| Guide variants and rejected guide options | Native exact type/payload, modifier rejection and conflict scenarios. |
| Missing/malformed handoff and native port normalization | Native handoff validation, opaque payload and port-shape regressions. |
| Optional/zero request values and protobuf defaults | Direct native presence, opt-in and sentinel assertions. |
| Stream chunks, terminal reasons, stop visibility, logprob alignment and prompt metadata | Shared terminal-reason and prompt-metadata helpers; native streaming conversion, stop visibility, selected/top logprobs, invalid shapes, ranks, normalization and early-frame details remain local. |
| Worker identity/options and cleanup before startup | Shared worker scenarios; parser/Encode and native administration exceptions remain local. |

## Pinned native limitations

vLLM 0.29.0's [`inference.proto`](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/rust/proto/inference.proto)
defines zero max-new-tokens as a sentinel. Tests forward it without calculating
an engine default. Its [converter](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/rust/src/server/src/grpc/convert.rs)
forwards opaque transfer data and collapses engine Abort, Error and Repetition
into native Aborted; the sidecar cannot reconstruct those distinct causes.

`vllm-proto` 0.3.0 exposes optional `effective_attention_block_size`, but the
pinned engine's [Control producer](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/rust/src/server/src/grpc/control.rs)
does not supply it. The sidecar now consumes nonzero authoritative values,
falls back for absent/zero legacy values and rejects overflow. Actual DCP
metadata reporting remains an upstream blocker; isolated fixtures do not
establish native producer compatibility.

Multiple output sequences, thinking budget, decoded tensor/media inputs,
UUID-only media, remote-prefill bypass annotations, native trace fields and
arbitrary sampling overrides remain unsupported. Expressible unsupported
inputs receive explicit rejection tests. Native messages cannot expose missing
cached-token, expert-tensor or thinking-token fields. GPU work release and KV
transfer require the separate native integration evidence.

## Validation

The reorganized unit suite was collected and executed locally. Its inventory
retains all 81 prior scenario registrations, with five descriptive name changes.

| Selection | Executed result |
| --- | --- |
| Compiled inventory and exported runner | 81 passed: 11 common, 10 shared vLLM instances and 60 native cases; zero failed or ignored. |
| Complete common/vLLM library suites | 121 passed: 13 common and 108 vLLM; zero failed or ignored. This includes the 81 isolated units. |
| Retained integration suites | Eight testkit conformance, four vLLM Mocker and four SGLang Mocker tests passed. |
| CPU container | All 81 units passed in `sidecar-restack-unit:latest`, with `--network none`, read-only root and artifact mount, and `--cap-drop ALL`. |
| Static checks | `cargo fmt --all --check`, runner Black and Ruff checks passed. |
| Targeted Clippy | Common, vLLM and testkit packages passed with `--all-targets --no-deps -- -D warnings`. |

The permanent runner self-tests and their CI invocation have been removed. Five
one-time checks passed from a temporary copy: inventory validation, export
completeness, CLI compatibility, compiled Rust lane selection and workspace
execution with later-lane failures and a custom benchmark. These small Rust
programs do not constitute a full-workspace Dynamo run. This validation does not
claim current-head GitHub CI or full-workspace execution.

### Previous shared-framework boundary (`0857c0722e`)

Before consolidation and adapter removal, `0857c0722e` passed 81 units
(11 common, 28 shared vLLM instances and 42 native cases), including a
network-isolated CPU run. Its 121 common/vLLM library tests, eight conformance
cases and both four-case Mocker suites also passed. These historical results
belong to the previous structure; the table above records fresh validation of
the reorganized suite.

Use Rust 1.96.1, protoc 30.2 and an external `CARGO_TARGET_DIR`. Set `PROTOC` and
`PROTOC_INCLUDE` to that compiler and its matching includes. From the repository
root:

```sh
python3 lib/sidecar/testkit/run.py --suite unit --framework vllm --lane pre-merge --list
python3 lib/sidecar/testkit/run.py --suite unit --framework vllm --lane pre-merge
python3 lib/sidecar/testkit/run.py --suite unit --framework vllm --export "$artifacts"
docker build -f lib/sidecar/testkit/CPU.Dockerfile -t sidecar-units "$artifacts"
docker run --rm --network none sidecar-units --suite unit --framework vllm --lane pre-merge
docker run --rm --network none sidecar-units --suite unit --framework vllm --lane nightly --list
```

The common target explicitly enables `tonic-v14`, which vLLM consumes, without
relying on workspace feature unification. Record selected names, failures and
ignored cases when validating a new revision. Wire-preservation commands are in
[COVERAGE.md](COVERAGE.md).

### Previous unit boundary

At the previous boundary `b3ab1638`, on foundation `286d6fd5`, the 2026-09-22
record reports **62 isolated cases (11 common, 51 vLLM)** passing in a CPU
container with external networking disabled, zero failed or ignored. Its eight
shared foundation cases and 102 complete common/vLLM library cases also passed
(13 common, 89 vLLM); those selections overlap. Both unchanged four-case Mocker
suites passed independently, and targeted common/vLLM/testkit Clippy passed with
warnings denied. These historical results do not certify the changed shared
bodies, setup, lane selection or workflows.

## Historical execution and failure evidence

The following results belong to the superseded stack on Dynamo base
`cdcd721e72fdfd521c93f35c7f91745b1f7dd01f`; they do not establish results for the
refreshed main dependencies or new PR heads.

The initial unit implementation collected and executed 61 cases: 11 common and
50 vLLM, all passed with zero failed or ignored. A subsequent complete library
run passed 98 cases: 13 common (11 isolated and 2 transport) and 85 vLLM
(50 isolated and 35 socket). These selections overlap; their totals must not be
added. The common/vLLM executions took 0.10/0.15 seconds.

The independent second-PR snapshot was built on historical wire commit
`73192d69351b51763a018edc75b7ec275d9fa68a`, without the third PR's runtime or
process changes. Its first CPU container ran 109 collected cases: 61 isolated,
9 wire scenarios, 2 retained vLLM Mocker, 2 common transport and 35 vLLM socket
cases. This historical snapshot had already applied wire migrations that are
now reserved for #15091, so its retained-suite counts do not describe #15089.

The final G5 case,
`engine::unit_worker::draft_updates_require_both_native_capabilities`, checks
absent metadata, every draft/transfer flag combination, exact advertisement and
rejection before native-client access. It passed, bringing the historical
isolated suite to 62 cases (11 common, 51 vLLM). The final CPU container passed
all 62 with zero failed or ignored. Including that snapshot's 48 wire/retained
cases, historical #15089 at `10456cb1397acfe5cd9bafdf8e3bfd5369a1c12c` executed
110 cases. Its [Pre Merge run](https://github.com/ai-dynamo/dynamo/actions/runs/35414971016)
and [full PR run](https://github.com/ai-dynamo/dynamo/actions/runs/35414973825)
passed on that exact head. Containers had external networking disabled and no
GPU devices, model cache or inference-engine installation.

Four initial assertion failures exposed three production defects: system-only
stop reasons leaked; aborted prefill required or published success handoff; and
effective block metadata was ignored, including an unrepresentable value.
Regression fixes retain exact assertions. A separate JSON expectation assumed
sorted keys despite insertion-order serialization; correcting the expected
order fixed that harness defect without changing production.

| Temporary production mutation | Assertion that failed, then passed after restoration |
| --- | --- |
| Swap presence/frequency penalty fields | `convert::unit_requests::representative_request_preserves_all_supported_native_fields` |
| Increment emitted token IDs | `convert::unit_responses::empty_messages_and_delta_tokens_preserve_terminal_usage` |
| Map Unavailable to Unknown | `error::unit_common::maps_transport_statuses_to_backend_errors` |

Each historical mutation compiled and failed its targeted assertion; restored
sources passed and no mutation remains. These observations motivate the
regressions; they are not a new mutation run on the refreshed stack. Existing
Python coverage was not executed by the unit commands above. Counts describe
observed cases, not acceptance quotas or full legacy Python parity.

[endpoints]: tests/unit/shared.rs
[args]: tests/unit/shared.rs
[transport]: tests/unit/shared.rs
[errors]: tests/unit/shared.rs
[worker]: tests/unit/vllm.rs
[config]: tests/unit/vllm.rs
[ranks]: tests/unit/vllm.rs
[requests]: tests/unit/vllm.rs
[responses]: tests/unit/vllm.rs
[json]: tests/unit/vllm.rs
[lora]: tests/unit/vllm.rs
[legacy]: ../vllm/src/tests.rs
[streaming]: tests/conformance.rs
[wire-errors]: tests/conformance.rs
[cancellation]: tests/conformance.rs
[lifecycle]: tests/conformance.rs

[shared-requests]: tests/unit/shared.rs
[shared-responses]: tests/unit/shared.rs
[shared-model]: tests/unit/shared.rs
[shared-worker]: tests/unit/shared.rs
