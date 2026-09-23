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
instantiated for vLLM first. SGLang unit adapters are follow-up work. The existing
shared wire suite still instantiates both backends, both retained Mocker suites
remain, and E2E allocation is unchanged. Additional wire, process and native
integration belong to [#15091](https://github.com/ai-dynamo/dynamo/pull/15091), as
mapped in [COVERAGE.md](COVERAGE.md).

## Shared bodies and native adapters

| Source | Responsibility |
| --- | --- |
| `fixtures.rs` | Minimal canonical request with tokens and default options. |
| `fixtures/vllm.rs` | Rich requests and native model, response, media and handoff fixtures used by vLLM units and retained wire tests. |
| `requests/shared.rs` | Canonical request inputs, success/rejection assertions and explicit native representation expectations. |
| `responses/shared.rs` | Token chunks, usage, logprob opt-in/alignment, stop visibility and prefill completion behavior. |
| `config/model.rs`, `config/worker_scenarios.rs` | Common configuration results and worker lifecycle assertions. |
| `requests/vllm.rs`, `responses/vllm.rs`, `config/{vllm,worker}.rs` | Thin adapters calling real production functions, plus native regression assertions. |
| `common/`, `errors/common.rs` | Tests of shared production code, executed once rather than once per backend. |

Shared files are included as a child module named `shared` by each adapter.
Their compiled names identify the backend instance and the scenario. Request
adapters read the actual native fields; response/configuration adapters return
actual common Dynamo output types. Expected constants describe native encoding,
not a second implementation of conversion. Private production functions remain
private. Source inclusion adds no sidecar-to-testkit Cargo dependency.

A capability means either successful preservation or an explicit rejection;
both execute assertions. Missing support must not become an early-returning
passing test. Native-only tests remain where a shared assertion would invent
another backend's protocol or hide relevant wire details.

| Native exception | Why it remains backend-specific |
| --- | --- |
| Legacy `vllm_tito` envelopes, KV aliases and port normalization | These exact keys, compatibility precedence and wire representations are vLLM contracts. Shared cases still check canonical controls and valid/invalid handoffs. |
| Protobuf Struct integer bounds and nonfinite values | The double-number encoding has precision limits absent from SGLang's typed integers and JSON strings. |
| Image-only Encode, media identifiers and encoder responses | SGLang sidecar has no corresponding Encode service; its future encoder must not inherit vLLM-only media restrictions. |
| LoRA loading/inventory, RL capabilities and draft updates | These management schemas are distinct from shared adapter selection or worker configuration. |
| Discovery fallback, topology ownership and startup compatibility | Shared tests assert the resulting identity, limits and effective block size; native tests retain field precedence, overflow and protocol-specific compatibility checks. |
| Malformed vLLM messages, numeric logprob normalization and handoff timing | Native invalid shapes, clamping rules and completed-prefill metadata differ from SGLang's response metadata and early bootstrap handoff. Common output assertions remain shared. |

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
`--framework sglang` fails until its unit adapters are implemented; it does not
fall back to vLLM. The compatibility aliases are `--level pre-merge` for
`--suite unit --lane pre-merge`, and `--level unit` or `--level all` for
`--suite unit --lane all`. Conflicting `--level` and `--lane` values fail.

The runner inventories compiled test binaries, classifies common/shared/native
cases, validates every governed lane marker, and executes exact selected names.
Export records all unit lanes in versioned `tests.json`, even when an earlier
lane was requested. Running exported artifacts recollects each binary and
checks it against the stored inventory before selection. Empty, duplicate,
missing or mismatched inventories, failures, and ignored governed tests fail.
The current source declares **81 tests: 11 common, 28 shared vLLM instances and
42 native regressions**, all `pre_merge`. Compiled collection and execution
confirmed this inventory; counts track coverage rather than an acceptance quota.

The PR workflow selects pre-merge. Push-triggered execution of
`pre-merge.yml` selects post-merge, and nightly Rust coverage includes all lanes.
The workspace CI entry point validates the unit inventory, runs other packages
with their original Cargo arguments, then runs all targets of registered unit
owners with lane filters. This preserves their legacy socket and executable
tests without forwarding libtest arguments to unrelated custom benchmarks.
The isolated unit selection does not claim integration execution.

## Enabling SGLang in a follow-up

Implement SGLang adapters under its private production owners, supply native
fixtures and explicit expectations, and register the backend in the runner.
Then `--framework sglang` must instantiate the existing shared bodies, and
`--framework all` must include both backend instances while common tests run
once. Consolidate existing SGLang model, prefill-limit, rank-fallback, handoff and
logprob assertions before removing any old test. Do not copy shared bodies or
add dummy adapters solely to make the flag succeed.

Account separately for SGLang's typed gRPC and opaque HTTP paths. For example,
typed gRPC rejects seed, nonzero priority and some guide forms, whereas the
native HTTP envelope forwards native sampling fields and uses a different
priority representation. Backend exclusions must identify the path and reason.

The source audit identified parity gaps for typed SGLang special-token policy,
canonical cache controls, invalid top-k, orphan media identifiers and conflicting
guides. Decide whether each input must be supported or explicitly rejected and
validate that behavior before declaring its shared case covered. These are
unexecuted parity observations, not demonstrated user-facing failures. Early
bootstrap handoff also needs its own protocol assertions rather than vLLM's
completed-prefill handoff expectation.

## R01–R32 mapping

Each row credits retained assertions before additions. File links identify the
executable owner; deferred scenarios do not count as executed coverage.

| ID | Retained coverage, additions or justified disposition | Owner |
| --- | --- | --- |
| R01 | Move existing endpoint authority, scheme, IPv6 and HTTP checks unchanged. | [Common endpoints][endpoints] |
| R02 | Move common defaults, overrides and zero rejection; defer new SGLang clamp cases. | [Common arguments][args] |
| R03 | Retain discovery/config checks; add exact WorkerConfig fields, model identity, parser and Encode rejection through production `from_discovered`. | [Worker config][worker], [model config][config]; retained bootstrap sockets |
| R04 | Execute the actual retry policy with virtual time: first failure/retry, all pool slots, one absolute deadline, bounded attempt/sleep and retained peer cause. | [Common transport][transport]; existing connection sockets |
| R05 | Move and extend the status table to all 17 codes, categories, RPC/code/message text and both tonic versions. | [Common errors][errors] |
| R06 | Move exact nondefault native field assertions from the aggregate socket test; add optional/zero/false sentinels, deduplicated stop IDs and top-k boundaries. Keep gRPC metadata and outputs in wire tests. | [Requests][requests] |
| R07 | Retain existing logprob/media/Encode rejections; add supported/default versus unsupported generation, thinking, media, overrides and cache-input cases. Preserve no-RPC admission checks. | [Requests][requests]; [retained socket tests][legacy] |
| R08 | Assert exact guide types/payloads and conflict, backend and whitespace rejection. | [Requests][requests] |
| R09 | Exercise real ResponseState empty messages, delta tokens, empty engine text, terminal and usage. Actual generate-loop termination stays in wire tests. | [Responses][responses]; [streaming wire][streaming] |
| R10 | Check typed malformed sequence/count/finish/logprob/prompt metadata failures. Real stream termination after conversion failure belongs to the additional wire coverage in #15091. | [Responses][responses]; [error wire][wire-errors] |
| R11 | Keep open/read/EOF and postterminal consumption at the real generation boundary; do not copy its loop into units. | [Error wire][wire-errors], [streaming wire][streaming] |
| R12 | Keep cancellation and remote release at wire level, including decode's first-token safeguard. | [Cancellation wire][cancellation]; [retained decode sockets][legacy] |
| R13 | Add vLLM unstarted generation and idempotent cleanup; remove only its matching wire subsection after replacement validation. Preserve SGLang's original subsection and both backends' active cleanup. Post-cleanup admission additions belong to #15091. | [Worker config][worker], [lifecycle wire][lifecycle] |
| R14 | Concurrent identity and cancellation isolation use the shared wire scenario, without a duplicate backend unit loop. | [Cancellation wire][cancellation] |
| R15 | Retain selected-only logprobs; add exact multi-token selected/top IDs, ranks, values, opt-in and alignment. | [Responses][responses] |
| R16 | Retain prompt-metadata timing; add exact first-position null, selected/alternate payload and terminal-only opt-in. | [Responses][responses] |
| R17 | Add user string/token, system EOS and hidden-token overlap table; fix exposed system-only stop reasons. | [Responses][responses] |
| R18 | vLLM releases GenerateStream; it has no targeted Abort RPC to test. Explicit-RPC fixtures for other backends are deferred; R12 remains required. | [Cancellation wire][cancellation] |
| R19 | Retain opaque/repeated handoff sockets; add decode precedence, port normalization, malformed/missing payload, prefill suppression/usage and failed-prefill no-handoff. | [Requests][requests], [responses][responses]; retained handoff sockets; process/native additions in #15091 |
| R20 | Assert current canonical prefixed cache identity, no model-checksum fallback, bypass precedence and redundant-input consistency. | [Requests][requests] |
| R21 | Move existing JSON cases; add signed integer limits, fractions and nested nonfinite incoming values. | [JSON conversion][json] |
| R22 | Retain negative-infinity normalization; add NaN, positive infinity and finite-underflow association checks. | [Responses][responses] |
| R23 | Defer new SGLang JSON discovery fixtures; cover supported vLLM native identity, aliases, optional metadata and startup compatibility. | [Model config][config] |
| R24 | Retain vLLM local ranks and capacity checks; consume authoritative effective block size with legacy fallback, without reproducing engine arithmetic. Stock 0.29 producer limitation remains below. | [Model config][config], [ranks][ranks] |
| R25 | Defer new SGLang health fixtures. Isolate vLLM incompatible local configuration; retain existing wire checks; additional process registration sequencing belongs to #15091. | [Worker config][worker]; integration |
| R26 | SGLang bootstrap-address policy is not a vLLM contract. Cover vLLM opaque port handling in R19 and roles in R03. | New SGLang cases deferred |
| R27 | SGLang room/rendezvous protocol is not a vLLM contract. Cover failed-prefill/success handoff in R19. | New SGLang cases deferred |
| R28 | Refresh stale unsupported claim: check vLLM LoRA name and role-specific DP/prefill ranks, fallback, load schema and inventory identity; retain administration sockets. | [Requests][requests], [LoRA][lora], [legacy sockets][legacy] |
| R29 | vLLM's request schema has no SGLang trace-header field. Preserve shared tracing coverage; do not invent a native field. | New SGLang cases deferred |
| R30 | Preserve existing SGLang released-protobuf tag test and CI. vLLM uses published `vllm-proto` 0.3.0. | Existing SGLang activation retained; no new backend unit cases |
| R31 | TRT mandatory max-tokens adaptation is not vLLM behavior. vLLM absent/zero sentinel forwarding is R06. | New TRT cases deferred |
| R32 | vLLM GenerateResponse has no TRT cached-token-count field; do not estimate engine cache usage. | New TRT cases deferred |

## Preserved and consolidated assertions

The original unit increment moved these 21 pure definitions from
`vllm/src/tests.rs`. The mapping below follows their assertions through shared
extraction; renamed or split scenarios are not lost coverage. Socket tests remain
in [that file][legacy]. Rich/native builders are in
[the vLLM fixtures](tests/unit/fixtures/vllm.rs), used by both layers.

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
| `skip_special_tokens_is_forwarded_without_compatibility_envelope` | `special_token_policy_is_forwarded_or_explicitly_rejected` in [shared requests][shared-requests]. |
| `compatibility_envelope_preserves_typed_controls` | [Requests][requests] |
| `native_sampling_is_rejected_instead_of_silently_discarded` | [Requests][requests] |
| `prefill_uses_canonical_controls_without_decode_sampling_json` | [Requests][requests] |
| `released_envelope_hydrates_kv_transfer_with_canonical_precedence` | [Requests][requests] |
| `canonical_dynamo_priority_is_converted_for_vllm` | Shared priority scenario, including signed-minimum input, with literal vLLM wire expectations. |
| `unsafe_media_uuids_are_rejected` | [Requests][requests] |
| `encode_requests_reject_non_image_media` | [Requests][requests] |
| `encode_response_enforces_terminal_contract` | [Responses][responses] |
| `prompt_logprobs_are_retained_for_the_terminal_chunk` | Shared prompt opt-in/positions/values; native early-frame metadata regression in [responses][responses]. |
| `negative_infinity_logprobs_are_normalized` | [Responses][responses] |
| `zero_output_logprobs_omits_top_logprobs` | Zero/absent/top-candidate table in [shared responses][shared-responses]. |

Existing inline common argument/endpoint/error tests and vLLM JSON, rank and
[candidate extraction](tests/unit/requests/candidates.rs) tests also moved to
their owning isolated modules. The broad aggregate socket test's exact request
field assertions moved into
`representative_request_preserves_all_supported_native_fields` at the previous
boundary. Its current replacements are shared canonical sampling/stopping and
native extension-field assertions; new execution evidence is required below. Registration, transport DP metadata, tokens/text/logprobs
and usage assertions remain at their wire boundary. No existing SGLang/TRT or
Python/E2E test is migrated or removed by this increment. The four retained
Mocker tests for each of vLLM and SGLang remain; the five additional vLLM wire
replacements are reserved for #15091, not this unit boundary.

The remaining mixed tests are split by obligation:

| Previous combined assertion | Current ownership |
| --- | --- |
| Stops, top-k, selected adapter and rank hints | Four shared request scenarios; Encode rank omission remains native. |
| Guide variants and rejected guide options | Shared exact type/payload, modifier support/rejection and conflict scenarios. |
| Missing/malformed handoff and native port normalization | Shared decode handoff validation; native opaque payload and port-shape regression. |
| Optional/zero request values and protobuf defaults | Shared presence/opt-in behavior; exact native sentinel message remains native. |
| Stream chunks, terminal reasons, stop visibility, logprob alignment and prompt metadata | Shared response scenarios; native invalid shapes, ranks, normalization and early-frame details remain native. |
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

## Validation of the shared scenarios and lanes

Local validation of the shared-scenario and lane changes produced these results:

| Selection | Executed result |
| --- | --- |
| Isolated units from compiled binaries | 81 passed: 11 common, 28 shared vLLM instances and 42 native regressions; zero failed or ignored. |
| Complete common/vLLM library suites | 121 passed: 13 common and 108 vLLM; zero failed or ignored. This includes the 81 isolated units. |
| Runner self-tests | Five passed, covering collection, lane selection and artifact validation. |
| Retained integration suites | Eight testkit conformance, four vLLM Mocker and four SGLang Mocker tests passed; zero failed or ignored. |
| Artifact export | All 81 isolated tests were inventoried and exported. |
| CPU container | All 81 units passed with zero failed or ignored, using read-only exported binaries and `--network none`, without engines, GPUs or model mounts. |
| Static checks | Workspace formatting, Ruff and Black for the runner and its tests, and workflow YAML parsing passed. |
| Targeted Clippy | Common, vLLM and testkit packages passed with `--all-targets --no-deps -- -D warnings`. |

No current-head GitHub CI or full-workspace test execution was performed for
these changes. The runner self-tests exercise workspace command selection; they
do not execute every workspace package.

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
ignored cases for the new revision. Wire-preservation commands are in
[COVERAGE.md](COVERAGE.md).

### Previous unit boundary

At the previous boundary `b3ab1638`, on foundation `286d6fd5`, the 2026-09-22
record reports **62 isolated cases (11 common, 51 vLLM)** passing in a CPU
container with external networking disabled, zero failed or ignored. Its eight
shared foundation cases and 102 complete common/vLLM library cases also passed
(13 common, 89 vLLM); those selections overlap. Both unchanged four-case Mocker
suites passed independently, and targeted common/vLLM/testkit Clippy passed with
warnings denied. These historical results do not certify the changed shared
bodies, adapters, lane selection or workflows.

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

[endpoints]: tests/unit/common/endpoint.rs
[args]: tests/unit/common/args.rs
[transport]: tests/unit/common/transport.rs
[errors]: tests/unit/errors/common.rs
[worker]: tests/unit/config/worker.rs
[config]: tests/unit/config/vllm.rs
[ranks]: tests/unit/config/ranks.rs
[requests]: tests/unit/requests/vllm.rs
[responses]: tests/unit/responses/vllm.rs
[json]: tests/unit/requests/json.rs
[lora]: tests/unit/requests/lora.rs
[legacy]: ../vllm/src/tests.rs
[streaming]: tests/conformance.rs
[wire-errors]: tests/conformance.rs
[cancellation]: tests/conformance.rs
[lifecycle]: tests/conformance.rs

[shared-requests]: tests/unit/requests/shared.rs
[shared-responses]: tests/unit/responses/shared.rs
[shared-model]: tests/unit/config/model.rs
[shared-worker]: tests/unit/config/worker_scenarios.rs
