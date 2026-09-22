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
their production owners. The `unit_` module filter selects isolated tests: no
sockets, processes, engines or model downloads. Common production code runs once;
only vLLM receives new backend unit coverage. The existing shared wire suite
continues to instantiate vLLM and SGLang. Both retained Mocker suites and all
existing E2E allocation remain. Additional wire, process and native integration
belong to [#15091](https://github.com/ai-dynamo/dynamo/pull/15091), as mapped in
[COVERAGE.md](COVERAGE.md).

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

## Preserved test definitions

These 21 pure definitions moved from `vllm/src/tests.rs`, retaining their
assertions and names. Socket tests remain in [that file][legacy]. Shared pure
builders moved to [unit fixtures](tests/unit/fixtures.rs), used by both layers.

| Original test | New owner |
| --- | --- |
| `engine_config_advertises_supported_capabilities` | [Model config][config] |
| `rl_worker_metadata_identifies_zero_parallelism_dimensions` | [Model config][config] |
| `discovery_rejects_zero_data_parallelism` | [Model config][config] |
| `startup_compatibility_rejects_parallelism_change` | [Model config][config] |
| `discovery_rejects_incompatible_model_metadata` | [Model config][config] |
| `discovery_rejects_nonzero_dp_start_without_local_size` | [Model config][config] |
| `engine_config_normalizes_total_kv_blocks_per_dp_rank` | [Model config][config] |
| `engine_config_handles_zero_and_inexact_aggregate_kv_capacity` | [Model config][config] |
| `oversized_logprob_counts_are_rejected` | [Requests][requests] |
| `skip_special_tokens_is_forwarded_without_compatibility_envelope` | [Requests][requests] |
| `compatibility_envelope_preserves_typed_controls` | [Requests][requests] |
| `native_sampling_is_rejected_instead_of_silently_discarded` | [Requests][requests] |
| `prefill_uses_canonical_controls_without_decode_sampling_json` | [Requests][requests] |
| `released_envelope_hydrates_kv_transfer_with_canonical_precedence` | [Requests][requests] |
| `canonical_dynamo_priority_is_converted_for_vllm` | [Requests][requests] |
| `unsafe_media_uuids_are_rejected` | [Requests][requests] |
| `encode_requests_reject_non_image_media` | [Requests][requests] |
| `encode_response_enforces_terminal_contract` | [Responses][responses] |
| `prompt_logprobs_are_retained_for_the_terminal_chunk` | [Responses][responses] |
| `negative_infinity_logprobs_are_normalized` | [Responses][responses] |
| `zero_output_logprobs_omits_top_logprobs` | [Responses][responses] |

Existing inline common argument/endpoint/error tests and vLLM JSON, rank and
[candidate extraction](tests/unit/requests/candidates.rs) tests also moved to
their owning isolated modules. The broad aggregate socket test's exact request
field assertions moved into
`representative_request_preserves_all_supported_native_fields` only after that
replacement passed. Registration, transport DP metadata, tokens/text/logprobs
and usage assertions remain at their wire boundary. No existing SGLang/TRT or
Python/E2E test is migrated or removed by this increment. The four retained
Mocker tests for each of vLLM and SGLang remain; the five additional vLLM wire
replacements are reserved for #15091, not this unit boundary.

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

## Validation of the refreshed unit boundary

On 2026-09-22, the restacked unit boundary on foundation `286d6fd5` collected
and executed all **62 isolated cases (11 common, 51 vLLM)** in its own CPU
container with external networking disabled: zero failed or ignored. The runner
selects common code once. All **8 shared foundation cases** and **102 complete
common/vLLM library cases** also passed (13 common, 89 vLLM); the 62 isolated
cases are included in that 102, not additional coverage. The five legacy wire
replacements remain present. Both unchanged four-case Mocker suites were collected and passed independently.
Targeted common/vLLM/testkit Clippy with warnings denied passed. Current-head
trusted CI is pending; historical green runs below do not certify this restack.

Use Rust 1.96.1, protoc 30.2 and an external `CARGO_TARGET_DIR`. Set `PROTOC` and
`PROTOC_INCLUDE` to that compiler and its matching includes. From the repository
root, collect and execute the isolated boundary with:

```sh
python3 lib/sidecar/testkit/run.py --level pre-merge --export "$artifacts"
docker build -f lib/sidecar/testkit/CPU.Dockerfile -t sidecar-units "$artifacts"
docker run --rm --network none sidecar-units --level pre-merge
cargo test --locked -p dynamo-sidecar-common --features tonic-v14 --lib unit_ -- --list
cargo test --locked -p dynamo-vllm-sidecar --lib unit_ -- --list
cargo test --locked -p dynamo-sidecar-common --features tonic-v14 --lib unit_
cargo test --locked -p dynamo-vllm-sidecar --lib unit_
```

The common target explicitly enables `tonic-v14`, which vLLM consumes, without
relying on workspace feature unification. Collection is separate from execution;
record actual selected names, failures and ignored cases for the new revision.
The wire-preservation commands are in [COVERAGE.md](COVERAGE.md).

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
