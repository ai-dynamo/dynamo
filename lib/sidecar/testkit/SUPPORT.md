<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# vLLM Support Matrix reconciliation

This ledger accounts for all **162 capability IDs** in the read-only DEP Support
Matrix and its two unnumbered structured-output restrictions. It refreshes that
matrix's Dynamo `8e9a96f` / vLLM 0.28 assumptions against the implementation base
`cdcd721e72fdfd521c93f35c7f91745b1f7dd01f`, the changes in this stack, published
`vllm-proto` 0.3.0, and vLLM 0.29.0
(`98dff2a81d747d1dba01a47f939f48c3526d4206`). Pins are in
[Cargo.toml](../../../Cargo.toml) and [container/context.yaml](../../../container/context.yaml).
Only vLLM receives new fixtures and activation. Existing SGLang/TensorRT-LLM,
shared-production and E2E coverage remains with its current owner.

An entry credits only the stated boundary; forwarding a parameter does not prove
engine mathematics, native acceptance or production parity. No row claims
production validation. Required incomplete coverage is distinguished from
separately owned E2E, performance and deployment work.

## Evidence and ownership

| Label | Owner and execution evidence |
| --- | --- |
| U | PR2 [isolated units](UNITS.md): 62 cases passed (11 common, 51 vLLM), including the final isolated-container run. |
| R | Retained [vLLM socket tests][retained], including LoRA/RL/media/metadata: 37 retained socket cases passed (2 common, 35 vLLM), including the isolated CPU-container runs; U is accounted separately. |
| W | PR1 [CPU wire](COVERAGE.md): 9 shared conformance cases and 2 retained [Mocker handoff/KV relay cases][mocker] passed. |
| P | PR3 [process integration](PROCESS.md): 6 cases passed after rebuilding the real sidecar. |
| N | PR3 [native integration](NATIVE.md): actual GPU cancellation/drop passed; native handoff failed before transfer. Native logprob/structured-output compatibility passed after correcting its test inputs and native layout expectations. |
| S | Existing shared-production or other owning tests cited below were retained and inspected; this reconciliation does **not** claim their execution. Existing CI remains allocated. |
| Unsupported | No supported sidecar contract is invented. Rejection tests cover expressible inputs where identified. |
| Deferred | The named E2E, engine, performance or deployment owner remains responsible; no execution credit is taken. |

Commands, collected cases, reproduced failures and per-layer limitations are in
the linked reports. The counts overlap where noted and are not an acceptance
quota. Final per-commit/container/current-head CI evidence must be read separately;
source inspection and CI configuration are not executions.

## Required outstanding evidence

- **A2 / F3 / H10 / J2:** native NIXL handoff fails on pinned vLLM: its protobuf
  `Struct` decoder retains numeric fields as floats, and `range(remote_pp_size)`
  rejects `1.0`. Decode emits zero tokens; completed-transfer telemetry is empty.
  The exact output and positive-transfer assertions remain enabled and failing.
- **F2 / F3 / F4:** DEP integration C12's cancellation after native peer acceptance
  and before the first decode token has CPU coverage, but actual transfer/work
  release remains blocked by the failed native handoff. Cancelling before reading
  a buffered token would not prove that NIXL was still in flight.
Completed during reconciliation: **B14 / B20 / B21 / J2**, C13's native logprob
and structured-output case passed on the pinned engine. It checks aligned
candidate and prompt metadata, exact usage, and parsed schema-conforming text.
Completed during reconciliation: **G5**, a new isolated `draft_updates_require_both_native_capabilities` case
  checks absent capabilities and all flag combinations, exact advertisement and
  rejection before accessing a native client. The focused case and final 62-unit isolated CPU container passed. The
  retained positive route case continues to cover successful updates.

Stock 0.29 also lacks two metadata producers already represented by the newer
published protocol: `data_parallel_size_local` and
`effective_attention_block_size`. The sidecar's positive fixtures establish
consumption only. The pinned [Control schema][native-control-proto] and
[producer][native-control] expose neither field; world size and LoRA are exposed.
The E+P+D metadata-only decode change [vLLM #54814][native-epd] is also absent
from v0.29.0; its merge commit is not an ancestor of the pin.

## A. Serving topologies

| IDs | Current vLLM disposition and sufficient owner |
| --- | --- |
| A1 | U/R/W/P cover aggregated conversion, real native-wire streaming and Worker ingress. N proves basic real-engine generation and cancellation; preserved [sidecar E2E][e2e] owns HTTP serving. |
| A2 | R `prefill_decode_handoff_is_opaque_and_repeatable`, W Mocker handoff and P real PrefillRouter cover metadata and orchestration. N's actual NIXL transfer remains a failing required blocker, not covered by the CPU peers. |
| A3 | U `worker_config_preserves_options_and_discovers_model_identity`, R `component_honors_config_for_aggregated_but_fixes_disagg_roles`, P advertised prefill/decode roles. |
| A4 | Conditional `x-bypass-remote-prefill` annotation remains unsupported. U cache-bypass alias tests concern prefix caching, not this distinct topology annotation. |
| A5 | The old blanket rejection is stale: U rank-range tests and R `hybrid_discovery_routes_and_tracks_only_local_absolute_dp_ranks` cover supported supplied local ownership. Stock 0.29 lacks the local-size producer; actual hybrid/multinode TP/PP/EP execution is not certified. Multinode scheduling belongs engine/deployment validation. |
| A6 | Headless native worker launch is engine/deployment-owned; no new multinode launch matrix or execution claim. Retain [launch scripts][launch]. |
| A7, A8, A9, A10 | Embedding, pooling/classification, diffusion/image/video generation and realtime transcription endpoints remain unsupported by this vLLM sidecar. These are not generation-request transport cases. |

## B. Request and response contracts

| IDs | Current vLLM disposition and sufficient owner |
| --- | --- |
| B1, B2, B3 | U `representative_request_preserves_all_supported_native_fields` owns sampling/penalties/seed forwarding; W/P retain native ingress observations. Deterministic model replay is engine/E2E-owned, not inferred from seed serialization. |
| B4 | U stop deduplication, visible-token rejection, user string/token and hidden/system EOS response table; fixes hidden system-only stop-reason leakage. R retains no-RPC rejection. Visible stop-token semantics remain unsupported. |
| B5, B6, B7 | U exact include-stop-string, ignore-EOS and min/max fields, absent/zero sentinels and prefill clamping. N basic generation exercises limits/ignore-EOS. Broader actual stop-string/model-EOS behavior belongs existing engine/E2E suites. |
| B8, B9, B10, B11 | U supported defaults versus explicit rejection for nondefault length penalty, thinking budget, multiple sequences/best-of and beam search. R/P assert unsupported `n=2` never submits native work. Separate beam API is outside this generation contract; unsupported is more precise than a blanket N/A label. |
| B12, B13 | Logit bias/allowed-token arbitrary sampling extras are not exposed by the released sidecar contract. U `native_sampling_is_rejected_instead_of_silently_discarded` and request-boundary table retain explicit rejection. Frontend omission of logit bias is a separate existing limitation. |
| B14, B15, B16, B17, B18 | U `guide_variants_preserve_exact_type_and_payload` covers JSON schema, regex, grammar, structural tag and choice. W checks supported serialized options. N adds JSON-schema acceptance and parsed exact result; its execution is pending. No grammar/model-quality sweep is added. |
| B19 | U rejects request-level backend and whitespace overrides; engine-wide backend selection remains a native launch option. Testing each engine decoding implementation is deferred to its owner. |
| B20, B21 | U exact selected/top/prompt logprobs, ranks, associations, nonfinite handling and opt-in; W/P exact streamed values and terminal-only prompt metadata. N requested-logprob native compatibility is implemented, execution pending. |
| B22 | Cumulative logprob remains unpopulated. No inference from per-token values and no unsupported success assertion. |
| B23, B24 | U terminal taxonomy, malformed enums, user stop strings/IDs and hidden EOS; W/P transport errors and cancellation. Pinned upstream collapses Abort/Error/Repetition into native Aborted; the sidecar cannot recover absent distinctions. |
| B25, B26 | U ResponseState plus W exact token order, stream termination, post-terminal suppression and usage; P verifies Worker composition. N actual generation checks terminal/usage. In-process differential totals and timing belong E2E/performance. |
| B27 | Supported input is preprocessed token IDs; R preserves native text including empty buffered chunks, W checks decoded text, N schema case consumes native text. A separate text-input RPC path remains unsupported; HTTP preprocessing stays E2E-owned. |
| B28 | Stale untested claim: U `canonical_dynamo_priority_is_converted_for_vllm` and representative request test cover priority direction/value. Actual scheduling policy is engine-owned. |
| B29 | U canonical prefixed cache identity, absent fallback, matching redundant values and precedence; R no-RPC conflicting-salt rejection. |
| B30 | Prompt truncation is not exposed; U pins native zero sentinel. Shared public API rejection remains with the existing frontend owner. |
| B31 | Stale unsupported claim: R LoRA enablement, selection, lifecycle, admission/unload races, inventory recovery and publication rollback; U request rank/name hints, schema and inventory. Real adapter weight mathematics is outside this transport rollout. |
| B32 | U rejects explicit tool/reasoning parser configuration and P asserts absent advertised parser names. Parser implementation remains outside the sidecar contract. |
| B33 | Request-specific custom sampling/logits extras are explicitly rejected by U. Engine-wide processors loaded through native CLI remain engine-owned; no new Python plugin suite. |
| B34 | `embedding_bias` is TensorRT-LLM-specific and outside vLLM applicability. Existing other-backend coverage is preserved. |
| Unnumbered constraints | U guide table tests both `whitespace_pattern` rejection and multiple simultaneous structured-output constraint rejection; these missing matrix rows are not silently omitted. |

## C. Multimodal and Encode handoff

| IDs | Current vLLM disposition and sufficient owner |
| --- | --- |
| C1, C2, C3, C7 | R `mixed_multimodal_media_is_forwarded_with_image_uuid_only` covers image/video/audio URL/data-URI wire variants, prompt usage and image-only UUIDs, including P/D forwarding. Pinned native [conversion][native-convert] now accepts audio/video; the 0.28 release gate is stale. Native codec/model execution is separately owned; audio chat-template placeholder and model-metadata constraints remain documented. |
| C4, C5, C6 | Raw/input-audio/file media, processor kwargs, decoded tensors/RDMA media and UUID-only media remain unsupported. U request-boundary rejection and R no-RPC checks cover expressible invalid inputs. Raw bytes absent from Dynamo's representation are not fabricated. |
| C8, C9, C10 | R `encoder_cache_handoff_is_opaque_for_e_pd_and_e_p_d` and missing-terminal-metadata rejection; U Encode terminal validation and opaque transfer conversion. C9 additionally requires upstream metadata-only decode commit [#54814][native-epd], absent from pinned 0.29.0. Actual native P/D transfer is also blocked as A2. Encoder/model inference is not claimed by CPU tests. |
| C11, C12 | Audio/video encoder disaggregation and per-item encoder fan-out remain unsupported. U image-only Encode rejection is retained; no new unsupported topology. |
| C13 | S [EncoderRouter][encoder] `encoder_failure_falls_back_to_downstream` owns preserving original media on hop failure; existing committed-worker admission tests remain. No claim of live encoder failure or native inline encoding from these local assertions. |
| C14 | U `unsafe_media_uuids_are_rejected` retains separator, NUL and dot-component rejection before submission. |
| C15 | Codec licensing/image contents belong packaging/compliance validation; no legal or decoder approval is inferred from these tests. |
| C16 | R verifies media forwarding through P/D; duplicated preparation cost belongs engine/performance measurements, not another sidecar contract test. |
| C17 | Legacy MM template/custom encoder/local-media/file-size flags have no sidecar equivalent. Unsupported overrides remain outside this rollout. |

## D. KV events, routing and cache

| IDs | Current vLLM disposition and sufficient owner |
| --- | --- |
| D1, D2 | R aggregate/hybrid tests assert complete native event-source rank sets and wildcard address rewriting; W retained `sidecar_relays_stored_and_evicted_blocks` verifies actual store/evict relay and indexer observation. The same mode-independent source/relay code is tested once. Live disaggregated all-stream engine publication is not certified; engine event production and topology E2E remain separate. |
| D3 | Exposed identity/capacity/rank contracts are U/R/P; retained [router E2E][router-tests] owns cache-aware selection. Effective-size producer and engine-load limitations still prevent a blanket routing-parity claim. |
| D4 | Engine load/cache gauges remain unbound. Shared router admission reservations are not substituted for missing native metrics. |
| D5 | U consumes authoritative effective attention block size, tests legacy fallback and overflow; fixes ignored metadata. Stock 0.29 omits the producer field: actual DCP metadata correctness remains an upstream blocker. No engine arithmetic is reimplemented. |
| D6 | S [Worker][worker] `build_local_model_decode_keeps_local_indexer`, `build_local_model_aggregated_keeps_local_indexer`, and Encode force-off; W retained event/indexer round-trip. Distributed indexer recovery belongs router E2E. |
| D7, D8 | U local ranks and role-specific rank hints; R hybrid absolute-rank routing and native gRPC metadata. Native attention-DP scheduling belongs engine coverage; stock 0.29 lacks hybrid local-size metadata as A5. |
| D9 | KVCR/router hints remain unsupported by the sidecar path. No request-hint contract is invented. |
| D10 | Native connector loading is engine-owned; stock attachment still lacks legacy KVBM event consolidation. Existing [KVBM integration][kvbm-tests] is preserved; connector-specific transfer/placement is deferred, not covered by the generic relay. |
| D11 | TensorRT-LLM connector is not vLLM-applicable. |
| D12 | Native offload exists, but offload-tier capacity is not registered by this sidecar; unsupported metadata capability. |
| D13 | R RL route test preserves `pause_generation.clear_cache` payload; standalone legacy flush route remains unadvertised. Actual prefix-cache clearing is native control behavior, not established by payload forwarding. |
| D14 | No verified Mooncake event producer/sidecar contract; unsupported/unverified connector capability, not a required invented producer fixture. |
| D15 | R asserts wildcard ZMQ hosts become the native gRPC host. Explicit loopback remains unchanged; cross-host/pod reachability is deployment-owned and unverified, not proved by local sockets. |

## E. Observability and health

| IDs | Current vLLM disposition and sufficient owner |
| --- | --- |
| E1 | S [metrics][metrics] `lifecycle_gauges_register_and_observe` owns shared registration/observation once. |
| E2, E4, E5, E13 | Engine metric binding, foreign Prometheus passthrough, forward-pass metrics and `DYN_FPM_TRACE` are not supplied by this sidecar. Native scheduler gauges used by N are external test observations, not newly advertised Dynamo telemetry. |
| E3 | Existing shared metric names/labels remain with [metrics][metrics]; full engine scrape parity is unavailable because E2/E4 are unsupported. No cardinality/performance claim. |
| E6, E7 | S [system health][health] and [Worker health-route tests][worker] now cover payloadless serving readiness and manual canary policy; the old unconditional NotReady concern is stale. No automatic vLLM role/model payload is supplied. New process readiness is registration evidence, not a native canary test. |
| E8 | Automatic role-specific prefill/embedding/omni canary payloads remain unsupported. Manual shared payload support is not equivalent. |
| E9 | P delayed/failed/interrupted startup explicitly withholds registration until native Control/Inference readiness and valid metadata. U/R reject incompatible bootstrap/start metadata. |
| E10 | S [adapter tracing][adapter] now covers migration links, malformed/no-link handling, first-token stamping and tokenless prefill/Encode terminals. vLLM's released native request has no Dynamo trace-header field; engine-hop span linkage remains unsupported. No duplicate shared tracing suite. |
| E11 | U all gRPC status categories and response protocol failures; W/P exact setup/read/EOF errors and healthy reuse. Native internal-error versus abort distinctions and automatic engine-death withdrawal remain unavailable; N handoff demonstrates the real limitation. |
| E12 | Structured logging is shared-runtime-owned. Existing source/configuration is retained; no new log-schema parity assertion or execution claim. |
| E14 | Native NVTX flag/model instrumentation belongs engine/profiling validation. No trace was captured by this rollout. |

## F. Lifecycle and failure handling

| IDs | Current vLLM disposition and sufficient owner |
| --- | --- |
| F1 | W/P explicit cancellation and pure consumer drop observe remote release and active Mocker scheduler drain; N confirms real native scheduler idle and follow-up serving. vLLM uses stream release, not a targeted Control Abort RPC. |
| F2 | U aborted prefill cannot emit success handoff; P failed-prefill/cancelled P/D cleanup is covered. Actual native prefill/handoff cancellation remains blocked C12 evidence, not inferred from aggregate cancellation. |
| F3 | R both decode cancellation safeguards and premature EOF; P cancels accepted handoff before the first token, drains both peers and reuses routing. Actual in-flight NIXL cancellation/work release is still a required native blocker. |
| F4 | W pre-admission/held-header cancellation, P pending-header cancellation and shared runtime regression fix; actual long-prefill/native-before-first-token release is not established by N's after-first-token aggregate case. C12 remains blocked. |
| F5 | R two-connection use and W/P two-request cancellation isolation are bounded correctness coverage. Cancellation storms, pool exhaustion and head-of-line performance are deferred load/soak work; no arbitrary stress matrix. |
| F6 | S shared Worker drain ordering, failure cleanup and shutdown deadline tests; P successful bounded SIGTERM cleanup. vLLM still lacks quiescence introspection, so prefill can consume its drain budget; no premature-idle success claim. |
| F7, F9 | P actual SIGTERM during startup and active serving: endpoint withdrawal/router exclusion precedes native cleanup and successful process exit. Shared signal dispatch owns SIGINT; no duplicate equivalent process scenario. |
| F8 | Deferred-signal feature is SGLang-specific, not vLLM-applicable. |
| F10 | W actual native-peer termination proves typed stream failure and release. Ongoing engine-death health monitoring/automatic endpoint withdrawal remains unsupported; request failure is not credited as that policy. |
| F11 | Native test launcher owns child groups and cleanup; production supervisor descendant/orphan behavior and Kubernetes lifecycle belong retained launch/deployment tests. No production crash-supervision claim from fixture cleanup. |
| F12 | U/W/P cover bounded startup retries and request-error recovery on the same live peer. Engine restart/reconnect availability is not exercised; deferred deployment/fault-tolerance work, as distinguished from baseline reconnection/load additions in the integration plan. |
| F13 | S shared retry/router and adapter trace tests retain migration behavior; this stack fixes false worker inhibition after local cancellation. Full frontend migration between real workers remains separately owned [fault-tolerance coverage][fault-tests]. |
| F14, F15 | CRIU restore-standby and GMS shadow mode are not implemented by the attach-to-engine sidecar path. Existing snapshot/GMS suites remain; telemetry publication is not checkpoint/restore support. |

## G. Administration and weight updates

| IDs | Current vLLM disposition and sufficient owner |
| --- | --- |
| G1, G12 | R `rl_startup_*` cases, aggregate metadata and U RL identity validate authoritative nonzero world size, explicit older-engine fallback and HTTP URL metadata. Pinned 0.29 now supplies world size, so the old 0.28 blocker is stale. Actual shared HTTP discovery/collective-RPC behavior remains with the RL owner. |
| G2, G3, G4, G6 | R `rl_engine_routes_preserve_lifecycle_payloads_and_version` covers exact pause/resume/status/sleep/wake/transfer/version calls, responses and malformed lifecycle values; `sleep_status_remains_advertised_without_sleep_mode` preserves gating. Actual CUDA sleep/weight correctness stays with native RL tests. |
| G5 | R preserves positive draft weight-update dispatch. New U `draft_updates_require_both_native_capabilities` covers missing/disabled flags and exact no-client rejection; targeted execution is pending. |
| G7 | Stale unsupported claim: R LoRA management tests preserve identity, listing, discovery, independent replicas, locks, rollback/reconciliation, restart and shutdown publication; U schema and inventory checks. Hot swap remains explicitly rejected. |
| G8, G9 | Profile start/stop and elastic EP/capacity routes are not exposed by the sidecar. No unsupported endpoint is activated. |
| G10 | Typed native transfer is covered by G4; legacy disk/distributed/tensor aliases are not equivalent and remain unsupported/incomplete. |
| G11 | S shared Worker `model_taint_update_route_updates_registered_base_model`, reserved-taint rejection and route-ownership tests own metadata publication once. |
| G13 | Admin authentication is absent; network/proxy authentication is deployment/security-owned. Tests do not imply production exposure is authenticated. |

## H. Model metadata

| IDs | Current vLLM disposition and sufficient owner |
| --- | --- |
| H1, H2, H9 | U/R exact model/alias/context/capacity/rank metadata; P reads actual model cards, served identity, tokenizer/formatter and role endpoints. No comparison to in-process metadata is claimed. Effective-size/hybrid producer limitations remain as A5/D5. |
| H3, H6 | P uses real local model/tokenizer assets and publishes usable metadata. Uncached hub/offline/self-hosted asset resolution belongs shared model-loading/deployment coverage; no offline-download success claim. |
| H4 | Model chat-template rendering belongs shared frontend/E2E; P verifies formatter publication, not model-specific rendering. |
| H5 | U `worker_config_preserves_options_and_discovers_model_identity` explicitly preserves `--custom-jinja-template`; actual Jinja rendering belongs shared frontend tests. |
| H7, H8 | ModelExpress weight loading and native load-format/quantization/dtype/revision/loader configuration belong native engine and packaging tests. Independent metadata revision/source alignment remains a documented limitation, not newly implemented behavior. |
| H10 | Exact wheel/bundled-binary 0.29.0 and protocol 0.3.0 are required by N. Native cancellation ran; handoff exposed actual incompatibility. No compatibility claim across arbitrary engine revisions; optional metadata producer omissions are recorded explicitly. |

## I. Deployment and packaging

| IDs | Current vLLM disposition and sufficient owner |
| --- | --- |
| I1 | Existing [sidecar E2E][e2e] invokes aggregate launch scripts; retain tests and CI allocation. P exercises the executable directly. Other launch topologies remain deployment/E2E-owned and were not executed by this audit. |
| I2, I3 | Full DGD manifests and Kubernetes native-sidecar restart/ordering require deployment validation; no cluster launched or resource-floor claim revalidated here. |
| I4, I5, I6 | Image/crate/wheel publication and launcher packaging belong build/release checks. CPU test-image execution is not proof of published product availability; PyO3/Python launcher paths are unchanged. |
| I7 | Recipe migration is separately owned deployment work; existing recipes are preserved. |
| I8, I9 | Product SBOM validity and runtime non-root identity belong packaging checks; no certification inferred from Dockerfile inspection. |
| I10 | Native gRPC/ZMQ network policy is deployment/security-owned and remains an explicit missing product capability. |
| I11 | Retain [executable CLI smoke][cli]; P now proves actual CLI/environment startup and serving. Historical dispatcher smoke in the source matrix is not presented as a new execution. |
| I12 | SGLang-managed Python sidecar contract is not vLLM-applicable; new migration/activation deferred. |
| I13, I14 | Legacy deprecation and migration documentation are separately owned rollout work. This test stack makes no deprecation or production-readiness announcement. |

## J. Test infrastructure

| IDs | Current vLLM disposition and sufficient owner |
| --- | --- |
| J1 | U isolates existing pure assertions plus distinct additions; R keeps native socket assertions. Counts and relocation mapping are in UNITS.md, not the stale 29-test source count. Shared code runs once. |
| J2 | Dependency pinning plus independent real-engine N provides compatibility evidence; a second serializer using the same generated crate would not detect a shared wrong schema. No general released-tag golden matrix is claimed. Required native handoff remains failed and C13 passed. Existing SGLang golden-wire coverage is untouched. |
| J3 | Stale absent-E2E claim: existing `tests/serve/test_sidecar.py` covers all three aggregate sidecar launchers. It remains unchanged; new `tests/sidecar/test_native_integration.py` is direct integration, not replacement E2E. |
| J4, J5 | PR1 pre-merge isolated CPU unit/wire lane; PR3 post-merge/nightly wire/process and direct-native lanes. See [runner](run.py) and [CI workflow][ci]. Collected/executed local evidence is listed above; current-head trusted CI must still be verified per PR. |
| J6 | Configured nightly tests use the pinned Dynamo engine image, not arbitrary upstream HEAD. Upstream-engine CI ownership/bump campaigns are separate; no unobserved upstream run is credited. |
| J7 | In-process versus sidecar differential generation remains separately owned E2E/performance work; no no-regression parity claim from CPU equality assertions. |
| J8 | Native XPU/non-CUDA engine lanes are deferred absent a distinct adapter contract. CPU no-engine execution is not XPU inference validation. Existing hardware suites remain unchanged. |

## K. Performance

| IDs | Current vLLM disposition and sufficient owner |
| --- | --- |
| K1, K2, K3 | TTFT, ITL/TPOT distributions and throughput need matched performance campaigns; no baseline or speedup claim. |
| K4 | Native stream-interval support/tuning and smoothness are engine/performance-owned. W pins logical stream semantics, not latency or batching performance. |
| K5 | R `pool_uses_each_configured_connection` and U configured pool/retry policy retain correctness. Optimal default-eight sizing and load tradeoffs remain unmeasured performance work. |
| K6, K7, K8, K9 | Extra process overhead, handoff latency, duplicated media cost and KV-router hit-rate parity remain performance-owned. N's transfer failure is a correctness blocker, not a latency result. |
| K10, K11 | External Slurm, InfX and AgentX campaigns are separately owned; no external configuration/results certified. |
| K12 | Bounded lifecycle tests observe release; they do not replace a 24-hour FD/memory soak. Long-run leak/load validation remains performance/deployment-owned. |

Changes to the DEP's assumptions, boundaries or sequencing are recorded in
[DEVIATIONS.md](DEVIATIONS.md). None of these dispositions modifies the DEP or
removes existing assertions merely because another test has a similar name.

[retained]: ../vllm/src/tests.rs
[mocker]: ../../mocker/servers/vllm/tests/sidecar.rs
[e2e]: ../../../tests/serve/test_sidecar.py
[launch]: ../vllm/launch
[encoder]: ../../llm/src/kv_router/encoder_router.rs
[worker]: ../../backend-common/src/worker.rs
[metrics]: ../../backend-common/src/metrics.rs
[health]: ../../runtime/src/system_health.rs
[adapter]: ../../backend-common/src/adapter.rs
[router-tests]: ../../../tests/router
[kvbm-tests]: ../../../tests/kvbm_integration
[fault-tests]: ../../../tests/fault_tolerance
[cli]: ../vllm/tests/executable.rs
[ci]: ../../../.github/workflows/shared-sidecar-tests.yml
[native-control-proto]: https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/rust/proto/control.proto
[native-control]: https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/rust/src/server/src/grpc/control.rs
[native-convert]: https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/rust/src/server/src/grpc/convert.rs
[native-epd]: https://github.com/vllm-project/vllm/commit/761c5861e34edb52c290e3171feb3763732ba8ca
