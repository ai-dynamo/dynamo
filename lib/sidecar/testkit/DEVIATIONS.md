<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Deviations from the sidecar testing DEP

The DEP and every tab remain read-only. This file records rollout and evidence
changes separately. User scope takes precedence over the source reports.

| Original requirement | Change | Justification | Affected PRs/tests |
| --- | --- | --- | --- |
| Activate vLLM and SGLang profiles, then TensorRT-LLM | Instantiate only vLLM; shared production tests run once | Current rollout instruction; preserve all existing other-backend coverage | All three PRs; only `support/vllm.rs` added |
| Move native chat smoke to post-merge/nightly | Preserve existing E2E tests and lane allocation | E2E is handled separately; current rollout explicitly preserves it | All three PRs; existing `tests/serve/test_sidecar.py` unchanged |
| Old matrix vLLM assumptions (0.28.0, connection-only startup, absent media/LoRA/metadata) | Refresh against base `cdcd721e`, vLLM 0.29.0 and proto 0.3.0 | Current Control discovery, health, KV events, media, LoRA and RL are already supported | All three PRs; retained existing cases plus applicable new cases |
| Broad migration of related wire tests | Remove only five demonstrated duplicates; retain distinct metadata, management, decode and handoff assertions | Similar names do not imply equivalent coverage | DIS-2941 replacement accounting in COVERAGE.md |
| Approximate test counts | Use distinct behavioral obligations and collected/executed accounting | User explicitly rejects arbitrary count targets | All three PRs |
| Prototype task abortion as server teardown | Dedicated runtime owns tonic connections and handlers; explicit shutdown waits and joins it | Aborting the serving future alone can leave handlers alive when clients remain open | DIS-2941 `server.rs`, lifecycle teardown scenario |
| C1 describes configured vLLM identity without native discovery | Exercise the chosen base's real Control metadata/health discovery and exact published metadata | Pinned protocol 0.3.0 and implementation base expose native discovery; assumptions were stale | PR3 C1–C3, `vllm_registration_and_errors_recover_through_worker_ingress` |
| C1 includes tool/reasoning parser metadata | Assert absent parser names for vLLM; retain parser-option rejection in isolated coverage | The supported vLLM gRPC contract rejects these parser flags; no unsupported success case is invented | PR3 C1 and retained PR2 parser configuration tests |
| C10 worker discovery withdrawal was initially interpreted as all model-card deletion | Assert authoritative serving-endpoint removal, router exclusion/rejection and ordering before native cleanup/exit | The base Worker unregisters serving endpoints; file discovery retains model metadata without etcd lease expiry | PR3 `vllm_sigterm_withdraws_worker_and_releases_active_native_request` |
| Process integration plan primarily adds sidecar tests | Fix TCP pre-prologue cancellation and preserve empty-stream completion after local cancellation in shared runtime | Reproduced supported vLLM cancellation deadlock and false worker inhibition require production corrections; existing Python cancellation assertions and process pending-header assertions remain enabled | PR3 runtime `tcp/server.rs`, `egress/addressed_router.rs`, focused runtime regression, retained Python cancellation cases and process C8/C12 |
| Full frontend suggested where it owns prefill orchestration | Instantiate production PrefillRouter directly with real discovery, Worker endpoints and sidecar children | This exercises the handoff owner at the lowest sufficient boundary while preserving separately owned HTTP E2E coverage | PR3 `vllm_prefill_router_preserves_handoff_failure_and_cancellation` |
| Error matrices mention wire and process variants | Keep actual peer death and malformed native terminal at wire boundary; run representative typed setup/stream failures through Worker ingress | Distinct runtime composition assertions are added without duplicating every native fault at every layer | PR3 C5–C7 plus retained PR1 fault scenarios |
| Reuse scenarios wherever native backend contracts match | Keep five process scenarios generic over the shared fixture and a small process profile; retain native protobuf handoff assertions in the vLLM case | Startup, ingress and lifecycle contracts match; opaque prefill/decode payload schemas and transformations are native-specific. New other-backend fixtures and activation remain deferred by rollout scope | PR3 process lifecycle/cancellation scenarios, `process/vllm.rs`, `vllm_prefill_router_preserves_handoff_failure_and_cancellation` |

A runtime failure while a test unwinds triggers the server's cancellation fallback.
The explicit successful path waits for and joins the runtime; the panic fallback
cannot asynchronously join. The runner's bounded subprocess lifetime contains a
non-cooperative failure. No success claim is inferred from fallback cleanup.

## Isolated unit increment

| Original requirement | Change | Justification | Affected PRs/tests |
| --- | --- | --- | --- |
| R09–R14 propose an injected native client/stream fake across generation | Isolate ResponseState conversion; exercise actual stream consumption, read/EOF, cancellation and isolation in the shared wire suite | Lowest sufficient boundary without copying the generate loop or redesigning the concrete tonic client | DIS-2941 conformance; DIS-2942 unit responses and cleanup |
| R03 construction discovers a real engine | Extract private production `from_discovered(args, model)` and invoke it with in-memory metadata | Executes actual WorkerConfig construction without registration, sockets or downloads | DIS-2942 `engine::unit_worker` |
| R04 deadline testing through fake transport | Extract an attempt callback in the existing retry/pool policy; test shared code once with paused time | Executes the real loop and absolute deadline for both tonic versions without networking or wall-clock sleeps | DIS-2942 `common/transport.rs` and `unit_transport` |
| R20 specifies model-checksum cache fallback and rejection of redundant nvext | Assert current explicit prefixed cache identity, no checksum fallback, and acceptance only of matching redundant metadata | Chosen base changed the supported contract; restoring a fallback would invent a cache namespace | DIS-2942 `convert::unit_requests`; retained cache sockets |
| R17/R19 are described as missing checks | Fix reproduced system-only stop leakage and aborted-prefill success handoff | Supported sidecar output defects required production corrections; no assertions weakened | DIS-2942 `terminal_reasons_preserve_user_stops_and_hide_system_eos`, `prefill_success_preserves_handoff_but_cancelled_work_never_publishes_it` |
| Support D5 effective block size remains incomplete | Consume nonzero supplied effective size, preserve legacy fallback, reject overflow; retain stock vLLM 0.29 producer blocker | Published protocol exposes the authoritative value; sidecar must not reproduce engine arithmetic or claim absent producer support | DIS-2942 `model::unit_config`; native compatibility limitation in UNITS.md |
| Shared transport coverage through workspace feature unification | Explicitly enable `tonic-v14` when the runner builds the common crate alone | vLLM consumes that implementation; an isolated default-feature build otherwise omits its retained connection case and alternate error conversion assertions | DIS-2942 runner, shared unit and retained transport tests |
| PR1 wire lifecycle initially tests generation before start and repeated cleanup | PR2 moves those assertions exclusively into the isolated worker test after its replacement passes | Lowest sufficient boundary; PR1 remains independently covered, and wire retains active cleanup and post-cleanup admission | DIS-2942 `unstarted_generation_fails_and_cleanup_is_idempotent`; wire lifecycle hunk |

## Native compatibility increment

| Original requirement | Change | Justification | Affected PRs/tests |
| --- | --- | --- | --- |
| Native KV-transfer success on the selected engine | Keep the required native handoff case failing and block PR3 merge | Actual vLLM 0.29 converts numeric protobuf handoff fields to floats; NIXL `range(remote_pp_size)` fails before transfer. CPU handoff cannot substitute. | DIS-2943 `vllm_handoff_transfers_native_kv` |
| Two-GPU native handoff validation | Local reproduction uses two engines on one assigned GPU; CI retains two GPUs | Workstation has one GPU. This establishes the protocol failure, but not the required two-GPU success. | DIS-2943 native launcher; two-GPU CI pending |
| Engine-exported transfer completion metrics | Test-only worker extension observes the real NIXL completion callback | Pinned Rust frontend lacks completed-transfer metrics; positive actual bytes remain mandatory | DIS-2943 `native_probe.py`; no inference mutation |
| C12 requires native cancellation while handoff work is active | Record native transfer/work release as blocked rather than crediting CPU cancellation or response headers | Pinned vLLM fails prerequisite KV loading; headers do not establish an in-flight transfer | DIS-2943 native C12; CPU PrefillRouter cancellation remains enabled |
| Support Matrix hybrid DP and E+P+D assumptions | Distinguish supported sidecar consumption from absent pinned-engine metadata/local-size and encoder-placeholder producers | Current consumer tests pass, but stock vLLM 0.29 lacks these upstream changes | DIS-2942 metadata/rank units and retained media tests; DIS-2943 SUPPORT.md |
| C13 uses plain preprocessed requests | Disable the native frontend's automatic Qwen3 reasoning parser for the structured-output compatibility case | Pinned engine defers grammar until reasoning ends; raw prompts have no reasoning boundary. Exact JSON/schema checks remain required | DIS-2943 `vllm_native_logprobs_and_structured_output_are_compatible` |
