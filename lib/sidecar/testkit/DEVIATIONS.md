<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Deviations from the sidecar testing DEP

The DEP and all five tabs remain read-only. This report records departures from
the framework, plan, matrices and previous rollout separately. The user's
approved restack supersedes the earlier independent-three-PR instruction.

## Foundation and rollout

| Original requirement | Change | Justification | Affected PRs/tests |
| --- | --- | --- | --- |
| Create three new drafts independently of reference #14879 | Retain #14879 as the shared foundation, stack #15089 and #15091 on it, and supersede #15088 | Explicit later user authorization; use merge-based updates without rewriting published history | #14879 / DIS-2941, #15089 / DIS-2942, #15091 / DIS-2943; #15088 superseded |
| Earlier rollout instantiated only vLLM in the replacement foundation | Preserve #14879's existing four shared families for both vLLM and SGLang; new backend units and integration activation are vLLM-only | Existing SGLang coverage must not disappear when changing the stack base; no new SGLang/TRT fixture migration is authorized | #14879 conformance and native adapters; #15089 common/vLLM units; #15091 additions |
| Use selected main `cdcd721e` and the source matrices' older vLLM assumptions | Refresh the foundation onto main `4a0547f8ba2675f14d50e48d6aec53b1bc3cf3e3`; keep vLLM 0.29.0 and protocol 0.3.0 while taking updated shared Rust dependencies | Authorized base refresh; actual Control discovery, metadata, media, LoRA and RL contracts supersede stale matrix assumptions | #14879 lock/workspace refresh; #15089 mapping and compatibility checks |
| Prototype shared checks assumed one token per response and repeated backend enrollment | Compare accumulated token prefixes with native observations, check the native model field, and register retained backends once | #14879 review identified chunk-size coupling, vacuous observation checks and omission risk; these refine the same four families | #14879 conformance and existing vLLM/SGLang adapter hooks; no extra scenario family |
| Move native chat smoke to post-merge/nightly | Preserve existing E2E tests and CI allocation | E2E implementation is separately owned and existing coverage is required | All stack boundaries; existing sidecar E2E suites |
| Broadly migrate related wire tests in the foundation | Keep all existing vLLM/SGLang Mocker cases through #15089; defer the five mapped vLLM wire replacements to #15091 | The minimal shared foundation and isolated-unit increment must retain distinct integration assertions | #15091 replacement accounting in COVERAGE.md; no five-test deletion at #15089 |
| Approximate scenario counts in the plan | Track distinct obligations, collected names and execution evidence instead of targeting a count | User requires sufficient boundaries and no duplicate or unexecuted coverage credit | All stack boundaries |
| Reuse earlier successful stack validation as completion evidence | Preserve it as historical, revision-specific evidence; collect and validate each refreshed boundary | Main and shared dependencies changed; prior green checks cover their original heads only | #14879 and #15089 local validation passed on their recorded new heads; #15091 candidate CPU validation passed based on `b3ab1638`; current-head CI pending; UNITS.md and COVERAGE.md |

## Isolated unit increment

| Original requirement | Change | Justification | Affected PRs/tests |
| --- | --- | --- | --- |
| R09–R14 propose an injected native client/stream fake across generation | Isolate ResponseState conversion; keep real stream consumption, EOF, cancellation and isolation in the shared wire suite | Lowest sufficient boundary without copying the generation loop or redesigning its concrete tonic client | #14879 conformance; #15089 unit responses/cleanup; #15091 additional wire cases |
| R03 construction discovers a real engine | Extract private production `from_discovered(args, model)` and invoke it with in-memory metadata | Executes actual WorkerConfig construction without registration, sockets or downloads | #15089 `engine::unit_worker` |
| R04 deadline testing through fake transport | Inject the attempt callback into the existing retry/pool policy and use paused time | Exercises the production loop and one absolute deadline once for both tonic versions, without networking or wall-clock sleeps | #15089 `common/transport.rs` and `unit_transport` |
| R20 specifies model-checksum cache fallback and rejection of redundant nvext | Check explicit prefixed cache identity, no checksum fallback, and only matching redundant metadata | The implemented contract differs from the old matrix; inventing a fallback would change cache namespaces | #15089 `convert::unit_requests`; retained cache sockets |
| R17/R19 are described only as missing tests | Carry the reproduced system-only stop-leak and aborted-prefill handoff fixes with their regression assertions | Supported output defects need production corrections; no weakened assertion | #15089 stop-reason and failed-prefill response tests; historical reproduction in UNITS.md |
| Support D5 effective block size is incomplete | Consume a nonzero engine-supplied effective size, preserve legacy fallback and reject overflow | Engine-derived arithmetic remains upstream-owned; stock vLLM 0.29 still lacks the producer field | #15089 `model::unit_config`; explicit upstream limitation in UNITS.md |
| Common transport coverage relies on workspace feature unification | Enable `tonic-v14` explicitly for isolated common-crate execution | vLLM uses that implementation; its tests must not depend on unrelated crates enabling the feature | #15089 isolated runner and shared error/transport units |
| Shared wire lifecycle checks unstarted generation and repeated cleanup for both backends | Move only the vLLM no-I/O subsection into its isolated worker test after replacement validation; retain SGLang's original subsection | vLLM units cannot replace SGLang assertions; both active-stream cleanup paths remain wire-owned | #15089 `unstarted_generation_fails_and_cleanup_is_idempotent`; #14879 retained SGLang cleanup |
| Use one runner for isolated and retained wire execution | The #15089 runner selects only common/vLLM isolated modules; retained foundation and both Mocker suites keep their workspace execution | Keeps the unit boundary explicit and common code once, while preserving existing SGLang coverage without new activation | #15089 runner; separate preservation commands in COVERAGE.md |

## Additional wire and process integration (#15091)

| Original requirement | Change | Justification | Affected PRs/tests |
| --- | --- | --- | --- |
| Prototype task abortion as server teardown | Dedicated runtime owns tonic connections and handlers; explicit shutdown waits and joins it | Aborting the serving future alone can leave handlers alive when clients remain open | DIS-2943 `server.rs`, lifecycle teardown scenario |
| C1 describes configured vLLM identity without native discovery | Exercise the chosen base's real Control metadata/health discovery and exact published metadata | Pinned protocol 0.3.0 and implementation base expose native discovery; assumptions were stale | #15091 C1–C3, `vllm_registration_and_errors_recover_through_worker_ingress` |
| C1 includes tool/reasoning parser metadata | Assert absent parser names for vLLM; retain parser-option rejection in isolated coverage | The supported vLLM gRPC contract rejects these parser flags; no unsupported success case is invented | #15091 C1 and retained #15089 parser configuration tests |
| C10 worker discovery withdrawal was initially interpreted as all model-card deletion | Assert authoritative serving-endpoint removal, router exclusion/rejection and ordering before native cleanup/exit | The base Worker unregisters serving endpoints; file discovery retains model metadata without etcd lease expiry | #15091 `vllm_sigterm_withdraws_worker_and_releases_active_native_request` |
| Process integration plan primarily adds sidecar tests | Fix TCP pre-prologue cancellation and preserve empty-stream completion after local cancellation in shared runtime | Reproduced supported vLLM cancellation deadlock and false worker inhibition require production corrections; existing Python cancellation assertions and process pending-header assertions remain enabled | #15091 runtime `tcp/server.rs`, `egress/addressed_router.rs`, focused runtime regression, retained Python cancellation cases and process C8/C12 |
| Preserve existing shared-runtime consumers while enabling sidecar cancellation | Keep guarded media dispatches' original pre-prologue cancellation behavior | Early cancellation otherwise releases registered frontend media before remote use; a historically executed router/TCP regression catches the release. Guard ownership enables deferral only on the owning dispatch; ordinary sidecar cancellation is unchanged | #15091 `network.rs`, `tcp/server.rs`, `egress/addressed_router.rs`, `first_response_guard_defers_tcp_cancellation_until_prologue` |
| Full frontend suggested where it owns prefill orchestration | Instantiate production PrefillRouter directly with real discovery, Worker endpoints and sidecar children | This exercises the handoff owner at the lowest sufficient boundary while preserving separately owned HTTP E2E coverage | #15091 `vllm_prefill_router_preserves_handoff_failure_and_cancellation` |
| Error matrices mention wire and process variants | Keep actual peer death and malformed native terminal at wire boundary; run representative typed setup/stream failures through Worker ingress | Distinct runtime composition assertions are added without duplicating every native fault at every layer | #15091 C5–C7 plus retained foundation faults and #15091 wire additions |
| Reuse scenarios wherever native backend contracts match | Keep five process scenarios generic over the shared fixture and a small process profile; retain native protobuf handoff assertions in the vLLM case | Startup, ingress and lifecycle contracts match; opaque prefill/decode payload schemas and transformations are native-specific. New other-backend fixtures and activation remain deferred by rollout scope | #15091 process lifecycle/cancellation scenarios, `process/vllm.rs`, `vllm_prefill_router_preserves_handoff_failure_and_cancellation` |
| Superseded #15088 owns nine vLLM wire cases | Carry those obligations into #15091 by extending the four foundation vLLM families, without a second baseline vLLM enrollment | User-approved stack preserves #14879's scope and SGLang coverage while keeping distinct added integration checks | #15091 conformance; four retained SGLang foundation cases and four retained SGLang Mocker cases |
| Shared cancellation combines deterministic isolation and active engine-work release | Keep paused two-request wire isolation separate from active Mocker scheduler checks and actual native-engine checks | A response checkpoint alone does not establish scheduler activity; CPU scheduler observations do not establish engine release | #15091 cancellation wire/process/native cases |

A runtime failure while a test unwinds triggers server cancellation fallback.
The explicit successful path joins the owned runtime; the panic fallback cannot
asynchronously join. Bounded subprocess lifetime contains non-cooperative
failure. Fallback cleanup alone does not establish a successful teardown.

## Native compatibility increment

| Original requirement | Change | Justification | Affected PRs/tests |
| --- | --- | --- | --- |
| Native KV-transfer success on the selected engine | Keep the required native handoff case failing and block #15091 merge | Actual vLLM 0.29 converts numeric protobuf handoff fields to floats; NIXL `range(remote_pp_size)` fails before transfer. CPU handoff cannot substitute. | DIS-2943 `vllm_handoff_transfers_native_kv` |
| Two-GPU native handoff validation | Historical and refreshed local reproductions used two engines on one assigned GPU; CI retains two GPUs | Workstation has one GPU. This establishes the protocol failure, but not the required two-GPU success. | DIS-2943 native launcher; two-GPU CI pending |
| Engine-exported transfer completion metrics | Test-only worker extension observes the real NIXL completion callback | Pinned Rust frontend lacks completed-transfer metrics; positive actual bytes remain mandatory | DIS-2943 `native_probe.py`; no inference mutation |
| C12 requires native cancellation while handoff work is active | Record native transfer/work release as blocked rather than crediting CPU cancellation or response headers | Pinned vLLM fails prerequisite KV loading; headers do not establish an in-flight transfer | DIS-2943 native C12; CPU PrefillRouter cancellation remains enabled |
| Support Matrix hybrid DP and E+P+D assumptions | Distinguish supported sidecar consumption from absent pinned-engine metadata/local-size and encoder-placeholder producers | Refreshed unit/retained consumer checks passed at `b3ab1638`; stock vLLM 0.29 still lacks these upstream producers | DIS-2942 metadata/rank units and retained media tests; DIS-2943 SUPPORT.md |
| C13 uses plain preprocessed requests | Disable the native frontend's automatic Qwen3 reasoning parser for the structured-output compatibility case | Pinned engine defers grammar until reasoning ends; raw prompts have no reasoning boundary. Exact JSON/schema checks remain required | DIS-2943 `vllm_native_logprobs_and_structured_output_are_compatible` |

All earlier failure reproductions and execution results retain their original
revision attribution. Fresh #15089 local evidence belongs to
`b3ab1638513e255828acddc40f045d069ed6bc33`: 62 isolated container cases, 102 total
common/vLLM library cases, eight foundation and eight retained Mocker cases
passed, with formatting/Clippy/pre-commit/ownership checks. Those overlapping
selections are not additive; UNITS.md owns their detail.

The #15091 candidate based on that unit commit passed all 124 disjoint CPU
cases: 62 units, 56 wire and six process, with zero ignored. The vLLM-only
selector separately collected 116 cases. Refreshed runtime filters passed two
pre-prologue cases, one guard case and 82 serial TCP cases; these selections
overlap. All-target integration Clippy and formatting passed. COVERAGE.md and
PROCESS.md record the commands and execution logs. No final integration SHA is
attributed to this candidate evidence. Refreshed native execution collected
three cases: compatibility and cancellation passed; handoff reproduced the
upstream float-conversion failure. The two-GPU success topology and native
transfer-time cancellation remain unexecuted. Final current-head CI remains
pending; CPU results do not certify native transfer.
