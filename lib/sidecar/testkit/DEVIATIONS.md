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
| Earlier rollout instantiated only vLLM in the replacement foundation | Preserve the four wire families for both backends; establish shared unit bodies with vLLM activation first | Existing SGLang coverage remains; the approved follow-up supplies SGLang unit adapters without copying shared scenarios | #14879 conformance; #15089 shared/common/native units; SGLang follow-up |
| Use selected main `cdcd721e` and the source matrices' older vLLM assumptions | Refresh the foundation onto main `4a0547f8ba2675f14d50e48d6aec53b1bc3cf3e3`; keep vLLM 0.29.0 and protocol 0.3.0 while taking updated shared Rust dependencies | Authorized base refresh; actual Control discovery, metadata, media, LoRA and RL contracts supersede stale matrix assumptions | #14879 lock/workspace refresh; #15089 mapping and compatibility checks |
| Prototype shared checks assumed one token per response and repeated backend enrollment | Compare accumulated token prefixes with native observations, check the native model field, and register retained backends once | #14879 review identified chunk-size coupling, vacuous observation checks and omission risk; these refine the same four families | #14879 conformance and existing vLLM/SGLang adapter hooks; no extra scenario family |
| Move native chat smoke to post-merge/nightly | Preserve existing E2E tests and CI allocation | E2E implementation is separately owned and existing coverage is required | All stack boundaries; existing sidecar E2E suites |
| Broadly migrate related wire tests in the foundation | Keep all existing vLLM/SGLang Mocker cases through #15089; defer the five mapped vLLM wire replacements to #15091 | The minimal shared foundation and isolated-unit increment must retain distinct integration assertions | #15091 replacement accounting in COVERAGE.md; no five-test deletion at #15089 |
| Approximate scenario counts in the plan | Track distinct obligations, collected names and execution evidence instead of targeting a count | User requires sufficient boundaries and no duplicate or unexecuted coverage credit | All stack boundaries |
| Reuse earlier successful stack validation as completion evidence | Preserve it as historical, revision-specific evidence; collect and validate each refreshed boundary | Main and shared dependencies changed; prior green checks cover their original heads only | #14879 and #15089 refreshed local validation in UNITS.md and COVERAGE.md; new-head CI tracked separately |

## Isolated unit increment

| Original requirement | Change | Justification | Affected PRs/tests |
| --- | --- | --- | --- |
| Prior #15089 kept new backend unit bodies entirely vLLM-specific | Extract shared request, response, model and worker scenarios with thin native adapters | User requires #15089 to establish the expandable unit framework; native protocol regressions remain explicit | Shared source layout and assertion mapping in UNITS.md; SGLang activation deferred |
| Runner levels selected the same unit set and individual tests had no lanes | Each governed test declares its earliest lane; selection is cumulative and suite/backend/lane are independent | User requires test-level pre-merge, post-merge and nightly assignment before future integration additions | `sidecar_test!`, compiled inventory/export and workflow lane selection; every current unit remains pre-merge |
| Treat a currently unsupported option as an omitted shared case | Test supported preservation or explicit rejection; record unresolved parity gaps rather than silently passing | Prevent backend adapters from hiding lost canonical controls or weakening coverage to the intersection | Request capabilities; typed gRPC versus native HTTP distinctions and SGLang follow-up obligations in UNITS.md |
| R09–R14 propose an injected native client/stream fake across generation | Isolate ResponseState conversion; keep real stream consumption, EOF, cancellation and isolation in the shared wire suite | Lowest sufficient boundary without copying the generation loop or redesigning its concrete tonic client | #14879 conformance; #15089 unit responses/cleanup; #15091 additional wire cases |
| R03 construction discovers a real engine | Extract private production `from_discovered(args, model)` and invoke it with in-memory metadata | Executes actual WorkerConfig construction without registration, sockets or downloads | #15089 `engine::unit_worker` |
| R04 deadline testing through fake transport | Inject the attempt callback into the existing retry/pool policy and use paused time | Exercises the production loop and one absolute deadline once for both tonic versions, without networking or wall-clock sleeps | #15089 `common/transport.rs` and `unit_transport` |
| R20 specifies model-checksum cache fallback and rejection of redundant nvext | Check explicit prefixed cache identity, no checksum fallback, and only matching redundant metadata | The implemented contract differs from the old matrix; inventing a fallback would change cache namespaces | #15089 `convert::unit_requests`; retained cache sockets |
| R17/R19 are described only as missing tests | Carry the reproduced system-only stop-leak and aborted-prefill handoff fixes with their regression assertions | Supported output defects need production corrections; no weakened assertion | #15089 stop-reason and failed-prefill response tests; historical reproduction in UNITS.md |
| Support D5 effective block size is incomplete | Consume a nonzero engine-supplied effective size, preserve legacy fallback and reject overflow | Engine-derived arithmetic remains upstream-owned; stock vLLM 0.29 still lacks the producer field | #15089 `model::unit_config`; explicit upstream limitation in UNITS.md |
| Common transport coverage relies on workspace feature unification | Enable `tonic-v14` explicitly for isolated common-crate execution | vLLM uses that implementation; its tests must not depend on unrelated crates enabling the feature | #15089 isolated runner and shared error/transport units |
| Shared wire lifecycle checks unstarted generation and repeated cleanup for both backends | Move only the vLLM no-I/O subsection into its isolated worker test after replacement validation; retain SGLang's original subsection | vLLM units cannot replace SGLang assertions; both active-stream cleanup paths remain wire-owned | #15089 `unstarted_generation_fails_and_cleanup_is_idempotent`; #14879 retained SGLang cleanup |
| Use one runner for isolated and retained wire execution | The runner selects isolated units by suite, backend and cumulative lane using compiled names; retained wire suites keep their existing workflow ownership | Common code runs once; exported inventories are checked against binaries; missing lane metadata cannot silently omit governed units | #15089 runner; separate preservation commands in COVERAGE.md |

Additional wire/process/native implementation and its departures belong to
#15091. The unit boundary does not claim those additions, actual native KV
transfer, or resolution of the pinned-engine blockers.
