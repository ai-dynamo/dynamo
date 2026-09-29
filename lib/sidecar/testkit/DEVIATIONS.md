<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Deviations from the sidecar testing DEP

The planned DEP tabs remain unchanged; the explicitly requested SGLang follow-up
updates the Actual unit matrix and Bugs found tab. This report records departures from
the framework, plan, matrices and previous rollout separately. The user's
approved restack supersedes the earlier independent-three-PR instruction.
The parent #15243 is a separate alternative to #15089 on the same foundation base;
the original PR remains unchanged.

## Foundation and parent rollout

| Original requirement | Change | Justification | Affected PRs/tests |
| --- | --- | --- | --- |
| Create three new drafts independently of reference #14879 | Retain #14879 as the shared foundation, stack #15089 and #15091 on it, and supersede #15088 | Explicit later user authorization; use merge-based updates without rewriting published history | #14879 / DIS-2941, #15089 / DIS-2942, #15091 / DIS-2943; #15088 superseded |
| Earlier rollout instantiated only vLLM in the replacement foundation | Preserve the four wire families for both backends; isolated units cover common code and vLLM | Existing SGLang integration coverage remains; SGLang units are follow-up work | #14879 conformance; local common/vLLM units; SGLang follow-up |
| Use selected main `cdcd721e` and the source matrices' older vLLM assumptions | Refresh the foundation onto main `4a0547f8ba2675f14d50e48d6aec53b1bc3cf3e3`; keep vLLM 0.29.0 and protocol 0.3.0 while taking updated shared Rust dependencies | Authorized base refresh; actual Control discovery, metadata, media, LoRA and RL contracts supersede stale matrix assumptions | #14879 lock/workspace refresh; #15089 mapping and compatibility checks |
| Prototype shared checks assumed one token per response and repeated backend enrollment | Compare accumulated token prefixes with native observations, check the native model field, and register retained backends once | #14879 review identified chunk-size coupling, vacuous observation checks and omission risk; these refine the same four families | #14879 conformance and existing vLLM/SGLang adapter hooks; no extra scenario family |
| Move native chat smoke to post-merge/nightly | Preserve existing E2E tests and CI allocation | E2E implementation is separately owned and existing coverage is required | All stack boundaries; existing sidecar E2E suites |
| Broadly migrate related wire tests in the foundation | Keep all existing vLLM/SGLang Mocker cases through #15089; defer the five mapped vLLM wire replacements to #15091 | The minimal shared foundation and isolated-unit increment must retain distinct integration assertions | #15091 replacement accounting in COVERAGE.md; no five-test deletion at #15089 |
| Approximate scenario counts in the plan | Track distinct obligations, collected names and execution evidence instead of targeting a count | User requires sufficient boundaries and no duplicate or unexecuted coverage credit | All stack boundaries |
| Reuse earlier successful stack validation as completion evidence | Preserve it as historical, revision-specific evidence; collect and validate each refreshed boundary | Main and shared dependencies changed; prior green checks cover their original heads only | #14879 and #15089 refreshed local validation in UNITS.md and COVERAGE.md; new-head CI tracked separately |

## Parent vLLM unit increment

| Original requirement | Change | Justification | Affected PRs/tests |
| --- | --- | --- | --- |
| #15089 places units in shared/native case files expanded through source-group macros | Create a separate draft with all units beside production: 11 common and 71 vLLM cases; remove the unit source-group/setup macros and testkit unit tree | Explicit user request for a simpler alternative on the same base; preserve all inputs and assertions without changing #15089 | Local owner and assertion mapping in UNITS.md; ten formerly shared cases retained locally, existing LoRA lock-registry unit migrated; shared integration unchanged |
| Native fixtures lived under testkit and vLLM depended on testkit for `minimal_request()` | Move the native builders and minimal request to `vllm/src/test_fixtures.rs`; remove the testkit dev-dependency | Actual consumers are vLLM units and its retained fake-server suite; shared integration uses separate testkit helpers | No new testing feature or duplicated builder; shared integration unchanged |
| A Python runner classified, validated and exported isolated units | Remove the runner, inventory/manifest and compatibility aliases; use ordinary Cargo commands | Unit tests run with the rest of the Rust suite, without a custom execution layer | Historical runner checks remain revision-specific evidence in UNITS.md |
| Assign individual Rust unit tests to pre-merge, post-merge or nightly lanes | Remove lane markers and macros; use ordinary `#[test]` and `#[tokio::test]` attributes | User accepted review feedback to run these units whenever the normal Rust test job runs | All 82 isolated cases remain; no unit test is deferred to a later lane |
| Treat a currently unsupported option as an omitted case | Keep explicit support/rejection assertions beside the backend implementation | Preserve distinct native contracts and avoid silently omitted coverage | Typed gRPC versus native HTTP distinctions and SGLang follow-up obligations in UNITS.md |
| R09–R14 propose an injected native client/stream fake across generation | Isolate ResponseState conversion; keep real stream consumption, EOF, cancellation and isolation in the shared wire suite | Lowest sufficient boundary without copying the generation loop or redesigning its concrete tonic client | #14879 conformance; #15089 unit responses/cleanup; #15091 additional wire cases |
| R03 construction discovers a real engine | Extract private production `from_discovered(args, model)` and invoke it with in-memory metadata | Executes actual WorkerConfig construction without registration, sockets or downloads | Local vLLM worker cases |
| R04 deadline testing through fake transport | Inject the attempt callback into the existing retry/pool policy and use paused time | Exercises the production loop and one absolute deadline once for both tonic versions, without networking or wall-clock sleeps | Common transport cases in `common/src/transport/tests.rs`, registered once by the crate root |
| R20 specifies model-checksum cache fallback and rejection of redundant nvext | Check explicit prefixed cache identity, no checksum fallback, and only matching redundant metadata | The implemented contract differs from the old matrix; inventing a fallback would change cache namespaces | #15089 native request group; retained cache sockets |
| R17/R19 are described only as missing tests | Carry the reproduced system-only stop-leak and aborted-prefill handoff fixes with their regression assertions | Supported output defects need production corrections; no weakened assertion | #15089 stop-reason and failed-prefill response tests; historical reproduction in UNITS.md |
| Support D5 effective block size is incomplete | Consume a nonzero engine-supplied effective size, preserve legacy fallback and reject overflow | Engine-derived arithmetic remains upstream-owned; stock vLLM 0.29 still lacks the producer field | #15089 native model group; explicit upstream limitation in UNITS.md |
| Common transport coverage needs the `tonic-v14` implementation | The vLLM dependency enables it for combined common/vLLM and workspace runs; a common-only invocation needs an explicit feature | No redundant CI feature flag is needed | Common error/transport units and local commands in UNITS.md |
| Shared wire lifecycle checks unstarted generation and repeated cleanup for both backends | Move only the vLLM no-I/O subsection into its isolated worker test after replacement validation; retain SGLang's original subsection | vLLM units cannot replace SGLang assertions; both active-stream cleanup paths remain wire-owned | #15089 `unstarted_generation_fails_and_cleanup_is_idempotent`; #14879 retained SGLang cleanup |
| Route workspace execution and a second pre-merge CPU run through the unit runner | Restore one `cargo test --locked --all-targets` invocation; remove the additional container run and Dockerfile | Without lane filters, no package split is needed; ordinary workspace execution avoids building dependencies again with different features | Normal module names replace `unit_`; common transport retains its one-time registration; all 82 units remain pre-merge and nightly coverage remains unchanged |

## SGLang follow-up on #15243

SGLang units use the final local-module design from #15243, with direct Cargo
execution and no shared unit layer. The follow-up preserves all existing
SGLang assertions and integration suites. [UNITS.md](UNITS.md#sglang-follow-up)
records the exact new/retained split and remaining integration obligations.

The old matrix's cumulative-output and success-terminal-only rendezvous
assumptions do not match the current native protocol. Tests retain incremental
output and the existing early handoff after response headers. JSON plus regex
keeps its current forwarding behavior pending a separate policy decision.
The reproduced hidden/system stop-ID leak is corrected to match the existing
Python handler's user-stop contract, with a regression that failed before the fix.

Additional wire/process/native implementation and its departures belong to
#15091. The unit boundary does not claim those additions, actual native KV
transfer, or resolution of the pinned-engine blockers.

When #15091 is restacked, remove its obsolete positive `unit_` selector,
`--skip unit_` exclusion and any lane-based selections. Process/native export
belongs to that integration PR and must replace calls to the removed unit
runner and container tooling; those integration changes are not made here.
