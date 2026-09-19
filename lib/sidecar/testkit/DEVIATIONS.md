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
