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
