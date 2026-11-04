<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Matched-input hotspot result and statistics replay

This experiment compares upstream `2141a9690d670440d729e99771afb2fb53e79284` (B) with the native-tracker revision `80fa97214acd789460baf93a07daaceb31c7f3b4` (P). P enables observation and the experimental `progress_group` picker together; its complete costs are included.

These GPU measurements describe those historical revisions. The later adaptation to main `8b4ad6a1b041a249952586c519ca657f67522cef` changes the selection and stream-processing integration. Its validation is tracked separately; these results do not establish GPU performance for the adapted source.

### Fixed experiment

Two physical RTX 4090s, one worker per GPU, Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, vLLM 0.29.0+cu129, PyTorch 2.13.0+cu126, driver 595.71.05. Both arms enable native active/output-block and prefill tracking. The experiment uses aggregated serving, max sequences 8, 64-token blocks and a 32770-token batch limit.

Each group has two children sharing an exact 16384-token prefix and distinct one-token suffixes, with eight output tokens per child. Eight 32-token background requests run on the warm source; the alternative has none. The background total output budget is 16 or 128 per request, with ignored EOS and natural completion. These budgets are not remaining tokens at selection time. Children arrive at release+200/225 ms with at most 10 ms lateness; complete setup and drain fit each fixed 8-second slot.

Every arm restarts its backend/APC and verifies cold prewarm followed by full-hit natural parent seed/probe. All thirteen request bodies in a group are matched across B/P; background content is independent of generated worker IDs. One pre-issued 32-bit sequence determines adjacent BP/PB order, with alternating physical mapping. Two predetermined 16-pair waves each contain 256 formal and 64 warmup groups. No selected, replaced or extended samples were used.

### Results

The primary metric is the pooled group maximum of the two children's true first-content TTFT, expressed as P95. Every arm/budget has 128 formal groups.

| Background total output | B P95, ms | P P95, ms | Change | Approximate P/B interval |
| --- | ---: | ---: | ---: | --- |
| 16 | 332.637 | 332.922 | +0.0856% | [0.947860, 1.087420] |
| 128 | 2957.325 | 408.747 | -86.1785% | [0.133726, 0.140225] |

The fixed decision is `KEEP`: short16's approximate upper bound is +8.742%, below the predeclared +10% tolerance, and long128's upper ratio is below 1. The short interval includes improvement and cost; its mean changes from 283.838 to 286.172 ms. This is not a zero-overhead or exact-equivalence result.

B keeps both children on the warm source, computing 1/1 input tokens beyond cache in all 256 formal groups. P does the same for short16. For long128, all 128 P groups move together to the alternative: computed input tokens are 16385/1 and cached tokens are 0/16384. Input minus reported cached tokens defines this accounting; it is not GPU kernel time. Cold-prefix cost is retained in TTFT. No group splits or behavior exceptions were observed, including warmups.

### Replay once, without a GPU

Run with Python 3.12 from the repository root; only the standard library is used:

```bash
python3 examples/router/custom-policy-example/replay_hotspot_progress.py --check
```

The result is `PASS_FIXED_STATISTICS_REPLAY`. To inspect all 640 retained group records, including 128 warmups:

```bash
python3 examples/router/custom-policy-example/replay_hotspot_progress.py --emit-csv
```

The compressed CSV has synthetic group IDs, wave/pair/arm/phase, background budget, physical source index, raw group-max TTFT in nanoseconds and the audit outcome. It contains no prompts, machine addresses, access credentials or worker identities. The script checks data digests and reconstructs the frozen statistics byte for byte. It resamples whole pairs, stratified by wave and mapping: four strata of eight pairs, 20000 draws, seed 2026103001, two primary metrics and four tails with probability 0.05/4 per tail. Warmups remain in the export and are excluded from formal inference as declared.

This is an analysis replay, not a GPU deployment or independent raw-log audit. The original 32 durable pair archives were HTTP/native byte-replayed, with 64 audits of each kind, 8448 distinct identities of each kind and 320 matched thirteen-body cells. Full machine-specific launch/trace artifacts are retained separately; a portable GPU rerun recipe remains to be agreed.

### Scope

The bootstrap is approximate, with no finite-sample, conditional coverage or power guarantee. The accepted metric pools the two waves and hotspot sources; it is not a per-source safety guarantee. The complete replay output retains all per-wave/source median/P95 point estimates.

The result covers this two-worker, exact-prefix, calibrated workload with tracing enabled. It establishes neither short24/32 safety, a general migration boundary, default-off equivalence, production contention cost, throughput improvement nor energy reduction. Private selection-time progress and component lock/kernel timings were not exposed. Both new arms use the same dependencies; the aisimulate 0.13.0.dev2 prebuilt artifact was fixed, but its old-cloud binary digest was not recorded. Earlier cohorts and the interrupted 328-group attempt are excluded, and their original inconclusive conclusions remain unchanged.
