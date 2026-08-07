<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance analysis notes

Working notes from building the performance analysis front door. These are **findings, not
documentation** — shipped guidance lives under `docs/fern/pages/` and in
`.agents/skills/dynamo-benchmark/`. Nothing here is rendered on the docs site.

| File | Contents |
| --- | --- |
| [benchmarking-procedure.md](benchmarking-procedure.md) | The procedure within a single benchmark: what to declare, accuracy as a gate, targets and constraints, which data source answers which question and how early each must be planned for, and how far analysis can be systematized. The most developed document here. |
| [gaps.md](gaps.md) | Six gaps found in Dynamo's performance tooling, with evidence and severity. Each entry is written to be liftable into a GitHub issue as-is. |
| [ab-testing-request-distributions.md](ab-testing-request-distributions.md) | Standalone explainer for items 3 and 4 — why the repository cannot currently tell a small performance win from noise, worked through the embedding-cache benchmark, with a concrete fix for the part that is fixable now. Start here if those items read as abstract. |
| [layered-benchmarking.md](layered-benchmarking.md) | Do we need e2e / subsystem / micro tiers, or is finely-instrumented e2e enough? The one thing e2e structurally cannot do, what the repo already has, and three verified findings — recipes capture no server-side data, the two frontend harnesses are alternatives rather than a stack, and where micro coverage stops. |
| [benchmark-coverage-report.html](benchmark-coverage-report.html) | Rendered companion to the above: what exists at each tier, and an inference-path matrix scoring every stage per tier. Open in a browser. |
| [overlap-review.md](overlap-review.md) | Whether the new front door duplicates the existing AIPerf benchmarking guide, and what to do about the two places it does. |

Findings were verified against `origin/main` @ `c574e4d7c1a`. File and line references drift;
re-check before acting on one.

These notes are tracked so the open items survive the branch. Delete them once the gaps are
filed as issues and the overlap decisions are made.
