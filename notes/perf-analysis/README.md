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
| [gaps.md](gaps.md) | Six gaps found in Dynamo's performance tooling, with evidence and severity. Each entry is written to be liftable into a GitHub issue as-is. |
| [ab-testing-request-distributions.md](ab-testing-request-distributions.md) | Standalone explainer for gap 3 — why the repository cannot currently tell a small performance win from noise, worked through the embedding-cache benchmark, with a concrete four-step fix. Start here if gap 3 reads as abstract. |
| [overlap-review.md](overlap-review.md) | Whether the new front door duplicates the existing AIPerf benchmarking guide, and what to do about the two places it does. |

Findings were verified against `origin/main` @ `c574e4d7c1a`. File and line references drift;
re-check before acting on one.

These notes are tracked so the open items survive the branch. Delete them once the gaps are
filed as issues and the overlap decisions are made.
