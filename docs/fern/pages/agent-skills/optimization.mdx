---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Performance Optimization Skills
subtitle: The evidence-driven loop that benchmarks a confirmed baseline and challenges it with candidates.
---

These skills form the optimization workflow: capture a workload contract, benchmark a confirmed
baseline, then challenge it with one candidate at a time until the Service Level Objectives (SLOs)
are met or the budget runs out. The loop also uses
[`deploy-dynamo-recipe`](https://github.com/ai-dynamo/dynamo/tree/main/.agents/skills/deploy-dynamo-recipe),
listed on the [Deployment and Operations](deployment.mdx) page, to deploy the confirmed baseline and
each approved candidate.

A prompt that reaches them: "Optimize this deployment for output tokens per second per user under a
200 ms time-to-first-token SLO. Budget 8 GPU-hours and stop after three failed deployments."

<SearchableTable placeholder="Search optimization skills...">

| Phase | Skill | Use for |
| --- | --- | --- |
| Workload contract | [`synthesize-user-workload`](https://github.com/ai-dynamo/dynamo/tree/main/.agents/skills/synthesize-user-workload) | Capture the workload requirements and immutable user-provided or confirmed baseline deployment. |
| Baseline authoring | [`author-baseline-dgd`](https://github.com/ai-dynamo/dynamo/tree/main/.agents/skills/author-baseline-dgd) | Draft a baseline when no catalog recipe matches, then present it for explicit user confirmation. |
| Benchmark configuration | [`configure-aiperf-benchmark`](https://github.com/ai-dynamo/dynamo/tree/main/.agents/skills/configure-aiperf-benchmark) | Freeze the AIPerf workload, objective, load policy, and Kubernetes execution manifest. |
| Benchmark execution | [`run-aiperf-benchmark`](https://github.com/ai-dynamo/dynamo/tree/main/.agents/skills/run-aiperf-benchmark) | Launch, monitor, debug, and collect one run-scoped AIPerf benchmark. |
| Results analysis | [`analyze-aiperf-results`](https://github.com/ai-dynamo/dynamo/tree/main/.agents/skills/analyze-aiperf-results) | Validate and normalize results, evaluate SLOs, and compare compatible candidates. |
| Proposal selection | [`consult-perf-knowledge`](https://github.com/ai-dynamo/dynamo/tree/main/.agents/skills/consult-perf-knowledge) | Select one evidence-backed optimization proposal from the performance rules and engine guidance. |
| Candidate creation | [`create-optimization-hypothesis`](https://github.com/ai-dynamo/dynamo/tree/main/.agents/skills/create-optimization-hypothesis) | Apply only the selected change to create a challenger-ready deployment draft. |
| Adversarial review | [`perform-adversarial-review`](https://github.com/ai-dynamo/dynamo/tree/main/.agents/skills/perform-adversarial-review) | Review a candidate for correctness, comparability, feasibility, attribution, and worthwhile GPU spend. |

</SearchableTable>

See the
[optimization loop](https://github.com/ai-dynamo/dynamo/blob/main/agent-docs/guides/optimization/optimize-loop.md)
for the full sequence and the
[evidence rules](https://github.com/ai-dynamo/dynamo/tree/main/agent-docs/rules) for benchmark
validity requirements.
