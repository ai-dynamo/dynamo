---
name: configure-aiperf-benchmark
description: >-
  Selects and freezes a question-driven AIPerf workload, objective, load policy, and Kubernetes execution manifest for
  a successfully deployed Dynamo candidate. Use when a candidate needs performance characterization or a comparable
  measurement against a reference.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - aiperf
    - benchmarking
---

# Configure AIPerf Benchmark

<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Create a reproducible benchmark that answers the current performance question without changing the deployed candidate.
Freeze semantics only for runs used in the same direct comparison.

Read `agent-docs/rules/execution/user-workload.md`,
`agent-docs/rules/benchmarking/benchmark-isolation.md`,
`agent-docs/rules/benchmarking/comparison-uncertainty.md`,
`agent-docs/rules/benchmarking/concurrency-grid.md`,
`agent-docs/rules/benchmarking/evidence-eligibility.md`,
`agent-docs/rules/benchmarking/proxy-workload-selection.md`, and
`agent-docs/rules/benchmarking/series-boundaries.md` before selecting flags. Also read the user workload, deployment
ledger, and the AIPerf documentation matching the pinned source or runtime. Inspect matching deployment-example
`perf.yaml` files when available.

## Inputs

Require:

- exact user-workload path and SHA256;
- current `DEPLOY_ROOT`, applied DGD, deployment ledger, and successful smoke test;
- the baseline-characterization goal or approved candidate's performance question and target operating region; and
- an exact existing benchmark-plan path, SHA256, and series ID only when considering series reuse.

## Workflow

1. Require a successful smoke test and resolve the in-cluster frontend endpoint, served model, tokenizer, Kubernetes
   context, namespace, artifact collection path, and GPU count.
2. Decide whether the run is an absolute characterization or a direct comparison. For a comparison, identify the
   candidate and reference measurements needed to answer the question.
3. Reuse an existing series only when its workload and measurement semantics still answer the question. Otherwise
   assign a new series ID and require any comparison reference to be measured under the new plan.
4. Select workload input in this order:
   - user-provided Mooncake trace;
   - exact user-provided ISL/OSL and traffic controls;
   - closest compatible deployment-example trace as a `recipe_proxy`.
5. Validate a selected trace as JSONL. Record its path, SHA256, row count, timestamp range, ISL/OSL distribution,
   prefix/hash information, and any rows outside the served context limit. Do not silently filter or clip rows.
6. Choose the experiment that best exposes the behavior under investigation:
   - `trace_fidelity`: preserve timestamps and fixed-schedule behavior;
   - `static_shape`: preserve the exact synthetic ISL/OSL target;
   - `capacity`: vary concurrency or request rate over a bounded range.
7. Select the objective and metrics needed for the question:
   - with target SLOs, use goodput/good-request fraction and the stated attainment constraints;
   - without sufficient SLOs, preserve a throughput/latency/error Pareto view.
8. Configure one measured run by default. Set the measurement duration to 30 minutes or less. For request-count or
   fixed-schedule workloads, choose or validate a count or schedule that is expected to complete within that limit and
   set a bounded execution timeout. Stop rather than silently truncate or weaken a workload that cannot fit.
9. Do not enable repetitions only to obtain confidence intervals. Configure more than one run only when prior analysis
   documents why another run is necessary to resolve a consequential decision and why that value justifies the GPU
   cost. Exception: the once-per-series noise-floor pilot (n=3) required by
   `agent-docs/rules/benchmarking/comparison-uncertainty.md` is pre-authorized; it arrives as a `repeat_decision:
   necessary` from `analyze-aiperf-results` whose rationale names the series pilot, and runs as repetitions of one
   unchanged configuration. Plans stay immutable; no plan marker is involved.
10. Pin the AIPerf runtime version or source commit. Write the immutable series plan and the run-scoped
    `<DEPLOY_ROOT>/benchmark/aiperf-config.yaml` and `<DEPLOY_ROOT>/benchmark/perf.yaml`.

## Benchmark Plan

For a new series, write:

```text
<EXP_ROOT>/inputs/benchmark-plans/<series-id>.json
```

Use a filesystem-safe series ID. Never overwrite a plan. When reusing a series, use its exact existing path and SHA256
and change only endpoint address, Job identity, namespace, or artifact wiring in the run-scoped files.

The plan must identify:

- performance question, target operating region, and characterization or comparison intent;
- plan ID, benchmark-series ID, and any required reference candidates;
- user-workload path and SHA256;
- workload source: `user_trace`, `user_static`, or `recipe_proxy`;
- trace path/hash or exact synthetic distribution;
- fixed-schedule, concurrency, request-rate, request-count/duration, warmup, a per-run measurement limit of no more
  than 30 minutes, repetitions with a default of one, any approved repeat rationale, confidence policy and level when
  needed, and seed;
- endpoint type, streaming behavior, model, and tokenizer;
- target metrics, SLOs, goodput thresholds, and optimization direction;
- AIPerf source commit when available and required runtime version;
- proxy rationale and limitations when applicable.


## Manifest Rules

- `perf.yaml` is a Kubernetes Job, not the AIPerf-native config.
- Run AIPerf inside the cluster and use the target context and namespace.
- Prefer an existing compatible deployment-example Job as a starting point. Modify only the run-scoped copy.
- Enforce the 30-minute measurement limit with AIPerf workload controls. Do not rely only on a Kubernetes Job timeout.
- Ensure raw artifacts remain available after Job completion, either on a PVC or in the completed pod until copied.
- Read referenced secret names from existing manifests; never copy secret values into benchmark files.
- Do not embed a local host path that is unavailable to the benchmark pod.

Stop with a concrete configuration blocker when the performance question is missing, no defensible workload can be
selected, a required comparison cannot be made, or the planned run cannot fit within 30 minutes without changing its
semantics.
