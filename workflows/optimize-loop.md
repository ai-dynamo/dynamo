---
name: optimize-loop
description: >-
  Run the Dynamo recipe optimization loop from selected recipe through deployment, AIPerf evaluation, comparison,
  hypothesis review, and final reproducible recommendation.
agents:
  - recipe-explorer
  - recipe-deployer
  - perf-analyzer
  - hypothesis-generator
  - hypothesis-challenger
skills:
  - learn-from-experience
docs:
  - docs/reference/target-workload.md
  - docs/reference/run-artifacts.md
  - docs/guides/benchmark_tooling.md
  - docs/guides/result_storage.md
rules:
  - docs/rules/execution.md
  - docs/rules/logging.md
  - docs/rules/optimization.md
  - docs/rules/benchmarking.md
---

# Optimize Loop

Use this workflow for an end-to-end Dynamo configuration optimization job. Cluster setup is out of scope.

## 1. Establish The Run

Create `EXP_ROOT`, preserve the user-provided `target_workload.yaml`, inspect recipe and cluster evidence, and select a
baseline candidate. Record why the candidate matches the requested model, hardware, workload, and preferences.

## 2. Deploy The Candidate

Give the selected run-scoped DGD to `recipe-deployer`. Continue only when `smoke_test_artifact.json` reports success.
Functional deployment repair is owned by `recipe-deployer`; benchmark semantics must not be changed to hide a
deployment failure.

## 3. Configure And Run The Benchmark

Give the successful `DEPLOY_ROOT` to `perf-analyzer`.

- At iteration 0, create and freeze `EXP_ROOT/inputs/benchmark_plan.json`.
- At later iterations, reuse that plan and render only deployment-specific endpoint or Kubernetes wiring.
- Configure, run, audit, and analyze AIPerf in that order.
- When an invalid audit sets `next_action` to `rerun_benchmark`, return its blockers to `run-aiperf-benchmark`, rerun
  the same frozen workload, and audit the rerun before analysis.
- Stop performance interpretation when an invalid audit sets `next_action` to `stop`.

Iteration 0 becomes the original baseline. Every later valid iteration must be compared with the original baseline,
the previous valid iteration, the best prior valid result, and the full valid same-series history.
The history must include all prior valid same-series runs, not only winning candidates.

## 4. Generate And Challenge The Next Change

Give `performance_analysis.json`, the current successful manifests, and the candidate history to
`hypothesis-generator`. AIPerf evidence may identify a performance symptom, but it does not prove a kernel- or
engine-level root cause.

Give each proposed change to `hypothesis-challenger`. The challenger must reject non-comparable workload changes,
unsupported causal claims, duplicate attempts, and changes that violate the target constraints. The accepted next
candidate should change one knob family unless multiple changes are required for functionality.

## 5. Iterate Or Stop

Return to deployment for the accepted candidate. Keep previous manifests and benchmark evidence unchanged.

Stop when the user constraints are met and no justified candidate remains, the optimization budget is exhausted, a
concrete blocker prevents further progress, or the user redirects the run. Do not call a proxy workload a validated
production result.

## 6. Finalize And Learn

Produce the final recipe, applied manifests/commands, benchmark evidence, comparison history, rationale, attempted
alternatives, and limitations. Then invoke `learn-from-experience` once to write a run-local review of successful,
failed, wasteful, and reusable parts of the process.
