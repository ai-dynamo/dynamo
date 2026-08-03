---
name: optimize-loop
description: >-
  Run the Dynamo optimization loop from the initial user interview and user-provided DGD through workload synthesis,
  deployment, AIPerf evaluation, hypothesis review, and final reproducible recommendation.
agents:
  - user-interviewer
  - recipe-deployer
  - perf-analyzer
  - hypothesis-generator
  - hypothesis-challenger
docs:
  - agent-docs/references/definitions.md
rules:
  - agent-docs/rules/execution/deployment.md
  - agent-docs/rules/execution/logging.md
  - agent-docs/rules/execution/run-artifacts.md
  - agent-docs/rules/execution/user-workload.md
  - agent-docs/rules/benchmarking/benchmark-isolation.md
  - agent-docs/rules/benchmarking/comparison-uncertainty.md
  - agent-docs/rules/benchmarking/concurrency-grid.md
  - agent-docs/rules/benchmarking/evidence-eligibility.md
  - agent-docs/rules/benchmarking/proxy-workload-selection.md
  - agent-docs/rules/benchmarking/result-storage.md
  - agent-docs/rules/benchmarking/series-boundaries.md
  - agent-docs/rules/optimization/evidence-before-spend.md
  - agent-docs/rules/optimization/one-variable.md
  - agent-docs/rules/verification/config-engagement.md
  - agent-docs/rules/verification/implausible-speedup.md
  - agent-docs/rules/verification/overlap.md
  - agent-docs/rules/verification/stack-verdict.md
---

# Optimize Loop

Use this workflow for an end-to-end Dynamo configuration optimization job. The user supplies the baseline DGD;
`user-interviewer` captures it and hands it directly to `recipe-deployer`. There is no recipe-discovery or
recipe-selection step in this workflow.

When using Codex multi-agent mode, dispatch registered roles through `.codex/config.toml`. Each launcher must read and
follow its corresponding `agents/<role>/AGENTS.md` contract.

## 1. Interview, Capture The DGD, And Synthesize The Workload

Immediately dispatch `user-interviewer` with the user's first optimization message and any supplied attachments. It
invokes `synthesize-user-workload`, asks only for unresolved blocking facts, establishes `EXP_ROOT`, and writes:

```text
<EXP_ROOT>/user_workload.yaml
<EXP_ROOT>/inputs/user_provided_dgd.yaml
```

Do not invoke another specialized role until both files are valid. Preserve both exact paths and SHA256 values, and
pass both to `recipe-deployer`. Pass the workload path and SHA256 to every later role. If the user supplied a workload
file, validate and normalize it into the canonical path instead of creating a second contract. If
`user-interviewer` returns blocking questions, relay them to the user and dispatch the same role again with the
answers; do not advance the workflow meanwhile.

## 2. Validate The Baseline Handoff

Require the exact `EXP_ROOT`, `user_workload.yaml` path and SHA256, `user_provided_dgd.yaml` path and SHA256, and
zero-based iteration `0`. Confirm that the user-provided DGD's model, framework, hardware, precision, and topology do
not contradict the user workload. Do not edit, replace, or select an alternative DGD.

## 3. Deploy The Candidate

Give the exact assigned DGD path and SHA256, `user_workload.yaml` path and SHA256, iteration, and previous
`DEPLOY_ROOT` when applicable to `recipe-deployer`. For iteration 0, the assigned DGD is the immutable
`user_provided_dgd.yaml`; later iterations use the exact challenger-approved draft. The deployer creates:

```text
<EXP_ROOT>/artifacts/deploy-iter-<NNN>/
```

Continue only when `deployment_ledger.json` is complete and `smoke_test_artifact.json` reports `success: 1`.
Functional deployment repair is owned by `recipe-deployer`; do not benchmark a failed deployment or change benchmark
semantics to hide a deployment failure.

## 4. Configure, Run, And Analyze The Benchmark

Give the successful `DEPLOY_ROOT`, exact `user_workload.yaml` path and SHA256, and current performance question and
target operating region to `perf-analyzer`. For iteration 0, use a baseline-characterization question. For later
iterations, use the question approved with the candidate.

- Select or create the benchmark series that best answers the question. Reuse a plan only when it remains fit; write
  each new immutable plan under `EXP_ROOT/inputs/benchmark-plans/`.
- Invoke the three performance skills in order: configure, run, analyze.
- `analyze-aiperf-results` audits and normalizes the raw evidence before any SLO or comparison analysis.
- When `benchmark_audit.json` sets `next_action` to `rerun_benchmark`, return its blockers to
  `run-aiperf-benchmark`, rerun the active series unchanged, and invoke `analyze-aiperf-results` again.
- When a valid `performance_analysis.json` sets `repeat_decision` to `necessary`, pass its rationale and the decision it
  expects to resolve to `run-aiperf-benchmark`, run exactly one additional same-series repetition, and analyze the
  combined evidence again. Each further repetition requires a new `necessary` decision after reanalysis.
- Running benchmarks costs valuable GPU time. Rerun only when necessary.
- Continue to hypothesis generation only when the audit is `valid` or `valid_with_recovery` and both
  `benchmark_summary.json` and `performance_analysis.json` exist.

The earliest valid result in a series is its baseline. Make direct comparisons only with valid same-series references;
if none exists, report absolute performance. If the active plan requires a specific reference, arrange that measurement
before reporting a delta. Include all valid same-series runs in that series' history.

## 5. Generate And Challenge The Next Change

Give the current deployment ledger, successful `applied_manifests/deploy.yaml`, benchmark audit, summary, performance
analysis, active plan path, SHA256, and series ID, user workload, and relevant history to `hypothesis-generator`.

The generator writes under the current analyzed iteration:

```text
<DEPLOY_ROOT>/next-candidate/
|-- knowledge-consult.md
`-- deploy-draft.yaml
```

`knowledge-consult.md` is required. Create `deploy-draft.yaml`
only for a materialized proposal. The proposal must be backed by at least three distinct evidence categories,
including AIPerf profiler data, and change one independently testable knob. A coupled bundle is allowed only when
required for one functional mechanism or supported by evidence of an interaction.

Give the unchanged consultation and draft to `hypothesis-challenger`. The challenger appends its hash-bound review to
`EXP_ROOT/analysis/challenger-reviews.jsonl` and returns exactly one verdict:

- `approve`: send the existing draft path, SHA256, and review ID to `recipe-deployer`, and preserve the approved
  performance question and target operating region for `perf-analyzer`;
- `revise` or `reject`: return the objections and minimal required follow-up to `hypothesis-generator`.

The challenger must not edit the draft or create a replacement hypothesis. Do not spend GPU time on an unapproved
candidate.

## 6. Iterate Or Stop

After approval, assign the exact approved draft as iteration `<NNN + 1>` and return to deployment.
`recipe-deployer` alone creates the next `DEPLOY_ROOT` for iteration `<NNN + 1>` and retires only the previous DGD.
Keep every previous manifest, consultation, review, and benchmark artifact unchanged.

After deployment, choose the benchmark based on the approved performance question. It may reuse an applicable series
or create a new one. Never claim a direct gain across series.

Stop when the user constraints are met, no useful optimization ideas remain, the user-specified budget is exhausted,
or the generator returns `no-proposal`.

## 7. Finalize

Recommend the best valid candidate for the target objective, not automatically the most recent iteration. Write the
final configuration to `EXP_ROOT/final/recommended_config.md`, reproduction commands to
`EXP_ROOT/final/reproduced_commands.sh`, and limitations to `EXP_ROOT/final/known_limitations.md`. Include paths to the
user workload, original user-provided DGD, applied manifests, deployment ledgers, applicable benchmark plans, audits,
summaries, performance analyses, comparison histories, and raw AIPerf evidence. Do not call a proxy workload a
validated production result.
