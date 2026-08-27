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

<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Use this workflow for an end-to-end Dynamo configuration optimization job. The baseline DGD comes from the
interview's baseline-source ladder (`agents/user-interviewer/AGENTS.md`): supplied by the user, or a recipe or
authored draft the user explicitly confirmed. `user-interviewer` captures the confirmed baseline and hands it
directly to `recipe-deployer`. Selection and authoring happen only at interview time with user confirmation; the
LOOP itself has no recipe-discovery or recipe-selection step.

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

Require the exact `EXP_ROOT`, `user_workload.yaml` path and SHA256, `user_provided_dgd.yaml` path and SHA256,
`deployment.origin` (with `origin_source` for non-user origins), and
zero-based iteration `0`. Confirm that the user-provided DGD's model, framework, hardware, precision, and topology do
not contradict the user workload. Do not edit, replace, or select an alternative DGD.

## 3. Deploy The Candidate

Give the exact assigned DGD path and SHA256, `user_workload.yaml` path and SHA256, iteration, and previous
`DEPLOY_ROOT` when applicable to `recipe-deployer`. For iteration 0, the assigned DGD is the immutable
`user_provided_dgd.yaml`. No role selects or substitutes a baseline. When the user's DGD cannot run on the target as
provided — it targets different hardware, checkpoints, or fabric — the deployer records the blocking
incompatibilities in the deployment ledger and returns them; end the engagement with a report that states each
incompatibility and its evidence, and invite the user to start a new engagement - with a target-compatible DGD of
their own, or through the baseline-source ladder (rungs 2-3), which may use the incompatibility report as input
evidence (a changed baseline starts a new experiment, per `synthesize-user-workload`). Do not select a substitute, do not
rewrite the captured baseline, and do not park the run waiting for a new manifest. (A greenfield user without any
DGD is handled at the interview by the baseline-source ladder, never here.)
Later iterations use the exact challenger-approved draft. The deployer creates:

```text
<EXP_ROOT>/artifacts/deploy-iter-<NNN>/
```

Continue only when `deployment_ledger.json` is complete and `smoke_test_artifact.json` reports `success: 1`.
Functional deployment repair is owned by `recipe-deployer`; do not benchmark a failed deployment or change benchmark
semantics to hide a deployment failure.

## 4. Configure, Run, And Analyze The Benchmark

Give the successful `DEPLOY_ROOT`, exact `user_workload.yaml` path and SHA256, and current performance question and
target operating region to `perf-analyzer`. For iteration 0, use a baseline-characterization question. When `deployment.origin` is not `user`, iteration 0
is pure characterization: the baseline has no production history, so no result may be framed as an improvement or
regression against it beyond the same-series comparisons the benchmark rules already govern. For later
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
- The once-per-series noise-floor pilot (n=3) required by `comparison-uncertainty.md` is pre-authorized: the first
  time a series must adopt or retire a candidate on a small delta, `analyze-aiperf-results` returns
  `repeat_decision: necessary` with a rationale naming the series pilot, the repeats run through the normal path,
  and the analyzer records the measured noise floor and minimum detectable effect in that run's
  `performance_analysis.json`; every later same-series analysis copies both values forward so consumers always find
  them in the current `performance_analysis.json`.
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

Give the unchanged consultation and draft to `hypothesis-challenger`, along with the current
`EXP_ROOT/analysis/search-calibration.md` path. The challenger appends its hash-bound review to
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

The loop is always in exactly one state: `ACTIVE`, `PARKED_ON_ASKS`, `STOP_REQUESTED`, `STOP_GRANTED`, or
`BUDGET_STOP`. A `no-proposal` consultation never ends the engagement; it obligates the generator to produce exactly
one of the outcomes below. A `blocked` consultation also never ends the engagement: it names invalid or
inconsistent decision inputs; route it back to the step that owns the defective input (`perf-analyzer` for
analysis artifacts, `recipe-deployer` for deployment artifacts), repair, and re-enter step 4. The generator's
outcomes for `no-proposal` are exactly one of:

- the next candidate;
- an **ask**: a recorded question whose answer would unblock the highest-value deferred lever family. Asks are
  non-blocking: record the ask, surface it in one line of passing output, and keep working any still-testable
  family. Never deliver an ask through a blocking question tool during the loop — a blocking question suspends the
  turn before any goal hook can evaluate and can hang an unattended run; asks go to the artifact and the loop
  continues. Deduplicate pending asks, surface only the highest-value few, and lead the next operator-facing response
  with them. Enter `PARKED_ON_ASKS` (a pause, not a stop) only when pending asks are the only remaining work, and
  record how deployed resources are held and when they scale down in `reasoning_transcript.md`;
- a **stop-request**: the search-calibration ledger (`EXP_ROOT/analysis/search-calibration.md`) in a terminal
  state, meaning every lever family carries a
  terminal disposition in the ledger's own vocabulary: `tested`, `ruled-out`, `not-applicable`, or `deferred`
  (an answered ask resolves its family into one of these; `untested-promising` and `reopened-by-new-evidence` are
  non-terminal). Prepare the Finalize artifacts (section 7) BEFORE submitting the stop-request: at this point the top-level loop
  agent runs the section-7 correctness check and its status (or `correctness: unverified` with the reason the
  check was impossible) is recorded in `recommended_config.md` — the stop-request references the
  draft recommendation, and operator grant closes the engagement rather than starting its write-up. A `ruled-out` row must cite a measurement, a sourced hard
  constraint, a confirmed incompatibility, or an explicit operator decision; the generator's own unsourced reasoning
  does not qualify, and expected upside below the minimum detectable effect is `deferred`, not `ruled-out`.
  `deferred` is terminal for a family on exactly one of two recorded grounds: (a) its expected upside is below
  the primary series' measured minimum detectable effect (noise-floor deferral — terminal on that evidence
  alone), or (b) its ledger row records WHY its next cheapest informative experiment does not fit the remaining
  budget: estimate that experiment's cost from this engagement's own observed costs (deploy time, benchmark
  duration, GPU allocation), compare it against the remaining budget, and cite both numbers in the row. When
  the estimate fits and the recorded expected upside clears the minimum detectable effect, test the family, ask
  about it, or rule it out with qualifying evidence before requesting a stop; a fixed fraction of budget spent
  or remaining is never by itself a reason to defer.
  For a throughput-class objective, the recommendation additionally requires SATURATION EVIDENCE: the top of
  the measured operating-point curve must be flat within the series' measured noise floor, or the stop-request
  must record the explicit reason (budget arithmetic or an operator decision) in `known_limitations.md`. An
  objective still rising at the top of the measured grid leaves the operating-point family non-terminal:
  extend the grid before requesting a stop (within the workload's declared envelope; a user-pinned concurrency
  list makes extension an operator ask, per `concurrency-grid.md`).
  These judgments require the primary objective series' measured noise floor and minimum detectable effect:
  when no pilot repetition has produced them by stop-request or recommendation time, running that pilot (n>=3
  on the decision-point configuration) is PRE-AUTHORIZED and required — it is part of finishing, not a new
  candidate family. When the remaining budget cannot cover the pilot, `BUDGET_STOP` wins:
  record the missing noise floor and the resulting unquantified-delta risk in
  `known_limitations.md` and stop, rather than overrunning the budget or submitting a
  recommendation the challenger must reject. Hand the stop-request to `hypothesis-challenger` for evidence-class validation, passing the
  ledger path and the SHA256 of the submitted ledger state alongside the consultation; the challenger's verdict
  binds to that SHA256.

While a stop-request awaits challenger validation and operator grant (`STOP_REQUESTED`), continue confirmatory
runs, cleanup, and any still-testable work; launch no new candidate families. If the challenger or operator returns
objections, re-enter `ACTIVE`.

Stop only when the operator grants a validated stop-request (`STOP_GRANTED`), the authorized budget is exhausted
(`BUDGET_STOP`), access is lost and cannot be restored, or iteration 0 ends with the section-3 incompatibility
report (the baseline cannot run on the target; this is a valid engagement end, not a premature stop). Never stop because a report exists. Derive budget
consumption from existing artifacts — wall clock from `manifest.yaml`'s session start; failed deploys from
deployment ledgers' failed-attempt records; GPU-hours from GPU ALLOCATION time: for each deployment ledger, the
span from its recorded `allocated_at` (first GPU pod scheduled — a Pending pod holds no GPUs) to its recorded
`torn_down_at` (or to now, for a live deployment) times the ledger's `gpus_requested`, summed across
deployments. Benchmark-duration-only accounting undercounts real spend by excluding weight-load and hold time
and must not be used — check the totals against the contract's `budgets:`
at every iteration boundary, and cite them in every stop-request delta. A `null` budget leaves that limit
ungated.

## 7. Finalize

Before recommending, the top-level loop agent runs a correctness regression check whenever the recommended configuration differs from the
user-provided baseline in an output-affecting dimension (parallelism or reduction order, speculative decoding,
quantization, attention or MoE backend, KV dtype or reuse): replay 8-16 fixed representative prompts with frozen
continuations through both configurations and compare teacher-forced per-token log-probabilities, calibrating the
tolerance with a baseline-versus-baseline repeat; require zero request failures, malformed responses, non-finite
scores, and no unexpected truncation. Run it against the currently deployed recommended configuration; run the
baseline side only when the baseline is still live or can be redeployed within remaining budget. If no such check
is possible, record an ask; report a waived check as
`correctness: unverified`, never as a pass. Record the correctness status in `recommended_config.md`.

When the recommended or baseline configuration has speculative decoding enabled, state in the recommendation that
its acceptance behavior was measured on synthetic benchmark content and the measured magnitude may not transfer to
production traffic; acceptance length is passively observable in production telemetry and should be confirmed there.

Recommend the best valid candidate for the target objective, not automatically the most recent iteration. The
`hypothesis-generator` holds the pen for the final/ artifacts (in the stop-request path it writes them before
submission; the loop agent contributes only the correctness status). Write the
final configuration to `EXP_ROOT/final/recommended_config.md`, reproduction commands to
`EXP_ROOT/final/reproduced_commands.sh`, and limitations to `EXP_ROOT/final/known_limitations.md`. Include paths to the
user workload, original user-provided DGD, applied manifests, deployment ledgers, applicable benchmark plans, audits,
summaries, performance analyses, comparison histories, and raw AIPerf evidence. Do not call a proxy workload a
validated production result.
