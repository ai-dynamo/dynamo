---
name: perform-adversarial-review
description: >-
  Adversarially reviews an evidence-backed Dynamo optimization proposal and DGD draft for comparability, duplication,
  attribution, correctness, feasibility, and worthwhile GPU spend. Use after hypothesis-generator writes
  knowledge-consult.md and deploy-draft.yaml and before recipe-deployer creates the next deployment iteration.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - optimization
    - review
    - aiperf
    - kubernetes
---

# Perform Adversarial Review

<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Try to falsify a proposed optimization experiment before it consumes GPU time. Review the proposal; do not generate a
second one, edit its draft, deploy it, or run AIPerf.

## Inputs

Require:

- `EXP_ROOT` and the current source iteration;
- exact `EXP_ROOT/user_workload.yaml` path and SHA256 supplied by the parent;
- exact active benchmark-plan path, SHA256, and series ID returned by `perf-analyzer`;
- current `DEPLOY_ROOT/deployment_ledger.json`;
- current `DEPLOY_ROOT/applied_manifests/deploy.yaml`;
- current `DEPLOY_ROOT/benchmark/benchmark_audit.json`;
- current `DEPLOY_ROOT/benchmark/benchmark_summary.json`;
- current `DEPLOY_ROOT/benchmark/performance_analysis.json`;
- current `DEPLOY_ROOT/next-candidate/knowledge-consult.md`;
- current `DEPLOY_ROOT/next-candidate/deploy-draft.yaml` (proposal reviews only; a stop-request carries no draft
  and its absence is not an objection);
- prior deployment and benchmark artifacts;
- `EXP_ROOT/analysis/search-calibration.md` (and, for a stop-request, the submitted ledger SHA256 cited in
  `knowledge-consult.md`); and
- `EXP_ROOT/analysis/hypothesis-backlog.jsonl` and `EXP_ROOT/analysis/challenger-reviews.jsonl` when present; and
- `EXP_ROOT/manifest.yaml` (session start time, for stop-request budget arithmetic).

Review only a consultation whose decision is `proposed` and whose draft materialization completed successfully. For
`no-proposal` or `blocked` that carries no stop-request, return without writing a candidate verdict. For a
stop-request (a `no-proposal` consultation whose delta cites the search-calibration ledger path and a
submitted SHA256 — a `blocked` consultation never carries one), run the Stop-Request Validation below instead of
returning.

## Stop-Request Validation

1. Verify the SHA256 cited in `knowledge-consult.md` matches the on-disk
   `EXP_ROOT/analysis/search-calibration.md`. On mismatch, reject: the ledger moved after submission.
2. Validate completeness and evidence class against the ledger, not the consult file (which carries only the
   delta): every lever family carries a terminal disposition (`tested`, `ruled-out`, `not-applicable`, or `deferred` — an answered ask resolves its family into one of these; `untested-promising` and `reopened-by-new-evidence` are non-terminal); every `ruled-out` row cites a measurement, a sourced hard constraint, a confirmed incompatibility,
   or an explicit operator decision; every `deferred` row is terminal on a recorded ground - upside below the primary series' measured minimum
   detectable effect, or a cited cost-estimate-vs-remaining-budget arithmetic (reject when an above-MDE deferred
   family lacks that arithmetic, or when its estimate fits the remaining budget; return that family as the
   required follow-up); for a throughput-class objective the recommendation
   carries saturation evidence or a recorded budget/operator reason in `known_limitations.md` ("still rising at
   the top of the measured grid" is not terminal); and the stop-request's draft recommendation at `EXP_ROOT/final/recommended_config.md` carries its required
   `Correctness status:` line (require that path as an input for stop-request validation).
3. Verify the stop-request delta cites derived budget consumption (wall clock from `manifest.yaml` session start;
   failed-deploy count from the deployment ledgers; GPU-hours from GPU allocation time, per deployment ledger
   `allocated_at`-to-`torn_down_at` span — or to now, for a live deployment — times its `gpus_requested`,
   summed across deployments) and that the cited sources
   support the arithmetic, whenever any granted budget is non-null.
4. Append the verdict to `EXP_ROOT/analysis/challenger-reviews.jsonl` as for any review, binding it to the
   submitted ledger SHA256, and state in it that this is procedural validation, not independent adversarial
   assurance. A validated stop-request returns to the PARENT with state `STOP_REQUESTED` for operator grant; it is
   never a deployment handoff, and `return_to` is the parent, not `recipe-deployer`.

## Read The Applicable Rules

Always read:

- `agent-docs/rules/benchmarking/evidence-eligibility.md`;
- `agent-docs/rules/benchmarking/comparison-uncertainty.md`;
- `agent-docs/rules/benchmarking/series-boundaries.md`;
- `agent-docs/rules/optimization/evidence-before-spend.md`;
- `agent-docs/rules/optimization/one-variable.md`;
- `agent-docs/rules/verification/config-engagement.md`;
- `agent-docs/rules/verification/implausible-speedup.md`;
- `agent-docs/rules/verification/overlap.md`;
- `agent-docs/rules/verification/stack-verdict.md`;
- `agent-docs/rules/execution/user-workload.md` (the `resources.pinned` and `resources.gpu_ceiling` semantics);
- `agent-docs/guides/knob-tuning/tuning-hierarchy.md`; and
- the sources and repository guides cited for the proposal's central mechanism.

Read the Dynamo catalog for a Dynamo-owned knob, only the active engine guide for an engine-owned knob, the model-sizing
guides for a topology or memory-fit proposal, and the rate-matching guide for a disaggregated allocation proposal. Do
not invoke `consult-perf-knowledge` again or reconstruct a new shortlist. When relevant to the attached evidence, also
read the proxy-workload, concurrency-grid, or benchmark-isolation rule.

## Establish Review Integrity

Before judging the idea:

1. Require `benchmark_audit.json` to report `valid` or `valid_with_recovery`.
2. Confirm the plan, audit, summary, and analysis identify the same active series and every direct comparison is
   same-series. Record material resource or placement differences as limitations.
3. Recompute the source manifest, consultation, and draft SHA256 hashes.
4. Require the source and draft hashes to match `Materialization Handoff`.
5. Parse both manifests and independently compute their semantic diff.
6. Require the consultation, materialized diff, user workload, and performance analysis to identify the same model,
   engine, deployment, objective, and source operating region.
7. Require a concrete performance question, target operating region, and expected measurable effect for the candidate.

Treat a missing, stale, contradictory, or non-comparable input as a blocking objection. Do not repair it inside the
review.

## Challenge The Proposal

Attack the proposal from these directions:

- **Evidence**: Does it contain at least three distinct qualifying categories, including AIPerf profiler data? Does
  each source support the stated mechanism, or has contextual evidence been promoted beyond its limits?
- **Uncertainty**: Are changes at or below the measured noise floor of the active benchmark series classified as
  noise (per `comparison-uncertainty.md`)? For larger changes, does the conclusion match the
  available single-run or multi-run evidence? Were confidence intervals used only when deliberate repetitions made
  them useful, and is every requested repeat necessary enough to justify its GPU cost? Are degraded single-run
  statistics or surprising gains treated cautiously?
- **Redundancy**: Has the same semantic configuration already been tested, rejected, or left inconclusive under the
  same workload? If so, is there specific new evidence that makes this attempt different?
- **Frontier**: When the proposal selects or rejects a topology/config family, is the judgment made at each
  family's own SLO frontier per `tuning-hierarchy.md`? Reject a family selection argued from a single shared
  operating point when the families' frontiers differ or the winner's frontier is unmeasured.
- **Priority**: Compared with the existing backlog, does this candidate have competitive information value, likely
  impact on the primary objective, reversibility, GPU cost, and risk at the target operating region?
- **Attribution**: Does the complete diff express one independently testable knob? For a coupled bundle, is every field
  required for one mechanism or supported by prior interaction evidence, with an ablation where needed?
- **Provenance**: When `deployment.origin` is `recipe-confirmed` or `agent-authored`, reject any framing of the
  baseline as a production reference; iteration 0 characterizes an unvalidated starting point, and topology
  families inherited from it are open questions, not settled decisions.
- **Mechanism**: Does the proposed lever address a plausible reducible gap at the target operating region, or merely
  move work that evidence suggests is already bounded? Are internal causes still labeled as hypotheses?
- **Evaluation**: Is the expected effect tied to the primary objective or failed SLO? Does the proposal state what the
  candidate should teach us without prescribing favorable benchmark settings or assuming the current series must be
  reused? Could another declared metric or target operating point regress enough to defeat the experiment's value?
- **Feasibility**: Is the knob valid for the active Dynamo and engine versions? Reject outright any candidate that
  changes a knob listed in the contract's `resources.pinned` or whose deployment would exceed
  `resources.gpu_ceiling`; these are blocking objections regardless of evidence quality. Check GPU and replica
  arithmetic, memory headroom, startup and OOM risk, topology consistency, and whether engagement can be proven
  after deployment.
- **Correctness**: Does the draft preserve target-fixed model, framework, precision, hardware, and workload
  constraints? If the change can alter output behavior, is there a concrete correctness check and rollback criterion?
- **Spend**: Is the expected information value worth the deployment and benchmark cost? Could a cheaper evidence
  check resolve the uncertainty before GPU time is spent?

Use the attached consultation as the proposal's evidence boundary. Verify its claims, but do not reject it merely for
using concise prose or a flexible section layout.

## Choose A Verdict

Return exactly one:

- `approve`: no blocking objection remains; the existing draft is ready to enter deployment unchanged.
- `revise`: the same experiment is worth testing after a small, explicit correction.
- `reject`: the experiment is redundant, invalid, out of scope, unsafe, weakly supported, or unlikely to answer the
  target performance question.

An approval cannot contain a blocking objection. For `revise`, give the smallest useful revision. Return every
`revise` or `reject` verdict to `hypothesis-generator`; the generator decides which of its skills to rerun.
Never edit `knowledge-consult.md` or `deploy-draft.yaml` during review.

## Write The Review

Append one compact JSON object to:

```text
<EXP_ROOT>/analysis/challenger-reviews.jsonl
```

Use this contract:

```json
{
  "review_id": "deploy-iter-<NNN>-<draft-sha256-prefix>",
  "source_iteration": 0,
  "candidate_iteration": 1,
  "consult_path": "artifacts/deploy-iter-<NNN>/next-candidate/knowledge-consult.md",
  "consult_sha256": "",
  "candidate_path": "artifacts/deploy-iter-<NNN>/next-candidate/deploy-draft.yaml",
  "candidate_sha256": "",
  "verdict": "approve",
  "return_to": "recipe-deployer",
  "summary": "",
  "objections": [
    {
      "severity": "blocking",
      "check": "",
      "finding": "",
      "evidence": [],
      "required_resolution": ""
    }
  ],
  "revised_experiment_plan": null,
  "supersedes_review_id": null,
  "reviewed_at": ""
}
```

Order objections by severity and impact. Cite exact iteration IDs, paths, hashes, metrics, or source files. For
`approve`, set `objections` to only non-blocking cautions or an empty list and identify the exact approved candidate
path and hash. Set `return_to` to `recipe-deployer` only for an approved PROPOSAL; for a validated stop-request set it to the
parent (operator grant); otherwise set it to `hypothesis-generator`.
For `revise`, make `revised_experiment_plan` concise. Do not turn a rejection into an unrelated replacement hypothesis.

A stop-request validation returns `stop-validated` or `stop-rejected` instead of the proposal verdicts, and its
review ID binds to the submitted ledger SHA256 prefix (`deploy-iter-<NNN>-stop-<ledger-sha256-prefix>`).

Use a stable review ID bound to the draft hash. If that exact draft already has a review, return the existing record
instead of appending a duplicate. A revised draft receives a new review ID and names the prior record in
`supersedes_review_id`.

## Return

For `approve` (proposals only), return the review ID, exact candidate path and SHA256, performance question, and target operating region
to the parent and `recipe-deployer`. For `stop-validated` or `stop-rejected`, return the verdict and review ID to
the parent only (`STOP_REQUESTED` awaits operator grant; a rejection names the required follow-up families). The parent must carry the question and operating region into the candidate's
`perf-analyzer` assignment. For `revise` or `reject`, return the verdict, strongest objections, and any minimal revision
or required follow-up to `hypothesis-generator`. Do not create the next `DEPLOY_ROOT` for candidate iteration
`<NNN + 1>`.
