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

Try to falsify a proposed optimization experiment before it consumes GPU time. Review the proposal; do not generate a
second one, edit its draft, deploy it, or run AIPerf.

## Inputs

Require:

- `EXP_ROOT` and the current source iteration;
- `EXP_ROOT/target_workload.yaml`;
- `EXP_ROOT/inputs/benchmark_plan.json`;
- current `DEPLOY_ROOT/deployment_ledger.json`;
- current `DEPLOY_ROOT/applied_manifests/deploy.yaml`;
- current `DEPLOY_ROOT/benchmark/benchmark_audit.json`;
- current `DEPLOY_ROOT/benchmark/benchmark_summary.json`;
- current `DEPLOY_ROOT/benchmark/performance_analysis.json`;
- current `DEPLOY_ROOT/next-candidate/knowledge-consult.md`;
- current `DEPLOY_ROOT/next-candidate/deploy-draft.yaml`;
- prior deployment and benchmark artifacts; and
- `EXP_ROOT/analysis/hypothesis-backlog.jsonl` and `EXP_ROOT/analysis/challenger-reviews.jsonl` when present.

Review only a consultation whose decision is `proposed` and whose draft materialization completed successfully. For
`no-proposal` or `blocked`, return without writing a candidate verdict.

## Read The Applicable Rules

Always read:

- `agent-docs/rules/benchmarking/evidence-eligibility.md`;
- `agent-docs/rules/benchmarking/comparison-uncertainty.md`;
- `agent-docs/rules/benchmarking/series-boundaries.md`;
- `agent-docs/rules/optimization/evidence-before-spend.md`;
- `agent-docs/rules/optimization/one-variable.md`;
- `agent-docs/rules/verification/config-engagement.md`;
- `agent-docs/rules/verification/implausible-speedup.md`;
- `agent-docs/rules/verification/node-equivalence.md`;
- `agent-docs/rules/verification/overlap.md`;
- `agent-docs/rules/verification/stack-verdict.md`;
- `agent-docs/guides/knob-tuning/tuning-hierarchy.md`; and
- the sources and repository guides cited for the proposal's central mechanism.

Read the Dynamo catalog for a Dynamo-owned knob, only the active engine guide for an engine-owned knob, the model-sizing
guides for a topology or memory-fit proposal, and the rate-matching guide for a disaggregated allocation proposal. Do
not invoke `consult-perf-knowledge` again or reconstruct a new shortlist. When relevant to the attached evidence, also
read the proxy-workload, concurrency-grid, or benchmark-isolation rule.

## Establish Review Integrity

Before judging the idea:

1. Require `benchmark_audit.json` to report `valid` or `valid_with_recovery`.
2. Confirm the analyzed run and every decision-grade comparison use the same benchmark series and node-equivalent
   hardware.
3. Recompute the source manifest, consultation, and draft SHA256 hashes.
4. Require the source and draft hashes to match `Materialization Handoff`.
5. Parse both manifests and independently compute their semantic diff.
6. Require the consultation, materialized diff, target workload, and performance analysis to identify the same model,
   engine, deployment, objective, and operating region.

Treat a missing, stale, contradictory, or non-comparable input as a blocking objection. Do not repair it inside the
review.

## Challenge The Proposal

Attack the proposal from these directions:

- **Evidence**: Does it contain at least three distinct qualifying categories, including AIPerf profiler data? Does
  each source support the stated mechanism, or has contextual evidence been promoted beyond its limits?
- **Uncertainty**: Are history claims above the `0.5%` noise floor and clear of AIPerf confidence-interval overlap?
  Are degraded single-run statistics or surprising gains treated cautiously?
- **Redundancy**: Has the same semantic configuration already been tested, rejected, or left inconclusive under the
  same workload? If so, is there specific new evidence that makes this attempt different?
- **Priority**: Compared with the existing backlog, does this candidate have competitive information value, likely
  impact on the primary objective, reversibility, GPU cost, and risk at the target operating region?
- **Attribution**: Does the complete diff express one independently testable knob? For a coupled bundle, is every field
  required for one mechanism or supported by prior interaction evidence, with an ablation where needed?
- **Mechanism**: Does the proposed lever address a plausible reducible gap at the target operating region, or merely
  move work that evidence suggests is already bounded? Are internal causes still labeled as hypotheses?
- **Objective impact**: Is the expected effect tied to the primary objective or failed SLO? Could another declared
  metric or target operating point regress enough to defeat the experiment's value?
- **Feasibility**: Is the knob valid for the active Dynamo and engine versions? Check GPU and replica arithmetic,
  memory headroom, startup and OOM risk, topology consistency, and whether engagement can be proven after deployment.
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
- `defer`: the idea is plausible, but missing evidence or an unresolved prerequisite should be addressed before GPU
  spend.
- `reject`: the experiment is redundant, invalid, out of scope, unsafe, weakly supported, or unlikely to answer the
  target performance question.

An approval cannot contain a blocking objection. For `revise`, give the smallest useful revision. Return every
`revise`, `defer`, or `reject` verdict to `hypothesis-generator`; the generator decides which of its skills to rerun.
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
  "required_follow_up": [],
  "supersedes_review_id": null,
  "reviewed_at": ""
}
```

Order objections by severity and impact. Cite exact iteration IDs, paths, hashes, metrics, or source files. For
`approve`, set `objections` to only non-blocking cautions or an empty list and identify the exact approved candidate
path and hash. Set `return_to` to `recipe-deployer` only for `approve`; otherwise set it to `hypothesis-generator`.
For `revise`, make `revised_experiment_plan` concise. For `defer`, make `required_follow_up` concrete. Do not turn a
rejection into an unrelated replacement hypothesis.

Use a stable review ID bound to the draft hash. If that exact draft already has a review, return the existing record
instead of appending a duplicate. A revised draft receives a new review ID and names the prior record in
`supersedes_review_id`.

## Return

For `approve`, return the review ID plus the exact candidate path and SHA256 to `recipe-deployer`. For `revise`,
`defer`, or `reject`, return the verdict, strongest objections, and any minimal revision or required follow-up to
`hypothesis-generator`. Do not create the next `deploy-iter-<NNN>` directory.
