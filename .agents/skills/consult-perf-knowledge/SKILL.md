---
name: consult-perf-knowledge
description: >-
  Consults the repository performance rules and applicable Dynamo and engine guides to select one evidence-backed
  optimization proposal, then writes the generator's knowledge-consult.md reasoning record. Use after perf-analyzer
  completes a valid comparable AIPerf run and before create-optimization-hypothesis materializes a DGD draft.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - optimization
    - aiperf
    - hypothesis
    - performance
---

# Consult Performance Knowledge

Turn the current audited performance finding into one documented configuration proposal. Write the reasoning record;
do not edit a deployment manifest, deploy anything, or run AIPerf.

## Inputs

Require:

- `EXP_ROOT` and the zero-based current optimization iteration;
- `EXP_ROOT/target_workload.yaml`;
- `EXP_ROOT/inputs/benchmark_plan.json`;
- current `DEPLOY_ROOT/deployment_ledger.json`;
- current successful `DEPLOY_ROOT/smoke_test_artifact.json`;
- current `DEPLOY_ROOT/applied_manifests/deploy.yaml`;
- current `DEPLOY_ROOT/benchmark/benchmark_audit.json`;
- current `DEPLOY_ROOT/benchmark/benchmark_summary.json`;
- current `DEPLOY_ROOT/benchmark/performance_analysis.json`;
- prior deployment and benchmark artifacts; and
- `EXP_ROOT/analysis/hypothesis-backlog.jsonl` and `EXP_ROOT/analysis/challenger-reviews.jsonl` when present.

Treat the current `deploy-iter-<NNN>` as the source iteration and `<NNN + 1>` as the candidate iteration. Never create
the next deployment-iteration directory.

## Read The Applicable Knowledge

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
- `agent-docs/guides/knob-tuning/tuning-hierarchy.md`;
- all three files under `agent-docs/guides/model-sizing/`;
- `agent-docs/guides/knob-tuning/dynamo.md`; and
- only the active engine guide: `vllm.md`, `sglang.md`, or `tensorrt-llm.md`.

Additionally read:

- `agent-docs/guides/rate-matching/matching.md` for a disaggregated allocation decision;
- `agent-docs/rules/benchmarking/concurrency-grid.md` when interpreting a capacity or concurrency series;
- `agent-docs/rules/benchmarking/proxy-workload-selection.md` when the audit identifies a recipe proxy;
- `agent-docs/rules/benchmarking/benchmark-isolation.md` when the audit reports an isolation limitation; and
- `agent-docs/references/reference-repos.md` before consulting current framework or Kubernetes source or official
  documentation.

Do not load guides for inactive engines. Verify version-sensitive flags and defaults against the active image, checked
out source, generated help, or official documentation. Treat evidence transferred across a model, engine version,
hardware class, topology, or workload as an explicit assumption.

## Validate The Decision Inputs

Proceed with a proposal only when:

- the smoke test succeeded;
- the benchmark audit status is `valid` or `valid_with_recovery`;
- the benchmark-series identity matches the canonical plan;
- the summary and performance analysis refer to the current candidate and executed workload;
- the exact successful source manifest exists;
- target-fixed model, framework, precision, hardware, workload, and SLO constraints are known; and
- comparison runs used as decision evidence are valid, same-series, and node-equivalent.

If an input is missing, inconsistent, invalid, or non-comparable, write a `blocked` consultation. If the inputs are
valid but no defensible lever meets the evidence gate, write `no-proposal`. Do not manufacture a candidate.

## Establish The Performance Finding

Record:

- the primary objective or failed SLO and target operating region;
- absolute current metrics and client-visible symptoms;
- comparisons with the original valid baseline, previous valid iteration, best prior result per objective, and
  relevant same-series history;
- AIPerf confidence intervals, coefficient of variation when available, and the `0.5%` noise floor;
- proxy limitations, missing metrics, node differences, or other uncertainty; and
- whether any surprising prior result needs engagement or plausibility rechecking.

AIPerf establishes client-visible behavior, not a router, scheduler, transfer, or backend root cause. State internal
mechanisms as hypotheses unless separate engagement or runtime evidence supports them.

## Enforce The Evidence Gate

A proposed candidate must cite at least three distinct evidence categories, and one must be AIPerf profiler data:

1. AIPerf profiler data and audited analysis tied to the objective or SLO.
2. Dynamo or active-engine source, official documentation, or performance guidance.
3. Same-series benchmark history from prior candidates.
4. Model architecture details relevant to the mechanism.
5. Hardware speed-of-light or roofline analysis when the mechanism concerns compute, memory, or communication limits.

Count categories, not citations. Multiple AIPerf metrics still count as one category. For each item, record the exact
path or citation, observation, what it supports, and its limitation. Do not invent category 5 when no applicable
analysis exists. If fewer than three categories qualify, use `no-proposal` and name the missing evidence.

## Select One Lever

Follow `tuning-hierarchy.md`:

1. Classify the exact model and compute memory fit, minimum parallelism, and headroom.
2. Screen topology first. Keep it unchanged unless current evidence supports a structural mismatch.
3. When topology is viable, consider Tier 2 families in default priority order.
4. Record why every higher-priority family through the selected family was retained, skipped, rejected, or superseded
   by stronger current-run evidence.
5. Consider Local Planner only when autoscaling is the explicit objective and the current single DGD already contains
   it.
6. Deduplicate the shortlist against all prior attempts and challenger reviews.
7. Rank surviving choices by direct evidence, expected effect on the primary objective, information value, risk,
   reversibility, experiment cost, and diff size.

Select one independently testable knob. A coupled bundle is allowed only when every changed field is required for one
functional mechanism or prior isolated evidence supports the interaction. Classify the reason as
`functionality-required` or `evidence-supported-interaction`; list every field and any required follow-up ablation.

Do not change benchmark semantics or weaken target-fixed constraints. Do not retry an equivalent failed or inconclusive
candidate unless new evidence explains why its outcome may differ.

## Write The Consultation

Create:

```text
<EXP_ROOT>/artifacts/deploy-iter-<NNN>/next-candidate/knowledge-consult.md
```

Use `DEPLOY_ROOT/next-candidate/` as `HYPOTHESIS_ROOT`.
Write the file for `proposed`, `no-proposal`, and `blocked` outcomes. Do not create `deploy-draft.yaml`; that belongs to
`create-optimization-hypothesis`.

Use this as a loose outline, not a form. Keep the `Decision`, `Evidence`, `Proposed Change`, and
`Materialization Handoff` sections so the next skill can find the required facts. Organize the reasoning in whatever
way best explains the recommendation, add useful subsections, and omit irrelevant prompts.

```markdown
# Performance Knowledge Consultation: Candidate Iteration <NNN>

## Decision
- Status: proposed | no-proposal | blocked

## Reasoning
Summarize the primary objective/SLO, the measured problem, relevant comparisons and uncertainty, applicable model or topology constraints, tuning guidance, prior attempts, and why this is the most useful next experiment. Include assumptions or missing evidence that affect confidence.

## Evidence
- Qualifying category count:

| Category | Evidence and source | Relevance and limitation |
|---|---|---|

## Proposed Change
- Candidate type: single-knob | coupled-bundle | none
- Knob owner: Dynamo | vLLM | SGLang | TensorRT-LLM | none
- Primary knob:
- Expected measurable effect:
- Risks/metrics that may regress:
- Coupling reason: none | functionality-required | evidence-supported-interaction
- Required follow-up ablation (if necessary):

## Materialization Handoff
- Source manifest:
- Source manifest SHA256:
- Intended draft: next-candidate/deploy-draft.yaml
- Draft manifest SHA256: pending

```

Keep every evidence item used to support the decision, grouped under its qualifying category. Include at least three
distinct categories, including AIPerf profiler data, but keep each entry concise. Name any repository guide, source, or
official documentation that supplied a constraint or recommendation.

Include relevant benchmark history and tuning-hierarchy decisions in the reasoning
without reproducing every rejected option. Define a future improvement as supported only when it exceeds the `0.5%`
noise floor and clears AIPerf's confidence intervals in the improving direction; otherwise classify it as noise or
inconclusive.

## Return

For `proposed`, return `knowledge-consult.md` to `create-optimization-hypothesis`. For `no-proposal` or `blocked`, return
the consultation to the caller and stop without creating a draft.
