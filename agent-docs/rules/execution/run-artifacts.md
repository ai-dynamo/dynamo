<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Run Artifacts

Canonical local artifact layout for Dynamo agent sessions.

## Definitions

- `EXP_ID`: stable id for one end-to-end session.
- `EXP_ROOT`: `runs/<EXP_ID>/`, created by `user-interviewer`.
- `ITERATION`: zero-based optimization iteration assigned to one candidate DGD.
- `DEPLOY_ID`: `deploy-iter-<NNN>`.
- `DEPLOY_ROOT`: `runs/<EXP_ID>/artifacts/<DEPLOY_ID>/`, created by `recipe-deployer`.
- `HYPOTHESIS_ROOT`: `<DEPLOY_ROOT>/next-candidate/`, created by `hypothesis-generator` for a proposal derived from
  the current successful deployment.

## Tree

```text
runs/<EXP_ID>/
|-- manifest.yaml
|-- user_workload.yaml
|-- reasoning_transcript.md
|-- inputs/
|   |-- user_provided_dgd.yaml
|   `-- benchmark-plans/
|       `-- <series-id>.json
|-- analysis/
|   |-- hypothesis-backlog.jsonl
|   |-- challenger-reviews.jsonl
|   |-- performance_findings.jsonl
|   |-- asks.jsonl
|   `-- search-calibration.md
|-- final/
|   |-- recommended_config.md
|   |-- reproduced_commands.sh
|   `-- known_limitations.md
`-- artifacts/
    `-- deploy-iter-000/
        |-- deployment_ledger.json
        |-- smoke_test_artifact.json
        |-- applied_manifests/
        |   |-- model-cache.yaml
        |   |-- model-download.yaml
        |   |-- model-validate.yaml
        |   `-- deploy.yaml
        |-- logs/                       # created only for targeted failure logs
        |-- benchmark/
        |   |-- perf.yaml
        |   |-- aiperf-config.yaml
        |   |-- benchmark_execution.json
        |   |-- benchmark_audit.json
        |   |-- benchmark_summary.json
        |   |-- performance_analysis.json
        |   |-- performance_analysis.md
        |   `-- raw_aiperf/
        `-- next-candidate/
            |-- knowledge-consult.md
            `-- deploy-draft.yaml        # created only for a materialized proposal
```

## Artifact Types

- `manifest.yaml`: session metadata, timestamps, repo commit, cluster context name, and agent versions when known.
  Created by `user-interviewer` when it establishes `EXP_ROOT`; any role that changes cluster context updates it.
- `user_workload.yaml`: canonical workload and target-cluster contract synthesized by `user-interviewer`, including
  the baseline DGD's canonical path and SHA256.
- `reasoning_transcript.md`: time-stamped long-running document capturing the agent's reasoning, key decisions, actions, rationale, and status.
  Created and maintained by the top-level loop agent (the session running `optimize-loop.md`) per
  `agent-docs/rules/execution/logging.md`; specialized roles contribute through their own artifacts.
- `user_provided_dgd.yaml`: immutable baseline DGD supplied by the user and captured by `user-interviewer`.
- `benchmark-plans/<series-id>.json`: immutable performance question, workload, measurement semantics, objectives, and
  required references for one benchmark series.
- `hypothesis-backlog.jsonl`: append-only record of generated optimization proposals and their source evidence.
  Appended by `hypothesis-generator` (via `consult-perf-knowledge`) once per consultation, whatever the outcome, so
  the challenger's redundancy check always runs against complete history.
- `challenger-reviews.jsonl`: append-only, hash-bound adversarial reviews of materialized proposals and
  stop-request validations (the latter bound to the submitted ledger SHA256).
- `performance_findings.jsonl`: append-only performance findings produced from valid benchmark analyses.
- `deployment_ledger.json`: assigned source DGD path and SHA256, manifests applied, readiness status, endpoint,
  smoke-test result, concise diagnostics, blockers, and cleanup commands.
- `smoke_test_artifact.json`: required `recipe-deployer` result containing the full smoke-test API request, full
  `api_response`, and success flag.
- `benchmark_execution.json`: active plan path and SHA256, exact Kubernetes/AIPerf execution, status, retries, artifact
  collection, and blockers.
- `benchmark_audit.json`: active plan and series identity, schema, completeness, workload identity, request/error, and
  comparability validity checks.
- `benchmark_summary.json`: normalized AIPerf metrics, units, benchmark inputs, and error counts without interpretation.
- `performance_analysis.json`: target-SLO evaluation, absolute results, and applicable comparisons to the series
  baseline, previous and best valid same-series results, and same-series history.
- `performance_analysis.md`: concise human-readable findings and limitations.
- `applied_manifests/`: one final run-scoped copy of each manifest type used. After success, these are the exact files
  that produced the successful smoke test.
- `logs/`: optional, targeted failure output that is useful beyond the concise ledger excerpt.
- `raw_aiperf/`: unmodified AIPerf output files copied by the benchmarking flow.
- `next-candidate/`: reasoning and the optional draft for the candidate proposed from this deployment iteration.
- `knowledge-consult.md`: required consultation result for `proposed`, `no-proposal`, and `blocked` outcomes.
- `deploy-draft.yaml`: candidate DGD created only for a materialized proposal; it remains here until challenger
  approval assigns it to the next deployment iteration.
- `asks.jsonl` (under `EXP_ROOT/analysis/`): append-only operator-ask record, written by whichever role records
  the ask (typically `hypothesis-generator` via `consult-perf-knowledge`), with the question, blocked lever
  family, expected upside, status (`pending` or `answered`), and the answer once received. Deduplicate before
  appending. A stop-request is not a separate file: it is the search-calibration ledger
  (`EXP_ROOT/analysis/search-calibration.md`) in a terminal state, plus its challenger validation. The ledger is
  the authoritative family table; the submitting iteration's `knowledge-consult.md` records only the stop-request
  delta and cites the ledger path and the SHA256 of the ledger state submitted for validation.

## Deployment Directories

- `user-interviewer` creates `EXP_ROOT`, `user_workload.yaml`, and `inputs/user_provided_dgd.yaml` once for the
  optimization job. No role ever edits them afterward; a baseline that cannot run on the
  target ends the engagement, and a changed user DGD starts a new experiment.
- `recipe-deployer` creates one `DEPLOY_ROOT` for every newly assigned candidate DGD.
- Benchmarking and analysis agents add their results to that candidate's existing `DEPLOY_ROOT`.
- `hypothesis-generator` creates `HYPOTHESIS_ROOT` inside the current analyzed `DEPLOY_ROOT`; it does not create the
  next deployment iteration. When preparing a stop-request it also writes the three `final/` artifacts
  (`recommended_config.md` with its `Correctness status:` line, `reproduced_commands.sh`, `known_limitations.md`)
  BEFORE submission, per `optimize-loop.md` section 6.
- `hypothesis-challenger` reviews the handoff in place. After approval, `recipe-deployer` creates the new
  `DEPLOY_ROOT`.
- Use a zero-padded iteration, for example `deploy-iter-003`.
- Keep retries and compatibility patches for the same candidate in the same `DEPLOY_ROOT`.
- Create the next deployment directory only when the optimization loop assigns a new candidate.
- Before iteration > 0, remove only the previous iteration's DGD. Keep its deployment directory and successful YAML
  unchanged, and preserve shared PVCs, model-cache jobs, namespaces, and secrets.

## Final Manifest Set

- Copy every manifest used into `applied_manifests/` and never modify the handed-off source files.
- Update the run-scoped copy in place when a compatibility patch is required, then reapply it.
- Record the assigned DGD path and SHA256, every patch, and every reapply in `deployment_ledger.json`.
- Do not retain numbered intermediate manifest copies.
- After a successful smoke test, keep exactly one final file per manifest type used. If blocked, keep only the latest
  attempted files and mark the ledger blocked.

## Rules

- Put every session artifact under `EXP_ROOT` and every candidate attempt under exactly one `DEPLOY_ROOT`.
- Prefer paths relative to `EXP_ROOT` inside JSON ledgers.
- Do not generate broad cluster snapshots, duplicate endpoint responses, successful pod logs, or unrelated command
  output. The deployment ledger is the primary operational record.
- Never store secret values, kubeconfig contents, tokens, or registry credentials.
- Never overwrite a benchmark-series plan. Create a new plan and series ID when the performance question requires
  different workload or measurement semantics.
- Never overwrite raw AIPerf output or a previous iteration's benchmark files.
- Never modify the successful source manifest while generating a hypothesis. Write the full proposed manifest to
  `next-candidate/deploy-draft.yaml`.
- For `no-proposal` or `blocked`, retain `next-candidate/knowledge-consult.md` and do not create a misleading
  `deploy-draft.yaml`.
- Make direct comparisons only when `benchmark_audit.json` marks every run valid and their plan and benchmark-series
  identities match.
- `recommended_config.md` MUST carry a `Correctness status:` line — `verified (scope stated)`,
  `unverified (waived: reason)`, or `blocked (ask recorded)`. A recommendation without it is incomplete.
- Do not treat a final recommendation as reproducible unless it points to the user workload, original user-provided
  DGD, applied manifests, deployment ledger, applicable benchmark plans, audits, summaries, performance analyses, and
  raw benchmark artifacts.
