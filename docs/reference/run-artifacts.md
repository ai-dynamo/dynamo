# Run Artifacts

Canonical local artifact layout for Dynamo agent sessions.

## Definitions

- `EXP_ID`: stable id for one end-to-end session.
- `EXP_ROOT`: `runs/<EXP_ID>/`, created by `recipe-explorer`.
- `ITERATION`: zero-based optimization iteration assigned to one candidate recipe.
- `DEPLOY_ID`: `deploy-iter-<NNN>`.
- `DEPLOY_ROOT`: `runs/<EXP_ID>/artifacts/<DEPLOY_ID>/`, created by `recipe-deployer`.

## Tree

```text
runs/<EXP_ID>/
|-- manifest.yaml
|-- target_workload.yaml
|-- inputs/
|   |-- recipe_inventory.json
|   |-- selected_recipe.json
|   `-- benchmark_plan.json
|-- analysis/
|   |-- hypothesis_backlog.jsonl
|   |-- challenger_reviews.jsonl
|   `-- performance_findings.jsonl
|-- final/
|   |-- recommended_config.md
|   |-- reproduced_commands.sh
|   `-- known_limitations.md
|-- experience_summary.json
|-- experience_review.md
`-- artifacts/
    `-- deploy-iter-000/
        |-- deployment_ledger.json
        |-- smoke_test_artifact.json
        |-- applied_manifests/
        |   |-- model-cache.yaml
        |   |-- model-download.yaml
        |   `-- deploy.yaml
        |-- logs/                       # created only for targeted failure logs
        `-- benchmark/
            |-- perf.yaml
            |-- aiperf-config.yaml
            |-- benchmark_execution.json
            |-- benchmark_audit.json
            |-- benchmark_summary.json
            |-- performance_analysis.json
            |-- performance_analysis.md
            `-- raw_aiperf/
```

## Artifact Types

- `manifest.yaml`: session metadata, timestamps, repo commit, cluster context name, and agent versions when known.
- `target_workload.yaml`: canonical user-provided workload and target-cluster contract.
- `recipe_inventory.json`: structured inventory from `recipe-explorer`.
- `benchmark_plan.json`: canonical workload source, trace hash, benchmark series, AIPerf settings, objectives, and
  comparison contract frozen at iteration 0.
- `deployment_ledger.json`: manifests applied, readiness status, endpoint, smoke-test result, concise diagnostics,
  blockers, and cleanup commands.
- `smoke_test_artifact.json`: required `recipe-deployer` result containing the full smoke-test API request, full
  `api_response`, and success flag.
- `benchmark_execution.json`: exact Kubernetes/AIPerf execution, status, retries, artifact collection, and blockers.
- `benchmark_audit.json`: schema, completeness, workload identity, request/error, and comparability validity checks.
- `benchmark_summary.json`: normalized AIPerf metrics, units, benchmark inputs, and error counts without interpretation.
- `performance_analysis.json`: target-SLO evaluation plus comparisons to the original baseline, previous valid run,
  best prior run, and all prior valid same-series runs.
- `performance_analysis.md`: concise human-readable findings and limitations.
- `applied_manifests/`: one final run-scoped copy of each manifest type used. After success, these are the exact files
  that produced the successful smoke test.
- `logs/`: optional, targeted failure output that is useful beyond the concise ledger excerpt.
- `raw_aiperf/`: unmodified AIPerf output files copied by the benchmarking flow.
- `experience_summary.json` and `experience_review.md`: one end-of-run review written by `learn-from-experience`.

## Deployment Directories

- `recipe-explorer` creates `EXP_ROOT` once for the optimization job.
- `recipe-deployer` creates one `DEPLOY_ROOT` for every newly assigned candidate recipe.
- Benchmarking and analysis agents add their results to that candidate's existing `DEPLOY_ROOT`.
- Use a zero-padded iteration, for example `deploy-iter-003`.
- Keep retries and compatibility patches for the same candidate in the same `DEPLOY_ROOT`.
- Create the next deployment directory only when the optimization loop assigns a new candidate.
- Before iteration > 0, remove only the previous iteration's DGD. Keep its deployment directory and successful YAML
  unchanged, and preserve shared PVCs, model-cache jobs, namespaces, and secrets.

## Final Manifest Set

- Copy every manifest used into `applied_manifests/` and never modify the tracked recipe source.
- Update the run-scoped copy in place when a compatibility patch is required, then reapply it.
- Record the source recipe path, every patch, and every reapply in `deployment_ledger.json`.
- Do not retain numbered intermediate manifest copies.
- After a successful smoke test, keep exactly one final file per manifest type used. If blocked, keep only the latest
  attempted files and mark the ledger blocked.

## Rules

- Put every session artifact under `EXP_ROOT` and every candidate attempt under exactly one `DEPLOY_ROOT`.
- Prefer paths relative to `EXP_ROOT` inside JSON ledgers.
- Do not generate broad cluster snapshots, duplicate endpoint responses, successful pod logs, or unrelated command
  output. The deployment ledger is the primary operational record.
- Never store secret values, kubeconfig contents, tokens, or registry credentials.
- Never overwrite raw AIPerf output or a previous iteration's benchmark files.
- Compare benchmark iterations only when `benchmark_audit.json` marks them valid and their benchmark-series identity
  matches the canonical plan.
- Do not treat a final recommendation as reproducible unless it points to the target workload, applied manifests,
  deployment ledger, benchmark plan, audit, summary, performance analysis, and raw benchmark artifacts.
