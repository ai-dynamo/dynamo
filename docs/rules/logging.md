# Logging Rules

Keep Dynamo agent artifacts reproducible and easy to inspect.

## Run Layout

- `recipe-explorer` creates `EXP_ROOT=runs/<EXP_ID>/` for one optimization job.
- `recipe-deployer` creates `DEPLOY_ROOT=${EXP_ROOT}/artifacts/deploy-iter-<NNN>/` for each new candidate.
- Keep retries and compatibility patches for one candidate in the same `DEPLOY_ROOT`.
- Keep every previous iteration directory and its successful YAML unchanged after retiring its DGD.

## Records

- Record the exact commands run, manifest paths applied, namespace, Kubernetes context, selected recipe, and artifact
  paths.
- Prefer paths relative to `EXP_ROOT` inside JSON ledgers so the experiment directory can be moved or shared.
- Keep one final copy of each used manifest type in `applied_manifests/`. Update run-scoped copies in place during
  bring-up, and record the source path, patch history, and reapply history in `deployment_ledger.json`.
- Create `logs/` only for targeted failure output needed beyond the ledger excerpt. Do not generate broad cluster
  snapshots, duplicate endpoint responses, successful pod logs, or unrelated command output.
- Do not retain numbered intermediate manifest copies.

## Safety

- Never write secret values, tokens, kubeconfig contents, or private registry credentials into logs or artifacts.
- Do not scatter new deployment or benchmark artifacts across unrelated top-level directories.
