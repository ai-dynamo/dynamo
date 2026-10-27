<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Logging Rules

Keep Dynamo agent artifacts reproducible and easy to inspect.

## Run Layout

- `user-interviewer` creates `EXP_ROOT=runs/<EXP_ID>/`, its canonical `user_workload.yaml`, and the immutable
  `inputs/user_provided_dgd.yaml`.
- `recipe-deployer` creates `DEPLOY_ROOT=${EXP_ROOT}/artifacts/deploy-iter-<NNN>/` for each new candidate.
- Keep retries and compatibility patches for one candidate in the same `DEPLOY_ROOT`.
- Keep every previous iteration directory and its successful YAML unchanged after retiring its DGD, except for the retiring role writing `torn_down_at` into that iteration's `deployment_ledger.json`.

## Records

- Record the exact commands run, assigned DGD path and SHA256, manifest paths applied, namespace, Kubernetes context,
  and artifact paths.
- Prefer paths relative to `EXP_ROOT` inside JSON ledgers so the experiment directory can be moved or shared.
- Keep one final copy of each used manifest type in `applied_manifests/`. Update run-scoped copies in place during
  bring-up, and record the source path, patch history, and reapply history in `deployment_ledger.json`.
- Create `logs/` only for targeted failure output needed beyond the ledger excerpt. Do not generate broad cluster
  snapshots, duplicate endpoint responses, successful pod logs, or unrelated command output.
- Do not retain numbered intermediate manifest copies.
- Maintain a `reasoning_transcript.md` document that is a long-running log of the agent's reasoning, key decisions, steps, rationale, status, etc. Include timestamps for each entry and keep it nicely organized and formatted for humans to read and review.

## Safety

- Never write secret values, tokens, kubeconfig contents, or private registry credentials into logs or artifacts.
- Do not scatter new deployment or benchmark artifacts across unrelated top-level directories.
