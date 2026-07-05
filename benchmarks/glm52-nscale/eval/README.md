<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Evaluation harnesses

The macOS control host is not a valid execution environment for the official x86_64 SWE and
Terminal-Bench task images. `runner.yaml` creates an amd64 NScale CPU pod with a Docker-in-Docker
sidecar and a persistent artifact volume. The runner targets the active inference service over
cluster DNS, avoiding fragile multi-day port forwards.

## Lifecycle

```bash
./deploy-runner.sh
./sync-runner.sh
./capture-runner-identity.sh
./exec-runner.sh bash
./fetch-result.sh --help
./teardown-runner.sh
```

Before changing a live runner resource, `deploy-runner.sh` verifies that local
`campaign.env` and the complete `eval/` tree still match `campaign.source_commit`.
It then syncs the adapters and `campaign.env` from that exact commit into the
runner's ephemeral `/workspace`, verifies the transferred archive hash, and captures
the runner identity. Both deploy and sync refuse to proceed while a harness process,
Docker task container, tmux session, or campaign lock is active. A changed runner
manifest also refuses implicit pod deletion; wait for active work and invoke
`teardown-runner.sh` explicitly. Harness checkouts, virtual environments, caches, raw
jobs, and results use `/artifacts` so they survive runner replacement.
The source manifest records relative file content hashes and normalized permission
modes. Every harness rehashes the current runner tree, requires its commit to equal the
serving deployment recipe, and embeds the resulting path-free `campaign_source`
identity in both run metadata and the runtime evaluator envelope.

The pod becomes Ready only after package setup finishes and the pinned Docker 27.5.1
CLI, Compose plugin, and Docker-in-Docker daemon all respond. `deploy-runner.sh` also
checks the setup marker and both Docker commands and fails fast if either container exits.

Runner captures are written to `../results/runtime/eval-runner/<UTC timestamp>/`.
They include source-bundle provenance, pinned image IDs, selected Kubernetes resource
identity, and toolchain versions. The capture intentionally does not read environment
variables, Kubernetes Secrets, container logs, or Docker registry configuration.

The 1 TiB `glm52-benchmark-artifacts` PVC is intentionally retained across runner restarts.
Raw trajectories, cloned harnesses, Docker metadata, and evaluator logs live under
`/artifacts/glm52-nscale`. Compact results and the final HTML report are copied back into this
branch.

## Guarded execution and phases

Every real suite command runs through `run-guarded.sh`. It takes an atomic PVC lock,
requires the active deployment binding to match the selected variant and phase, records
privacy-safe controller/pod/container identities before and after the command, and refuses
results if a serving container restarted or changed. Place its
`runtime-continuity.json` beside the compact suite artifact so the importer can verify it.
If a command exits nonzero while the runtime stays stable, a retry validates and archives
that failed attestation under `runtime-continuity.failures/<sha256>.json` before running.
A successful attestation is never overwritten.

Full results use two fresh-deployment phases: `ab` runs Dynamo then native and `ba` runs
native then Dynamo. Validation smokes use phase `validation` and are never imported as
full scores. The exact deployment and suite sequence is in `../RUNBOOK.md`.

## Endpoint mapping

| Variant | Cluster API base |
|---|---|
| Dynamo + vLLM | `http://glm52-dynamo-vllm-frontend:8000/v1` |
| vLLM serve | `http://glm52-vllm-serve:8000/v1` |
| Dynamo + SGLang | `http://glm52-dynamo-sglang-frontend:8000/v1` |
| SGLang serve | `http://glm52-sglang-serve:8000/v1` |

Run the suite-specific smoke commands before a full campaign. Full-suite wrappers refuse to run
when their pinned source checkout or required credential is missing.

## Fetch compact evidence

The artifact PVC is mounted only in the runner. Fetch an importer-ready directory through
the suite allowlist instead of copying a raw run tree:

```bash
./fetch-result.sh bfcl /artifacts/glm52-nscale/bfcl/dynamo-vllm/ab/full ./fetched/bfcl
./fetch-result.sh swebench /artifacts/glm52-nscale/swebench/results/dynamo-vllm-ab/verified ./fetched/swe
./fetch-result.sh terminalbench /artifacts/glm52-nscale/terminalbench/summaries/dynamo-vllm/ab ./fetched/terminal
```

The fetcher rejects paths outside `/artifacts/glm52-nscale`, symlinked remote
directories/files, and existing local destinations. It transfers only importer-required
compact evidence, never trajectories, task containers, or logs.
