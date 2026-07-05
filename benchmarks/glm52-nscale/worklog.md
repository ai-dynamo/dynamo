<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Worklog

## 2026-07-04 setup

- Created `rmccormick/glm52` from current `origin/main` at `aeda23b483` in an isolated
  worktree.
- Confirmed the NScale Kubernetes context and an active empty campaign namespace with a
  bound 50 TiB RWX `shared-model-cache` PVC.
- Confirmed the shared cache contains `nvidia/GLM-5.2-NVFP4` at
  `/opt/models/models--nvidia--GLM-5.2-NVFP4`.
- Pinned the vLLM and SGLang Dynamo nightlies from commit `5245c0fa6a` so the Dynamo and
  standalone member of each framework pair use the same Python/framework environment.
- Rejected an apparently idle candidate node because DCGM showed about 147.6 GiB allocated
  on every device. Selected another 8-GPU B200 node for sequential TP4 runs only after
  scheduler and physical-memory validation showed at least four clean GPUs.
- Replaced the historical 8,192-token smoke limit with a 409,600-token serving envelope for
  full-history coding-agent workloads. GLM-5.2's published SWE settings use 400K context;
  Terminal-Bench independently advertises its published 262,144-token window to the agent.
- Established sequential same-node testing rather than simultaneous two-node testing to avoid
  node and co-tenancy bias. Every stack is torn down and GPU cleanliness is rechecked before
  the next stack starts.

## 2026-07-05 harness and deployment hardening

- Deployed Dynamo + vLLM with `nvidia/GLM-5.2-NVFP4`, TP4/EP4, FP8 KV cache,
  262,144-token context, and the served alias `zai-org/GLM-5.2`. The cold start loaded
  47 shards (432.90 GiB), used 106.6 GiB per GPU for model weights, exposed 44.24 GiB
  per GPU for KV cache, and reported 995,520 KV tokens.
- Pinned the vLLM and SGLang runtime references to their Linux amd64 manifest digests.
  Recorded and enforced the same four physical B200 UUIDs on every serving deployment.
- Aligned the Dynamo vLLM frontend with standalone vLLM by enabling automatic tool choice
  and keeping the `glm47` tool parser plus `glm45` reasoning parser at the chat-processing
  layer. Exact text, forced-tool, and unforced-tool smoke requests pass.
- BFCL native Chat Completions produced valid results for the 5/5 selected smoke cases.
  Added an exact smoke gate after discovering upstream partial evaluation could exit zero on
  malformed output. Hardened full-run accounting to require 5,217 generated IDs (including
  111 prerequisites), 5,106 scored IDs, all 22 official categories, zero inference errors,
  and immutable resume metadata. Every request now has a recorded 64,000-token output cap.
- Found that the first Terminal-Bench runner had Docker CLI without Compose v2. Harbor emitted
  three infrastructure exceptions but exited zero; the original wrapper incorrectly printed
  PASS. Pinned Docker CLI/daemon 27.5.1 with Compose 2.33.0, shared `/artifacts` and
  `/workspace` into DinD, and made trial exceptions or missing rewards invalidate the run.
- Found that Harbor 0.17.1 records `max_output_tokens` for context accounting but does not
  forward it on the Chat Completions path. Added explicit Terminus-2 call kwargs and rejected
  the initial 8,192-, 32,768-, and 128,000-token greedy calibrations as non-representative:
  temperature-zero turns either reached the smaller caps or decoded continuously until the
  128,000-token trial was operator-cancelled after 436 seconds. The GLM-5.2 model card specifies
  Terminus-2 JSON parsing, temperature 1.0, top-p 1.0, 48,000 output tokens, 500 episodes, and
  a 256K context with a four-hour task timeout. The campaign now pins those exact settings.
  A first replacement smoke proved repeated terminating turns and active terminal progress,
  then exposed Harbor's 900-second default timeout on `write-compressor`; that result was
  rejected as infrastructure-invalid and the timeout multiplier was pinned to 16 (14,400s).
- The corrected Terminal-Bench validation smoke then completed all three exact pinned tasks
  in 37 minutes at 3/3 reward, mean 1.000, and zero exceptions. It is retained only as
  validation evidence because it predates the 409,600-token deployment/runtime binding and
  is not eligible for comparative report import.
- Moved all harness checkouts and virtual environments to the persistent artifact PVC. Runner
  source syncs are staged, SHA-256 verified, and recorded with branch, commit, dirty-path count,
  and exact bundle digest.
- Hardened all SWE-bench adapters with exact run-scope manifests, strict incomplete/error
  rejection, prediction-digest cache keys so regenerated patches cannot inherit stale
  evaluator output, exact task-image identities, and a live per-run Python environment
  freeze. The harness test suite passes 31/31.
- Confirmed `SERPAPI_API_KEY` is not present in the user environment or available Kubernetes
  secrets. Pinned BFCL source confirms the 200 web-search cases have no credential-free
  official path; a full leaderboard-comparable run requires a provisioned key. Other suites
  continue independently.
- Rebootstrapped BFCL after bounding requests and reran the new fail-closed smoke gate against
  Dynamo + vLLM. It passed all five exact IDs at 100% with zero inference errors: 30 model
  queries, 133,341 input tokens, 3,302 output tokens, and 48.41 aggregate query-seconds.
- Removed full Kubernetes Node dumps from committed runtime capture, restricted event capture
  to campaign pods, and added checkpoint provenance from Hugging Face download metadata. The
  cached model revision is `aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa`; all 47 weight-shard
  content identifiers are captured without rereading 432.9 GiB of weights.
- Expanded comparative closure from one temporal order to two fresh-deployment phases per
  suite: `ab` runs Dynamo then native and `ba` reverses the order. The report now requires
  40 full variant/suite/phase cells and computes order-balanced aggregates.
- Added a single privacy-safe runtime contract shared by BFCL, SWE-bench, Terminal-Bench,
  guarded execution, and result import. It binds the pinned source/template, model revision,
  effective context, framework image, hashed controller/pod/node identities, GPU-set/driver
  identity, and Dynamo/Grove image digests without committing live cluster identifiers.
- Added pre/post continuity attestations and an atomic PVC run lock. Imports reject serving
  restarts, identity changes, phase mismatches, concurrent runs, source drift, and missing
  task-image or environment identities.
- A pre-commit audit found and fixed fail-open reproducibility paths before any comparative
  result was accepted: full Terminal-Bench settings were still overrideable, persistent
  evaluator environments were only checked at bootstrap, a stale CPU-only controller could
  evade the fresh-deployment check, and synced evaluator source was not re-hashed at run time.
  Every suite now binds the exact source tree and live environment into its runtime envelope;
  deployment refuses any surviving campaign resource.
- Corrected the report closure contract to require 40 distinct serving deployments and prove
  the full per-suite order `AB Dynamo < AB native < BA native < BA Dynamo`. The BFCL importer
  now recomputes the pinned v4 weighted headline from category counts instead of trusting the
  aggregate CSV value, and report freshness includes every task-level and disagreement
  sidecar.
