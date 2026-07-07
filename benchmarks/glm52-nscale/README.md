<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.2 NScale agent benchmark campaign

This directory contains the reproducible deployment, evaluation, result, and report
artifacts for comparing `nvidia/GLM-5.2-NVFP4` through four OpenAI-compatible stacks:

1. Dynamo + vLLM
2. standalone `vllm serve`
3. Dynamo + SGLang
4. standalone SGLang

The requested suites are BFCL v4, SWE-bench Verified, SWE-bench Pro,
SWE-bench Multilingual, and Terminal-Bench 2.1.

## Fairness controls

- Run one stack at a time on the same four clean B200 GPU UUIDs; every worker
  rejects a different allocation before model startup.
- Run each Dynamo/native pair twice with reversed order: phase `ab` runs Dynamo then
  native, and phase `ba` runs native then Dynamo. Tear down and create fresh serving
  pods between every stack deployment. Campaign closure therefore requires 40 full
  stack/suite/phase results, not 20 single-order results.
- Pin both members of each pair to the same framework image manifest digest and
  model checkpoint revision (`aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa`).
- Keep tensor parallelism, context length, cache dtype, concurrency caps, and sampling
  settings identical within each pair.
- Pin suite-specific sampling identically across all four stacks. BFCL uses its
  deterministic temperature-zero default; SWE and Terminal-Bench use GLM-5.2's
  published temperature 1.0 agent settings.
- Preserve task order and record per-instance paired disagreements, not only aggregate scores.
- Capture the applied manifest, image digest, package versions, GPU identity, effective command,
  request counts, token counts, errors, and wall time for every run.
- Bind every harness run to a privacy-safe runtime identity and require unchanged controller,
  pod, image, container, and restart identities before and after the guarded command.
- Revalidate physical GPU memory before every deployment. Scheduler accounting alone is not
  sufficient on this shared cluster.

## Deployment-local configuration

Copy `campaign.local.env.example` to `campaign.local.env` and fill in the Kubernetes
context, namespace, node, and four GPU UUIDs. Provision `nvcr-secret` and
`hf-token-secret` in that namespace before deployment. The local file is ignored by Git
and excluded from evaluation-runner source bundles. Committed recipes and
compact benchmark results therefore contain no live cluster identifiers or credentials.
For official BFCL web-search cases, optionally create `glm52-benchmark-secrets` with a
`SERPAPI_API_KEY` entry; the runner references that Secret as optional and never captures
its values.

## Layout

```text
campaign.env                  pinned model/runtime campaign configuration
campaign.local.env.example    untracked cluster-placement template
deploy/                       manifests and lifecycle scripts
artifact-storage/             pin-safe Cinder output migration and runner overlay
eval/                         pinned harness setup and execution wrappers
results/                      committed task-level summaries and score tables
report/                       report generator and final self-contained HTML
worklog.md                    timestamped experiment decisions and observations
```

Large trajectories, task containers, raw logs, and runtime identity captures are retained on
the NScale artifact PVC or in ignored local paths.
The branch stores immutable manifests, commands, compact machine-readable results, and the
self-contained HTML report.

## Current status

One of 40 official cells is imported: Dynamo + vLLM A/B SWE-bench Verified resolved
436/500 instances for an official score of 87.2%. The exact 500-image Verified cache
prefill completed and passed its executable catalog/binding gate. Its 507-file
runner-local checkpoint was exported before the pin-safe Cinder output-volume cutover.
Deployment recipes, source/runtime attestations, and harness completeness gates are ready.
Dynamo + vLLM passed exact text, forced-tool, automatic-tool, and BFCL validation smokes.
A corrected Terminal-Bench validation run completed all three pinned tasks at 3/3 reward
with zero exceptions. Those observations predate the canonical 409,600-token runtime
binding and are validation evidence only; no comparative score has been imported. The
serving stack is torn down and the evaluation runner is idle while the immutable campaign
source is pinned. Full BFCL remains gated on a `SERPAPI_API_KEY` for its 200 official
web-search cases.
