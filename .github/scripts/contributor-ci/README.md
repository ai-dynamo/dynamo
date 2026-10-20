<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Trusted contributor CI routing

The native `pull_request` workflow is opt-in through the canonical repository
variable `TRUSTED_CONTRIBUTOR_CI_ENABLED=true`. It reads contributor standing
and routing code from main, plans images and CPU/GPU jobs, and checks out the
exact PR merge SHA only inside admitted contributor job containers. The main
policy reader in Velonix independently verifies every commit's author, signer,
signature, live PR head, workflow/run identity and actual runner assignment.
The routing job selects a cohort; it does not authorize hardware.

Images go to `210086341041.dkr.ecr.us-west-2.amazonaws.com/dynamo-pr-ci` with
PR/head/run/attempt/component/architecture tags. Mutable BuildKit exports use
`dynamo-pr-ci/cache` with PR/component/architecture tags. These repositories are
defined by the Velonix Terraform draft. EPP and test images stay in the same
contributor image repository. Worker discovery is confined to
`dynamo-pr-buildkit`, with no Kubernetes-driver or production-worker fallback.
IRSA credentials forwarded as BuildKit secrets can access only the PR cache.

The model-cache branch provides `hydrate-test-models` and the pytest runner's
`MODELS_DIR` support; merge that dependency before enabling this lane. Test jobs
read only the PR model store. MinIO is a disposable job service; its cached
image must be present before activation. The infrastructure hook template
supplies its server arguments. Existing canonical CI is unchanged while the
feature variable is absent. Once enabled, declared contributors leave the
automatic copied-branch approval flow and use the isolated lane. Explicit
maintainer approval remains governed by its existing policy.

The matrix builds selected backend/runtime/planner/frontend and sidecar images,
runs backend CPU tests on both architectures and AMD64 one-/two-GPU tests,
and retains the frontend image smoke check. Native sidecar binary integration,
the separate Rust/mypy stages and development-image builds still need parity
review against the canonical pipeline before this pilot can replace all of its
checks. Deployment, ACR copy, NGC publication and Triton Artifactory QA are
deliberately excluded. Other Triton CPU coverage remains selected.

Activation requires a trusted **required workflow** or dedicated GitHub App
check that verifies the expected workload at the exact head. The PR-owned
`Contributor workload result` is a report, not sufficient merge enforcement.
Labels and runner names are not access controls. In particular, a PR which
edits `.github/` must use the maintainer lane and cannot remove the external
merge requirement. Keep the feature disabled until this requirement, the
infrastructure, model publication, and signed/unsigned/stale/edited-workflow
canaries pass. This branch configures no repository rules or live workloads.
