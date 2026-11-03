#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Prepare the shallow build checkout and run resolve_diff_base.py, appending its
# outputs to $GITHUB_OUTPUT. Shared by the "Resolve diff baseline" steps in
# shared-build-image.yml, shared-build-sidecar.yml, and the
# build-deploy-component action. Run from the repository root.
#
# Inputs come from the environment, which the callers set through `env:` so
# workflow values never reach a script body (zizmor "template-injection"):
#   DIFF_EVENT_CONTEXT, DIFF_CURRENT_BRANCH, DIFF_BASE_BRANCH,
#   DIFF_MERGE_BASE_SHA, DIFF_CURRENT_SHA, DIFF_ARTIFACT_PREFIX, DIFF_REPO,
#   DIFF_RUN_ID, GITHUB_OUTPUT; GH_TOKEN is read by the resolver.
# Unset inputs expand to empty strings, as they did inline, so no `-u`.

set -eo pipefail

# The build checkout is shallow with no tags. Make the refs each baseline rule
# needs reachable (all best-effort; a miss just yields an empty base_sha and a
# baseline_unavailable diff). Depth 30 gives the first-parent baseline walks
# enough history to find an artifact. Fetch HEAD by SHA with an explicit depth;
# `--deepen` is far slower here.
git fetch --no-tags --depth=31 origin "$(git rev-parse HEAD)" --quiet || true # HEAD~N (main post-merge walk)
# Only the release rule reads tags. With no refspec, this fetch also downloads
# every branch on origin, so run it only for release builds.
if [[ "${DIFF_BASE_BRANCH}" == release/* || "${DIFF_CURRENT_BRANCH}" == release/* ]]; then
  git fetch --tags --quiet || true # vX.Y.Z tags (release branch)
fi
if [ "${DIFF_BASE_BRANCH}" = "main" ] && [ -n "${DIFF_MERGE_BASE_SHA}" ]; then
  git fetch --no-tags --depth=30 origin "${DIFF_MERGE_BASE_SHA}" --quiet || true # stable PR fork-point walk
fi

CURRENT_VERSION=$(grep -m1 -E '^version[[:space:]]*=' pyproject.toml | sed -E 's/.*"([^"]+)".*/\1/' || true)

python3 container/compliance/resolve_diff_base.py \
  --event-context "${DIFF_EVENT_CONTEXT}" \
  --current-branch "${DIFF_CURRENT_BRANCH}" \
  --base-branch "${DIFF_BASE_BRANCH}" \
  --merge-base-sha "${DIFF_MERGE_BASE_SHA}" \
  --current-sha "${DIFF_CURRENT_SHA}" \
  --current-version "${CURRENT_VERSION}" \
  --artifact-prefix "${DIFF_ARTIFACT_PREFIX}" \
  --repo "${DIFF_REPO}" \
  --current-run-id "${DIFF_RUN_ID}" \
  >> "${GITHUB_OUTPUT}"
