#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run from the repo root: fetches history, runs resolve_diff_base.py, appends outputs to $GITHUB_OUTPUT.
# Inputs are DIFF_* env vars set via `env:` (avoids template injection); unset ones expand empty, no `-u`.

set -eo pipefail

# Best-effort fetches into the shallow, tagless checkout; a miss means an empty base_sha.
# Depth 30 covers the baseline walks; fetching by SHA beats the much slower `--deepen`.
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
