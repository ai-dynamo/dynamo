#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
campaign_relative="benchmarks/glm52-nscale"
source_commit="$(jq -er '.campaign.source_commit
  | select(type == "string" and test("^[0-9a-f]{40}$"))' \
  "${CAMPAIGN_DIR}/results/summary.json")"

if ! git -C "${REPO_ROOT}" cat-file -e "${source_commit}^{commit}" 2>/dev/null; then
  echo "Pinned campaign source commit is unavailable: ${source_commit}" >&2
  exit 1
fi
if ! git -C "${REPO_ROOT}" merge-base --is-ancestor "${source_commit}" HEAD; then
  echo "Pinned campaign source commit is not an ancestor of HEAD: ${source_commit}" >&2
  exit 1
fi
tracked_drift="$(git -C "${REPO_ROOT}" diff --name-only "${source_commit}" -- \
  "${campaign_relative}/campaign.env" "${campaign_relative}/deploy" \
  "${campaign_relative}/eval/runtime_binding.py")"
untracked_drift="$(git -C "${REPO_ROOT}" ls-files --others --exclude-standard -- \
  "${campaign_relative}/campaign.env" "${campaign_relative}/deploy" \
  "${campaign_relative}/eval/runtime_binding.py")"
if [[ -n "${tracked_drift}" || -n "${untracked_drift}" ]]; then
  echo "Deployment sources differ from pinned commit ${source_commit}:" >&2
  printf '%s\n%s\n' "${tracked_drift}" "${untracked_drift}" | sed '/^$/d' >&2
  exit 1
fi

printf '%s\n' "${source_commit}"
