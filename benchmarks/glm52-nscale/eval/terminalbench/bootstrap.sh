#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

# uv-managed interpreter paths are embedded into the persistent venv. Keep the
# interpreter beside the checkout so a runner pod replacement does not leave a
# dangling /root/.local/share/uv symlink.
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$(dirname "${HARBOR_SOURCE_DIR}")/uv-python}"
export UV_PYTHON_INSTALL_DIR

for command in git uv; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Required command not found: ${command}" >&2
    exit 1
  fi
done

if [[ -e "${HARBOR_SOURCE_DIR}" && ! -d "${HARBOR_SOURCE_DIR}/.git" ]]; then
  echo "Refusing to replace non-Harbor path: ${HARBOR_SOURCE_DIR}" >&2
  exit 1
fi

if [[ ! -d "${HARBOR_SOURCE_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${HARBOR_SOURCE_DIR}")"
  git init --quiet "${HARBOR_SOURCE_DIR}"
  git -C "${HARBOR_SOURCE_DIR}" remote add origin "${HARBOR_REPOSITORY}"
  git -C "${HARBOR_SOURCE_DIR}" fetch --quiet --depth=1 origin "${HARBOR_COMMIT}"
  git -C "${HARBOR_SOURCE_DIR}" checkout --quiet --detach FETCH_HEAD
fi

actual_remote="$(git -C "${HARBOR_SOURCE_DIR}" remote get-url origin)"
actual_commit="$(git -C "${HARBOR_SOURCE_DIR}" rev-parse HEAD)"
if [[ "${actual_remote}" != "${HARBOR_REPOSITORY}" ]]; then
  echo "Harbor remote mismatch: expected ${HARBOR_REPOSITORY}, found ${actual_remote}" >&2
  exit 1
fi
if [[ "${actual_commit}" != "${HARBOR_COMMIT}" ]]; then
  echo "Harbor commit mismatch: expected ${HARBOR_COMMIT}, found ${actual_commit}" >&2
  exit 1
fi
if [[ -n "$(git -C "${HARBOR_SOURCE_DIR}" status --short --untracked-files=all)" ]]; then
  echo "Harbor source checkout is not clean: ${HARBOR_SOURCE_DIR}" >&2
  exit 1
fi

# The upstream uv.lock is part of HARBOR_COMMIT, so both Harbor and all
# transitive Python dependencies are immutable for this campaign.
uv sync \
  --directory "${HARBOR_SOURCE_DIR}" \
  --frozen \
  --no-dev

require_harbor

python_bin="${HARBOR_SOURCE_DIR}/.venv/bin/python"
resolved_version="$(${python_bin} -c 'import importlib.metadata; print(importlib.metadata.version("harbor"))')"
if [[ "${resolved_version}" != "${HARBOR_VERSION}" ]]; then
  echo "Installed package mismatch: expected ${HARBOR_VERSION}, found ${resolved_version}" >&2
  exit 1
fi

"${python_bin}" "${SCRIPT_DIR}/verify_dataset.py" \
  --dataset "${TERMINALBENCH_DATASET}" \
  --expected-content-hash "${TERMINALBENCH_DATASET_CONTENT_HASH}" \
  --expected-version-id "${TERMINALBENCH_DATASET_VERSION_ID}" \
  --expected-tasks "${TERMINALBENCH_TASK_COUNT}"

echo "Harbor bootstrap PASS"
echo "  source:  ${HARBOR_SOURCE_DIR}"
echo "  commit:  ${actual_commit}"
echo "  version: ${resolved_version}"
echo "  binary:  ${HARBOR_BIN}"
echo "  dataset: ${TERMINALBENCH_DATASET} (${TERMINALBENCH_TASK_COUNT} tasks)"
