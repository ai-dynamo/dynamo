#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

TERMINALBENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${TERMINALBENCH_DIR}/pins.env"

HARBOR_SOURCE_DIR="${HARBOR_SOURCE_DIR:-${TERMINALBENCH_DIR}/.cache/harbor-${HARBOR_COMMIT}}"
HARBOR_BIN="${HARBOR_BIN:-${HARBOR_SOURCE_DIR}/.venv/bin/harbor}"

require_harbor() {
  if [[ ! -x "${HARBOR_BIN}" ]]; then
    echo "Pinned Harbor executable not found: ${HARBOR_BIN}" >&2
    echo "Run ${TERMINALBENCH_DIR}/bootstrap.sh first." >&2
    return 1
  fi

  if ! command -v uv >/dev/null; then
    echo "uv is required to verify the pinned Harbor environment" >&2
    return 1
  fi

  local actual_commit actual_remote actual_version tracked_changes
  actual_commit="$(git -C "${HARBOR_SOURCE_DIR}" rev-parse HEAD)"
  actual_remote="$(git -C "${HARBOR_SOURCE_DIR}" remote get-url origin)"
  tracked_changes="$(git -C "${HARBOR_SOURCE_DIR}" status --short --untracked-files=all)"
  actual_version="$(${HARBOR_BIN} --version | awk '{print $NF}')"
  if [[ "${actual_commit}" != "${HARBOR_COMMIT}" ]]; then
    echo "Harbor source mismatch: expected ${HARBOR_COMMIT}, found ${actual_commit}" >&2
    return 1
  fi
  if [[ "${actual_version}" != "${HARBOR_VERSION}" ]]; then
    echo "Harbor version mismatch: expected ${HARBOR_VERSION}, found ${actual_version}" >&2
    return 1
  fi
  if [[ "${actual_remote}" != "${HARBOR_REPOSITORY}" ]]; then
    echo "Harbor remote mismatch: expected ${HARBOR_REPOSITORY}, found ${actual_remote}" >&2
    return 1
  fi
  if [[ -n "${tracked_changes}" ]]; then
    echo "Harbor source checkout has tracked modifications: ${HARBOR_SOURCE_DIR}" >&2
    return 1
  fi
  if ! uv sync --directory "${HARBOR_SOURCE_DIR}" --frozen --no-dev --check >/dev/null; then
    echo "Harbor virtual environment differs from the pinned uv.lock" >&2
    return 1
  fi
}

require_docker() {
  if ! command -v docker >/dev/null; then
    echo "Docker CLI not found" >&2
    return 1
  fi
  if ! docker info >/dev/null; then
    echo "Docker daemon is not reachable" >&2
    return 1
  fi
  if ! docker compose version >/dev/null; then
    echo "Docker Compose v2 plugin is required" >&2
    return 1
  fi
}
