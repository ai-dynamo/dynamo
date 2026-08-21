#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

artifact_dir="${BUILDKITE_ARTIFACT_DIR:-artifacts/buildkite/remote-builder}"
mkdir -p "${artifact_dir}"

docker version | tee "${artifact_dir}/docker-version.txt"
docker buildx version | tee "${artifact_dir}/buildx-version.txt"
docker buildx ls | tee "${artifact_dir}/buildx-ls.txt"
docker buildx inspect --bootstrap | tee "${artifact_dir}/buildx-inspect.txt"

driver="$({ awk -F: '
  tolower($1) ~ /^[[:space:]]*driver[[:space:]]*$/ {
    value = $2
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
    print tolower(value)
    exit
  }
' "${artifact_dir}/buildx-inspect.txt"; } || true)"

if [[ "${driver}" != "remote" ]]; then
  message="Expected Buildkite's hosted remote Docker builder, but the active Buildx driver is '${driver:-unknown}'."
  if command -v buildkite-agent >/dev/null 2>&1; then
    buildkite-agent annotate --style error --context remote-builder "${message}"
  fi
  echo "${message}" >&2
  exit 1
fi

if [[ "${SKIP_REMOTE_BUILDER_SMOKE:-false}" != "true" ]]; then
  docker buildx build \
    --progress=plain \
    --platform linux/amd64 \
    --output type=cacheonly \
    --file .buildkite/smoke/Dockerfile \
    .buildkite/smoke 2>&1 | tee "${artifact_dir}/smoke-build.log"
fi

if command -v buildkite-agent >/dev/null 2>&1; then
  buildkite-agent annotate --style success --context remote-builder \
    "Hosted remote Docker builder verified with a cache-only Buildx build."
fi
