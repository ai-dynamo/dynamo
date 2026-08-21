#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

framework="${IMAGE_FRAMEWORK:-dynamo}"
target="${IMAGE_TARGET:-base}"
build_target="${IMAGE_BUILD_TARGET:-}"
platform="${IMAGE_PLATFORM:-linux/amd64}"
cuda_version="${CUDA_VERSION:-13.0}"
push_image="${PUSH_IMAGE:-false}"
no_cache="${NO_CACHE:-false}"
commit_sha="${BUILDKITE_COMMIT:-local}"
image_tag="${IMAGE_TAG:-dynamo-buildkite-poc:${commit_sha:0:12}-${framework}-${target}}"
cache_run="${CACHE_RUN:-single}"
artifact_dir="${BUILDKITE_ARTIFACT_DIR:-artifacts/buildkite/image-build/${cache_run}}"
venv_dir="${BUILDKITE_BUILD_VENV:-.buildkite/.venv}"

case "${framework}" in
  dynamo|vllm|sglang|trtllm) ;;
  *) echo "Unsupported IMAGE_FRAMEWORK: ${framework}" >&2; exit 2 ;;
esac

case "${platform}" in
  linux/amd64|linux/arm64|linux/amd64,linux/arm64) ;;
  *) echo "Unsupported IMAGE_PLATFORM: ${platform}" >&2; exit 2 ;;
esac

case "${push_image}" in
  true|false) ;;
  *) echo "PUSH_IMAGE must be true or false" >&2; exit 2 ;;
esac

case "${no_cache}" in
  true|false) ;;
  *) echo "NO_CACHE must be true or false" >&2; exit 2 ;;
esac

mkdir -p "${artifact_dir}"
BUILDKITE_ARTIFACT_DIR="${artifact_dir}" SKIP_REMOTE_BUILDER_SMOKE=true \
  .buildkite/scripts/verify-remote-builder.sh

python3 -m venv "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --disable-pip-version-check \
  --requirement .buildkite/requirements.txt

render_args=(
  "--framework=${framework}"
  "--target=${target}"
  "--platform=${platform}"
  "--cuda-version=${cuda_version}"
  --output-short-filename
)
"${venv_dir}/bin/python" container/render.py "${render_args[@]}"
cp container/rendered.Dockerfile "${artifact_dir}/rendered.Dockerfile"

build_args=(
  --progress=plain
  --platform "${platform}"
  --file container/rendered.Dockerfile
  --tag "${image_tag}"
  --build-arg "DYNAMO_COMMIT_SHA=${commit_sha}"
)

if [[ -n "${build_target}" ]]; then
  build_args+=(--target "${build_target}")
fi

if [[ "${no_cache}" == "true" ]]; then
  build_args+=(--no-cache)
fi

if [[ "${push_image}" == "true" ]]; then
  build_args+=(--push)
else
  build_args+=(--output type=cacheonly)
fi

start_epoch="$(date +%s)"
set +e
docker buildx build "${build_args[@]}" . 2>&1 | tee "${artifact_dir}/build.log"
build_exit_code="${PIPESTATUS[0]}"
set -e
end_epoch="$(date +%s)"

jq -n \
  --arg framework "${framework}" \
  --arg target "${target}" \
  --arg build_target "${build_target}" \
  --arg platform "${platform}" \
  --arg cuda_version "${cuda_version}" \
  --arg image_tag "${image_tag}" \
  --arg cache_run "${cache_run}" \
  --argjson push_image "${push_image}" \
  --argjson duration_seconds "$((end_epoch - start_epoch))" \
  --argjson exit_code "${build_exit_code}" \
  '{
    framework: $framework,
    target: $target,
    build_target: $build_target,
    platform: $platform,
    cuda_version: $cuda_version,
    image_tag: $image_tag,
    cache_run: $cache_run,
    push_image: $push_image,
    duration_seconds: $duration_seconds,
    exit_code: $exit_code
  }' > "${artifact_dir}/result.json"

if (( build_exit_code != 0 )); then
  exit "${build_exit_code}"
fi

if command -v buildkite-agent >/dev/null 2>&1; then
  buildkite-agent annotate --style success --context "image-build-${cache_run}" \
    "${cache_run^} build of \`${framework}:${target}\` for \`${platform}\` completed in $((end_epoch - start_epoch)) seconds."
fi
