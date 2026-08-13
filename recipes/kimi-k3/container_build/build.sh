#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=versions.env
source "$recipe_dir/versions.env"

image="${IMAGE:?set IMAGE to a writable registry tag, for example registry.example.com/project/dynamo-vllm:arm64}"
output_dir="${OUTPUT_DIR:-$recipe_dir/out}"
validate_gpu="${VALIDATE_GPU:-1}"
cargo_build_jobs="${CARGO_BUILD_JOBS:-16}"

if [[ "$image" == *@* ]]; then
    echo "IMAGE must be a writable tag, not a digest: $image" >&2
    exit 2
fi
if [[ "${image##*/}" != *:* ]]; then
    echo "IMAGE must include an explicit tag: $image" >&2
    exit 2
fi
if [[ "$validate_gpu" != 0 && "$validate_gpu" != 1 ]]; then
    echo "VALIDATE_GPU must be 0 or 1" >&2
    exit 2
fi
if [[ "$(uname -m)" != aarch64 ]]; then
    echo "this recipe must build natively on an aarch64 host" >&2
    exit 2
fi

for command in docker git python3 sha256sum; do
    command -v "$command" >/dev/null || {
        echo "missing required command: $command" >&2
        exit 2
    }
done
docker buildx version >/dev/null

mkdir -p "$output_dir"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/dynamo-arm64-build.XXXXXX")"
builder="dynamo-arm64-clean-$$"

cleanup() {
    docker buildx rm --force "$builder" >/dev/null 2>&1 || true
    if [[ -d "$work_dir" ]]; then
        find "$work_dir" -depth -delete
    fi
}
trap cleanup EXIT

exec > >(tee "$output_dir/build.log") 2>&1

echo "started_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "architecture=$(uname -m)"
echo "destination=$image"

source_dir="$work_dir/dynamo"
git clone --filter=blob:none --no-checkout "$DYNAMO_REPOSITORY" "$source_dir"
git -C "$source_dir" fetch --depth=1 origin "$DYNAMO_COMMIT"
git -C "$source_dir" checkout --detach "$DYNAMO_COMMIT"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$DYNAMO_COMMIT"

(
    cd "$source_dir"
    python3 container/render.py \
        --framework vllm \
        --device cuda \
        --target runtime \
        --platform linux/arm64 \
        --cuda-version 13.0 \
        --output-short-filename
)

dockerfile="$source_dir/container/rendered.Dockerfile"
python3 "$recipe_dir/prepare_dockerfile.py" "$dockerfile"
actual_dockerfile_sha="$(sha256sum "$dockerfile" | awk '{print $1}')"
if [[ "$actual_dockerfile_sha" != "$DOCKERFILE_SHA256" ]]; then
    echo "rendered Dockerfile checksum mismatch" >&2
    echo "expected: $DOCKERFILE_SHA256" >&2
    echo "actual:   $actual_dockerfile_sha" >&2
    exit 1
fi

# Do not send Git history into BuildKit.
find "$source_dir/.git" -depth -delete

docker buildx create \
    --name "$builder" \
    --driver docker-container \
    --use
docker buildx inspect --bootstrap "$builder"

docker buildx build \
    --builder "$builder" \
    --platform linux/arm64 \
    --target runtime \
    --file "$dockerfile" \
    --build-arg "BASE_IMAGE=$CUDA_BASE_IMAGE" \
    --build-arg "BASE_IMAGE_TAG=$CUDA_BASE_TAG@$CUDA_BASE_DIGEST" \
    --build-arg "WHEEL_BUILDER_IMAGE=$WHEEL_BUILDER_IMAGE@$WHEEL_BUILDER_DIGEST" \
    --build-arg "RUNTIME_IMAGE=$VLLM_IMAGE" \
    --build-arg "RUNTIME_IMAGE_TAG=$VLLM_TAG@$VLLM_ARM64_DIGEST" \
    --build-arg "DYNAMO_COMMIT_SHA=$DYNAMO_COMMIT" \
    --build-arg "CARGO_BUILD_JOBS=$cargo_build_jobs" \
    --no-cache \
    --pull \
    --provenance=false \
    --metadata-file "$output_dir/build-metadata.json" \
    --tag "$image" \
    --push \
    "$source_dir"

docker buildx imagetools inspect "$image" | tee "$output_dir/image-inspect.txt"
digest="$(sed -n 's/^Digest:[[:space:]]*//p' "$output_dir/image-inspect.txt" | head -1)"
if [[ ! "$digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "could not resolve pushed manifest digest" >&2
    exit 1
fi

image_pin="$image@$digest"
printf '%s\n' "$image_pin" | tee "$output_dir/image-pin.txt"
docker buildx imagetools inspect --raw "$image_pin" > "$output_dir/manifest.json"

if [[ "$validate_gpu" == 1 ]]; then
    OUTPUT_DIR="$output_dir" "$recipe_dir/validate-image.sh" "$image_pin"
fi

echo "completed_at=$(date --iso-8601=seconds)"
echo "image=$image_pin"
