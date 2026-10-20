#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

for name in BUILD_FRAMEWORK BUILD_TARGET BUILD_ARCH BUILD_CUDA BUILD_IMAGE BUILD_CACHE; do
  [[ -n "${!name:-}" ]] || { echo "Missing ${name}" >&2; exit 1; }
done
registry=210086341041.dkr.ecr.us-west-2.amazonaws.com
[[ "$BUILD_IMAGE" == "$registry/dynamo-pr-ci:pr-"* ]] || exit 1
[[ "$BUILD_CACHE" == "$registry/dynamo-pr-ci/cache:pr-"* ]] || exit 1
[[ "$BUILD_ARCH" == amd64 || "$BUILD_ARCH" == arm64 ]] || exit 1
export AWS_DEFAULT_REGION=us-west-2
export BUILDKIT_NAMESPACE=dynamo-pr-buildkit
export SCCACHE_BUCKET=dynamo-ci-210086341041-us-west-2-pr-sccache

# No Kubernetes-driver fallback: the job has no host-cluster API credential.
route_file=$(mktemp)
trap 'rm -f "$route_file"' EXIT
GITHUB_OUTPUT="$route_file" bash .github/scripts/route_buildkit.sh --arch "$BUILD_ARCH" --flavor general
addresses=$(sed -n "s/^general_${BUILD_ARCH}=//p" "$route_file")
[[ -n "$addresses" ]] || { echo 'No contributor BuildKit worker is available' >&2; exit 1; }
endpoint=${addresses%%,*}
[[ "$endpoint" == tcp://buildkit-*.dynamo-pr-buildkit.svc.cluster.local:* ]] || exit 1
builder="pr-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}-${BUILD_FRAMEWORK}-${BUILD_TARGET}-${BUILD_ARCH}"
docker buildx create --name "$builder" --driver remote --platform "linux/$BUILD_ARCH" "$endpoint" --use
docker buildx inspect --bootstrap
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin "$registry"

cache_args=(--cache-from "type=registry,ref=$BUILD_CACHE"
  --cache-to "type=registry,ref=$BUILD_CACHE,mode=max,image-manifest=true,oci-mediatypes=true")
secrets=()
if [[ -n "${AWS_WEB_IDENTITY_TOKEN_FILE:-}" && -f "$AWS_WEB_IDENTITY_TOKEN_FILE" && -n "${AWS_ROLE_ARN:-}" ]]; then
  secrets+=(--secret "id=aws-web-identity-token,src=$AWS_WEB_IDENTITY_TOKEN_FILE"
    --secret id=aws-role-arn,env=AWS_ROLE_ARN)
else
  echo 'Contributor sccache workload identity is missing' >&2
  exit 1
fi
if [[ "$BUILD_FRAMEWORK" == sidecar ]]; then
  dockerfile=lib/sidecar/Dockerfile
else
  python3 container/render.py --framework "$BUILD_FRAMEWORK" --target "$BUILD_TARGET" \
    --platform "linux/$BUILD_ARCH" --cuda-version "$BUILD_CUDA" --output-short-filename
  dockerfile=container/rendered.Dockerfile
fi
extra_args=(--build-arg USE_SCCACHE=true)
if [[ "$BUILD_TARGET" == frontend ]]; then
  epp_image="${BUILD_IMAGE}-epp"
  docker buildx build --progress plain --platform "linux/$BUILD_ARCH" \
    -f deploy/inference-gateway/ext-proc/Dockerfile --build-context dynamo=. \
    --tag "$epp_image" --push "${secrets[@]}" \
    --cache-from "type=registry,ref=${BUILD_CACHE}-epp" \
    --cache-to "type=registry,ref=${BUILD_CACHE}-epp,mode=max,image-manifest=true,oci-mediatypes=true" \
    --build-arg USE_SCCACHE=true --build-arg "SCCACHE_BUCKET=$SCCACHE_BUCKET" \
    --build-arg SCCACHE_REGION=us-west-2 deploy/inference-gateway/ext-proc
  extra_args+=(--build-arg "EPP_IMAGE=$epp_image")
fi
docker buildx build --progress plain --platform "linux/$BUILD_ARCH" \
  --target "$BUILD_TARGET" -f "$dockerfile" --tag "$BUILD_IMAGE" --push \
  "${cache_args[@]}" "${secrets[@]}" "${extra_args[@]}" \
  --build-arg "SCCACHE_BUCKET=$SCCACHE_BUCKET" --build-arg SCCACHE_REGION=us-west-2 .

if [[ "$BUILD_FRAMEWORK" != sidecar ]]; then
  test_target=test_image
  [[ "$BUILD_TARGET" != planner ]] || test_target=test_planner
  docker buildx build --progress plain --platform "linux/$BUILD_ARCH" \
    --target "$test_target" -f container/Dockerfile.test \
    --build-arg "BASE_IMAGE=$BUILD_IMAGE" --tag "$BUILD_IMAGE-test" --push \
    --cache-from "type=registry,ref=$BUILD_CACHE" .
fi
