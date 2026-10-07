#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build and push the multi-arch EPP image for one commit, or reuse an image
# that CI already built from the same inputs.
#
# The reuse key is a hash of the git objects that the EPP Dockerfile copies
# from the repository, plus .dockerignore and this script. Builds on main and
# release branches tag inputs-<key>. Builds on any other ref tag
# <ref>-inputs-<key>. A build reuses only an image from main, a release
# branch, or its own ref, so main never ships an image that a pull request
# built.
#
# Required environment: EPP_IMAGE_REPO, EPP_SOURCE_SHA, GITHUB_REF_NAME,
# GITHUB_OUTPUT. Optional: DOCKER_PROXY, NO_CACHE, EPP_PLATFORMS, MAKE, and the
# sccache variables SCCACHE_S3_BUCKET, AWS_DEFAULT_REGION,
# AWS_WEB_IDENTITY_TOKEN_FILE, AWS_ROLE_ARN.
#
# Writes epp_image_uri=<image> to GITHUB_OUTPUT.

set -euo pipefail

: "${EPP_IMAGE_REPO:?}" "${EPP_SOURCE_SHA:?}" "${GITHUB_REF_NAME:?}" "${GITHUB_OUTPUT:?}"

cd "$(git rev-parse --show-toplevel)"

SELF=.github/scripts/build_epp_image.sh
EPP_DIR=deploy/inference-gateway/ext-proc
EPP_PLATFORMS="${EPP_PLATFORMS:-linux/amd64,linux/arm64}"
NO_CACHE="${NO_CACHE:-false}"
MAKE="${MAKE:-make}"
SHA_REF="${EPP_IMAGE_REPO}:${EPP_SOURCE_SHA}"
ERR_FILE="$(mktemp)"
trap 'rm -f "${ERR_FILE}"' EXIT

case "${GITHUB_REF_NAME}" in
  main | release/*.*.*) PROTECTED_REF=true ;;
  *) PROTECTED_REF=false ;;
esac
REF_TAG="$(printf '%s' "${GITHUB_REF_NAME}" | tr -c 'A-Za-z0-9_.-' '-' | sed -E 's/^[.-]+//' | cut -c1-80)"
REF_TAG="${REF_TAG:-ref}"

# Print the reuse key. Fail when the tree cannot be keyed safely.
reuse_key() {
  local dockerfile="${EPP_DIR}/Dockerfile"
  local -a paths
  local listing status

  # The key reads only one-line `COPY --from=dynamo` instructions. Any other
  # use of the dynamo context could add an input that the key misses.
  if [ "$(grep -c 'from=dynamo' "${dockerfile}")" -ne \
       "$(grep -c -E '^COPY --from=dynamo [^\\]+$' "${dockerfile}")" ]; then
    echo "${dockerfile} uses the dynamo context outside one-line COPY instructions" >&2
    return 1
  fi
  # The parser below cannot read JSON-form arguments, so it would miss them.
  if grep -q -E '^COPY --from=dynamo +\[' "${dockerfile}"; then
    echo "${dockerfile} uses a JSON-form COPY from the dynamo context" >&2
    return 1
  fi
  # EPP_DIR is also the main build context, with the Dockerfile and Makefile.
  mapfile -t paths < <(
    {
      sed -n -E 's/^COPY --from=dynamo (.+) [^ ]+$/\1/p' "${dockerfile}"
      echo "${EPP_DIR} .dockerignore ${SELF}"
    } | tr ' ' '\n' | sed -E 's:/+$::' | grep -v '^$' | sort -u
  )

  # The build context must hold exactly the committed objects.
  if ! status="$(git status --porcelain --ignored -- "${paths[@]}")"; then
    echo "git status failed on the EPP inputs" >&2
    return 1
  fi
  if [ -n "${status}" ]; then
    echo "EPP inputs differ from HEAD" >&2
    return 1
  fi
  if ! listing="$(git ls-tree HEAD -- "${paths[@]}")"; then
    echo "git ls-tree failed on the EPP inputs" >&2
    return 1
  fi
  if [ "$(printf '%s\n' "${listing}" | wc -l)" -ne "${#paths[@]}" ]; then
    echo "Some EPP inputs are not tracked files: ${paths[*]}" >&2
    return 1
  fi
  printf '%s\n' "${listing}" | sha256sum | cut -c1-16
}

has_all_platforms() {
  local ref="$1" raw have platform
  if ! raw="$(docker buildx imagetools inspect --raw "${ref}" 2>"${ERR_FILE}")"; then
    echo "No reusable EPP image at ${ref}: $(head -n 1 "${ERR_FILE}" | cut -c1-200)"
    return 1
  fi
  if ! have="$(printf '%s' "${raw}" | python3 -c '
import json, sys
for m in json.load(sys.stdin).get("manifests", []):
    p = m.get("platform") or {}
    print(p.get("os"), p.get("architecture"), sep="/")
' 2>"${ERR_FILE}")"; then
    echo "Cannot read the manifest of ${ref}: $(tail -n 1 "${ERR_FILE}" | cut -c1-200)"
    return 1
  fi
  for platform in ${EPP_PLATFORMS//,/ }; do
    if ! grep -q -x -F "${platform}" <<<"${have}"; then
      echo "EPP image ${ref} lacks ${platform}"
      return 1
    fi
  done
}

KEY=""
if [ "${NO_CACHE}" != "true" ]; then
  if KEY="$(reuse_key)"; then
    echo "EPP input key: ${KEY}"
  else
    echo "::warning::Cannot key the EPP inputs, so the EPP image is built without reuse"
    KEY=""
  fi
fi

if [ -n "${KEY}" ]; then
  CANDIDATES=("${EPP_IMAGE_REPO}:inputs-${KEY}")
  if [ "${PROTECTED_REF}" != "true" ]; then
    CANDIDATES+=("${EPP_IMAGE_REPO}:${REF_TAG}-inputs-${KEY}")
  fi
  for ref in "${CANDIDATES[@]}"; do
    if has_all_platforms "${ref}"; then
      # Tag the reused image with this commit, as a build would.
      if docker buildx imagetools create --tag "${SHA_REF}" "${ref}"; then
        echo "Reused ${ref} as ${SHA_REF}"
        echo "epp_image_uri=${SHA_REF}" >> "${GITHUB_OUTPUT}"
      else
        echo "::warning::Cannot tag ${ref} as ${SHA_REF}, so the frontend uses ${ref}"
        echo "epp_image_uri=${ref}" >> "${GITHUB_OUTPUT}"
      fi
      exit 0
    fi
  done
fi

EXTRA_ARGS=""
if [[ "${NO_CACHE}" == "true" ]]; then
  EXTRA_ARGS+="--no-cache "
else
  # Content-addressed cache tag: key the registry :cache on a hash of
  # the Rust EPP Dockerfile so Dockerfile changes land in a fresh
  # cache namespace.
  EPP_CACHE_REF="${EPP_IMAGE_REPO}:cache-$(sha256sum "${EPP_DIR}/Dockerfile" | cut -c1-12)"
  EXTRA_ARGS+="--cache-from type=registry,ref=${EPP_CACHE_REF} "
  if [[ "${GITHUB_REF_NAME}" == "main" ]]; then
    EXTRA_ARGS+="--cache-to type=registry,ref=${EPP_CACHE_REF},mode=max "
  fi
fi
if [ -n "${KEY}" ]; then
  if [ "${PROTECTED_REF}" = "true" ]; then
    EXTRA_ARGS+="--tag ${EPP_IMAGE_REPO}:inputs-${KEY} "
  else
    EXTRA_ARGS+="--tag ${EPP_IMAGE_REPO}:${REF_TAG}-inputs-${KEY} "
  fi
fi

# Enable sccache to match the runtime image build path
# (.github/actions/docker-remote-build). The runner pod injects
# AWS_WEB_IDENTITY_TOKEN_FILE + AWS_ROLE_ARN via IRSA; the Makefile
# forwards them to BuildKit as --secret mounts when both are set.
# If IRSA is unavailable, sccache will fail to start inside the
# container and the build proceeds without cache (never with a
# stale one) -- same fallback as the wheel_builder Rust steps.
SCCACHE_MAKE_ARGS=()
if [ -n "${SCCACHE_S3_BUCKET:-}" ] \
    && [ -n "${AWS_WEB_IDENTITY_TOKEN_FILE:-}" ] \
    && [ -f "${AWS_WEB_IDENTITY_TOKEN_FILE}" ] \
    && [ -n "${AWS_ROLE_ARN:-}" ]; then
  SCCACHE_MAKE_ARGS=(USE_SCCACHE=true "SCCACHE_BUCKET=${SCCACHE_S3_BUCKET}" "SCCACHE_REGION=${AWS_DEFAULT_REGION:-}")
else
  echo "::warning::sccache prerequisites missing (bucket or IRSA token); EPP Rust build will run without sccache"
fi

set -x
"${MAKE}" -C "${EPP_DIR}" image-multiarch-push \
  IMAGE_REPO="${EPP_IMAGE_REPO}" \
  GIT_TAG="${EPP_SOURCE_SHA}" \
  DOCKER_PROXY="${DOCKER_PROXY:-}" \
  MULTIARCH_PLATFORMS="${EPP_PLATFORMS}" \
  "${SCCACHE_MAKE_ARGS[@]}" \
  EXTRA_BUILD_ARGS="${EXTRA_ARGS}"
{ set +x; } 2>/dev/null
echo "epp_image_uri=${SHA_REF}" >> "${GITHUB_OUTPUT}"
