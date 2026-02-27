#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds and optionally pushes all Docker images needed for DGDR E2E testing:
#   - dynamo-operator    (Go operator)
#   - dynamo-frontend    (profiler + EPP frontend)
#   - vllm-runtime       (vLLM backend)
#   - sglang-runtime     (SGLang backend)
#   - trtllm-runtime     (TensorRT-LLM backend)
#
# Usage:
#   container/build_images.sh --repo <dockerhub-user> --tag <tag> [--images operator,frontend,vllm,sglang,trtllm] [--push]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REPO=""
TAG=""
IMAGES="operator,frontend,vllm,sglang,trtllm"
PUSH=false

usage() {
    cat <<EOF
Usage: $(basename "$0") --repo <docker-repo> --tag <tag> [OPTIONS]

Required:
  --repo <repo>       Docker registry username/org (e.g. hongkuanz196)
  --tag <tag>         Image tag (e.g. hzhou-0224)

Options:
  --images <list>     Comma-separated images to build (default: all)
                      Choices: operator, frontend, vllm, sglang, trtllm
  --push              Push images after building
  -h, --help          Show this help

Examples:
  # Build and push all images
  $(basename "$0") --repo hongkuanz196 --tag hzhou-0224 --push

  # Build only frontend and vllm locally
  $(basename "$0") --repo hongkuanz196 --tag dev --images frontend,vllm
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)
            [[ -z "${2:-}" || "$2" == --* ]] && { echo "Error: --repo requires a value"; usage 1; }
            REPO="$2"; shift 2 ;;
        --tag)
            [[ -z "${2:-}" || "$2" == --* ]] && { echo "Error: --tag requires a value"; usage 1; }
            TAG="$2"; shift 2 ;;
        --images)
            [[ -z "${2:-}" || "$2" == --* ]] && { echo "Error: --images requires a value"; usage 1; }
            IMAGES="$2"; shift 2 ;;
        --push)   PUSH=true; shift ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown option: $1"; usage 1 ;;
    esac
done

if [[ -z "$REPO" || -z "$TAG" ]]; then
    echo "Error: --repo and --tag are required"
    usage 1
fi

IFS=',' read -ra IMAGE_LIST <<< "$IMAGES"

BUILT_IMAGES=()

should_build() {
    local target="$1"
    for img in "${IMAGE_LIST[@]}"; do
        if [[ "$img" == "$target" ]]; then
            return 0
        fi
    done
    return 1
}

log_header() {
    echo ""
    echo "================================================================"
    echo "  Building: $1"
    echo "================================================================"
    echo ""
}

push_image() {
    local image="$1"
    if $PUSH; then
        echo "Pushing $image ..."
        docker push "$image"
    fi
}

# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------
build_operator() {
    local image="${REPO}/dynamo-operator:${TAG}"
    log_header "$image"
    docker build \
        -t "$image" \
        -f "${DYNAMO_ROOT}/deploy/operator/Dockerfile" \
        "${DYNAMO_ROOT}/deploy/operator/"
    push_image "$image"
    BUILT_IMAGES+=("$image")
}

# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
build_frontend() {
    local image="${REPO}/dynamo-frontend:${TAG}"
    local epp_image="dynamo-epp:${TAG}-build"

    log_header "EPP (intermediate): $epp_image"
    cd "${DYNAMO_ROOT}/deploy/inference-gateway/epp"
    make image-load IMAGE_TAG="$epp_image" DYNAMO_DIR="$DYNAMO_ROOT"

    log_header "$image"
    cd "$DYNAMO_ROOT"
    python container/render.py --framework=dynamo --target=frontend --output-short-filename
    docker build \
        --build-arg "EPP_IMAGE=${epp_image}" \
        -t "$image" \
        -f container/rendered.Dockerfile .
    rm -f container/rendered.Dockerfile
    push_image "$image"
    BUILT_IMAGES+=("$image")
}

# ---------------------------------------------------------------------------
# Backend runtimes (vllm, sglang, trtllm)
# ---------------------------------------------------------------------------
build_runtime() {
    local framework="$1"
    local image_name="$2"
    local extra_args="${3:-}"
    local image="${REPO}/${image_name}:${TAG}"

    log_header "$image"
    cd "$DYNAMO_ROOT"
    # shellcheck disable=SC2086
    python container/render.py --framework="$framework" --target=runtime --output-short-filename $extra_args
    docker build -t "$image" -f container/rendered.Dockerfile .
    rm -f container/rendered.Dockerfile
    push_image "$image"
    BUILT_IMAGES+=("$image")
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "Dynamo Root: $DYNAMO_ROOT"
echo "Repo:        $REPO"
echo "Tag:         $TAG"
echo "Images:      $IMAGES"
echo "Push:        $PUSH"

should_build "operator" && build_operator
should_build "frontend" && build_frontend
should_build "vllm"     && build_runtime vllm vllm-runtime
should_build "sglang"   && build_runtime sglang sglang-runtime
should_build "trtllm"   && build_runtime trtllm trtllm-runtime "--cuda-version=13.1"

echo ""
echo "================================================================"
echo "  Build Summary"
echo "================================================================"
if [[ ${#BUILT_IMAGES[@]} -eq 0 ]]; then
    echo "  No images built."
else
    for img in "${BUILT_IMAGES[@]}"; do
        echo "  $img"
    done
fi
if $PUSH; then
    echo ""
    echo "  All images pushed to registry."
fi
echo "================================================================"
