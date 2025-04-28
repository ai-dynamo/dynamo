#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

TAG=
RUN_PREFIX=
PLATFORM=linux/amd64

# Get short commit hash
commit_id=$(git rev-parse --short HEAD)

# if COMMIT_ID matches a TAG use that
current_tag=$(git describe --tags --exact-match 2>/dev/null | sed 's/^v//') || true

# Get latest TAG and add COMMIT_ID for dev
latest_tag=$(git describe --tags --abbrev=0 "$(git rev-list --tags --max-count=1 main)" | sed 's/^v//') || true
if [[ -z ${latest_tag} ]]; then
    latest_tag="0.0.1"
    echo "No git release tag found, setting to unknown version: ${latest_tag}"
fi

# Use tag if available, otherwise use latest_tag.dev.commit_id
VERSION=v${current_tag:-$latest_tag.dev.$commit_id}

PYTHON_PACKAGE_VERSION=${current_tag:-$latest_tag.dev+$commit_id}

# Frameworks
#
# Each framework has a corresponding base image.  Additional
# dependencies are specified in the /container/deps folder and
# installed within framework specific sections of the Dockerfile.

declare -A FRAMEWORKS=(["VLLM"]=1 ["TENSORRTLLM"]=2 ["NONE"]=3)
DEFAULT_FRAMEWORK=VLLM

SOURCE_DIR=$(dirname "$(readlink -f "$0")")
DOCKERFILE=${SOURCE_DIR}/Dockerfile
BUILD_CONTEXT=$(dirname "$(readlink -f "$SOURCE_DIR")")

# Base Images
TENSORRTLLM_BASE_IMAGE=tensorrt_llm/release
TENSORRTLLM_BASE_IMAGE_TAG=latest_squashed
TENSORRTLLM_PIP_WHEEL_PATH=""

VLLM_BASE_IMAGE="nvcr.io/nvidia/cuda-dl-base"
VLLM_BASE_IMAGE_TAG="25.03-cuda12.8-devel-ubuntu24.04"

NONE_BASE_IMAGE="ubuntu"
NONE_BASE_IMAGE_TAG="24.04"

NIXL_COMMIT=3810b36b1ad0b647488e163f0f4e8dfc22ffabdf
NIXL_REPO=ai-dynamo/nixl.git

get_options() {
    while :; do
        case $1 in
        -h | -\? | --help)
            show_help
            exit
            ;;
        --platform)
            if [ "$2" ]; then
                PLATFORM=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --framework)
            if [ "$2" ]; then
                FRAMEWORK=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --tensorrtllm-pip-wheel-path)
            if [ "$2" ]; then
                TENSORRTLLM_PIP_WHEEL_PATH=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --base-image)
            if [ "$2" ]; then
                BASE_IMAGE=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --base-image-tag)
            if [ "$2" ]; then
                BASE_IMAGE_TAG=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --target)
            if [ "$2" ]; then
                TARGET=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --build-arg)
            if [ "$2" ]; then
                BUILD_ARGS+="--build-arg $2 "
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --tag)
            if [ "$2" ]; then
                TAG="--tag $2"
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --dry-run)
            RUN_PREFIX="echo"
            echo ""
            echo "=============================="
            echo "DRY RUN: COMMANDS PRINTED ONLY"
            echo "=============================="
            echo ""
            ;;
        --no-cache)
            NO_CACHE=" --no-cache"
            ;;
        --cache-from)
            if [ "$2" ]; then
                CACHE_FROM="--cache-from $2"
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --cache-to)
            if [ "$2" ]; then
                CACHE_TO="--cache-to $2"
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --build-context)
            if [ "$2" ]; then
                BUILD_CONTEXT_ARG="--build-context $2"
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --release-build)
            RELEASE_BUILD=true
            ;;
        --)
            shift
            break
            ;;
         -?*)
            error 'ERROR: Unknown option: ' "$1"
            ;;
         ?*)
            error 'ERROR: Unknown option: ' "$1"
            ;;
        *)
            break
            ;;
        esac
        shift
    done

    if [ -z "$FRAMEWORK" ]; then
        FRAMEWORK=$DEFAULT_FRAMEWORK
    fi

    if [ -n "$FRAMEWORK" ]; then
        FRAMEWORK=${FRAMEWORK^^}

        if [[ -z "${FRAMEWORKS[$FRAMEWORK]}" ]]; then
            error 'ERROR: Unknown framework: ' "$FRAMEWORK"
        fi

        if [ -z "$BASE_IMAGE_TAG" ]; then
            BASE_IMAGE_TAG=${FRAMEWORK}_BASE_IMAGE_TAG
            BASE_IMAGE_TAG=${!BASE_IMAGE_TAG}
        fi

        if [ -z "$BASE_IMAGE" ]; then
            BASE_IMAGE=${FRAMEWORK}_BASE_IMAGE
            BASE_IMAGE=${!BASE_IMAGE}
        fi

        if [ -z "$BASE_IMAGE" ]; then
            error "ERROR: Framework $FRAMEWORK without BASE_IMAGE"
        fi

        BASE_VERSION=${FRAMEWORK}_BASE_VERSION
        BASE_VERSION=${!BASE_VERSION}

    fi

    if [ -z "$TAG" ]; then
        TAG="--tag dynamo:${VERSION}-${FRAMEWORK,,}"
        if [ -n "${TARGET}" ]; then
            TAG="${TAG}-${TARGET}"
        fi
    fi

    if [ -n "$PLATFORM" ]; then
        PLATFORM="--platform ${PLATFORM}"
    fi

    if [ -n "$TARGET" ]; then
        TARGET_STR="--target ${TARGET}"
    else
        TARGET_STR="--target dev"
    fi
}


show_image_options() {
    echo ""
    echo "Building Dynamo Image: '${TAG}'"
    echo ""
    echo "   Base: '${BASE_IMAGE}'"
    echo "   Base_Image_Tag: '${BASE_IMAGE_TAG}'"
    if [[ $FRAMEWORK == "TENSORRTLLM" ]]; then
        echo "   Tensorrtllm_Pip_Wheel_Path: '${TENSORRTLLM_PIP_WHEEL_PATH}'"
    fi
    echo "   Build Context: '${BUILD_CONTEXT}'"
    echo "   Build Arguments: '${BUILD_ARGS}'"
    echo "   Framework: '${FRAMEWORK}'"
    echo ""
}

show_help() {
    echo "usage: build.sh"
    echo "  [--base base image]"
    echo "  [--base-image-tag base image tag]"
    echo "  [--platform platform for docker build"
    echo "  [--framework framework one of ${!FRAMEWORKS[*]}]"
    echo "  [--tensorrtllm-pip-wheel-path path to tensorrtllm pip wheel]"
    echo "  [--build-arg additional build args to pass to docker build]"
    echo "  [--cache-from cache location to start from]"
    echo "  [--cache-to location where to cache the build output]"
    echo "  [--tag tag for image]"
    echo "  [--no-cache disable docker build cache]"
    echo "  [--dry-run print docker commands without running]"
    echo "  [--build-context name=path to add build context]"
    exit 0
}

missing_requirement() {
    error "ERROR: $1 requires an argument."
}

error() {
    printf '%s %s\n' "$1" "$2" >&2
    exit 1
}

get_options "$@"

# Automatically set ARCH and ARCH_ALT if PLATFORM is linux/arm64
if [[ "$PLATFORM" == *"linux/arm64"* ]]; then
    BUILD_ARGS+=" --build-arg ARCH=arm64 --build-arg ARCH_ALT=aarch64 "
fi

# Update DOCKERFILE if framework is VLLM
if [[ $FRAMEWORK == "VLLM" ]]; then
    DOCKERFILE=${SOURCE_DIR}/Dockerfile.vllm
elif [[ $FRAMEWORK == "TENSORRTLLM" ]]; then
    DOCKERFILE=${SOURCE_DIR}/Dockerfile.tensorrt_llm
elif [[ $FRAMEWORK == "NONE" ]]; then
    DOCKERFILE=${SOURCE_DIR}/Dockerfile.none
fi

if [[ $FRAMEWORK == "VLLM" ]]; then
    NIXL_DIR="/tmp/nixl/nixl_src"

    # Clone original NIXL to temp directory
    if [ -d "$NIXL_DIR" ]; then
        echo "Warning: $NIXL_DIR already exists, skipping clone"
    else
        if [ -n "${GITHUB_TOKEN}" ]; then
            git clone "https://oauth2:${GITHUB_TOKEN}@github.com/${NIXL_REPO}" "$NIXL_DIR"
        else
            # Try HTTPS first with credential prompting disabled, fall back to SSH if it fails
            if ! GIT_TERMINAL_PROMPT=0 git clone https://github.com/${NIXL_REPO} "$NIXL_DIR"; then
                echo "HTTPS clone failed, falling back to SSH..."
                git clone git@github.com:${NIXL_REPO} "$NIXL_DIR"
            fi
        fi
    fi

    cd "$NIXL_DIR" || exit
    if ! git checkout ${NIXL_COMMIT}; then
        echo "ERROR: Failed to checkout NIXL commit ${NIXL_COMMIT}. The cached directory may be out of date."
        echo "Please delete $NIXL_DIR and re-run the build script."
        exit 1
    fi

    BUILD_CONTEXT_ARG+=" --build-context nixl=$NIXL_DIR"

    # Add NIXL_COMMIT as a build argument to enable caching
    BUILD_ARGS+=" --build-arg NIXL_COMMIT=${NIXL_COMMIT} "
fi

if [[ $TARGET == "local-dev" ]]; then
    BUILD_ARGS+=" --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) "
fi

# BUILD DEV IMAGE

BUILD_ARGS+=" --build-arg BASE_IMAGE=$BASE_IMAGE --build-arg BASE_IMAGE_TAG=$BASE_IMAGE_TAG --build-arg FRAMEWORK=$FRAMEWORK --build-arg ${FRAMEWORK}_FRAMEWORK=1 --build-arg VERSION=$VERSION --build-arg PYTHON_PACKAGE_VERSION=$PYTHON_PACKAGE_VERSION"

if [ -n "${GITHUB_TOKEN}" ]; then
    BUILD_ARGS+=" --build-arg GITHUB_TOKEN=${GITHUB_TOKEN} "
fi

if [ -n "${GITLAB_TOKEN}" ]; then
    BUILD_ARGS+=" --build-arg GITLAB_TOKEN=${GITLAB_TOKEN} "
fi

if [[ $FRAMEWORK == "TENSORRTLLM" ]]; then
    if [ -n "${TENSORRTLLM_PIP_WHEEL_PATH}" ]; then
        BUILD_ARGS+=" --build-arg TENSORRTLLM_PIP_WHEEL_PATH=${TENSORRTLLM_PIP_WHEEL_PATH} "
    fi
fi

if [ -n "${HF_TOKEN}" ]; then
    BUILD_ARGS+=" --build-arg HF_TOKEN=${HF_TOKEN} "
fi
if [  ! -z ${RELEASE_BUILD} ]; then
    echo "Performing a release build!"
    BUILD_ARGS+=" --build-arg RELEASE_BUILD=${RELEASE_BUILD} "
fi

LATEST_TAG="--tag dynamo:latest-${FRAMEWORK,,}"
if [ -n "${TARGET}" ]; then
    LATEST_TAG="${LATEST_TAG}-${TARGET}"
fi

show_image_options

if [ -z "$RUN_PREFIX" ]; then
    set -x
fi

# Check if the TensorRT-LLM base image exists
if [[ $FRAMEWORK == "TENSORRTLLM" ]]; then
    if docker inspect --type=image "$BASE_IMAGE:$BASE_IMAGE_TAG" > /dev/null 2>&1; then
        echo "Image '$BASE_IMAGE:$BASE_IMAGE_TAG' is found."
    else
        echo "Image '$BASE_IMAGE:$BASE_IMAGE_TAG' is not found." >&2
        echo "Please build the TensorRT-LLM base image first. Run ./build_trtllm_base_image.sh" >&2
        echo "or use --base-image and --base-image-tag to an existing TensorRT-LLM base image." >&2
        echo "See https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html for more information." >&2
        exit 1
    fi
fi

$RUN_PREFIX docker build -f $DOCKERFILE $TARGET_STR $PLATFORM $BUILD_ARGS $CACHE_FROM $CACHE_TO $TAG $LATEST_TAG $BUILD_CONTEXT_ARG $BUILD_CONTEXT $NO_CACHE

{ set +x; } 2>/dev/null

if [ -z "$RUN_PREFIX" ]; then
    set -x
fi

{ set +x; } 2>/dev/null
