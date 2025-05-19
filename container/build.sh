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

if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
    echo "Error: Bash version 4.0 or higher is required. Current version: ${BASH_VERSINFO[0]}.${BASH_VERSINFO[1]}"
    exit 1
fi

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
TENSORRTLLM_BASE_IMAGE=nvcr.io/nvidia/pytorch
TENSORRTLLM_BASE_IMAGE_TAG=25.04-py3

# Important Note: Because of ABI compatibility issues between TensorRT-LLM and NGC PyTorch,
# we need to build the TensorRT-LLM wheel from source.
#
# There are two ways to build the dynamo image with TensorRT-LLM.
# 1. Use the local TensorRT-LLM wheel directory.
# 2. Use the TensorRT-LLM wheel on artifactory.
#
# If using option 1, the TENSORRTLLM_PIP_WHEEL_DIR must be a path to a directory
# containing TensorRT-LLM wheel file along with commit.txt file with the
# <arch>_<commit ID> as contents. If no valid trtllm wheel is found, the script
# will attempt to build the wheel from source and store the built wheel in the
# specified directory. TRTLLM_COMMIT from the TensorRT-LLM main branch will be
# used to build the wheel.
#
# If using option 2, the TENSORRTLLM_PIP_WHEEL must be the TensorRT-LLM wheel
# package that will be installed from the specified TensorRT-LLM PyPI Index URL.
# This option will ignore the TRTLLM_COMMIT option. As the TensorRT-LLM wheel from PyPI
# is not ABI compatible with NGC PyTorch, you can use TENSORRTLLM_INDEX_URL to specify
# a private PyPI index URL which has your pre-built TensorRT-LLM wheel.
#
# By default, we will use option 1. If you want to use option 2, you can set
# TENSORRTLLM_PIP_WHEEL to the TensorRT-LLM wheel on artifactory.
#
# Path to the local TensorRT-LLM wheel directory or the wheel on artifactory.
TENSORRTLLM_PIP_WHEEL_DIR="/tmp/trtllm_wheel/"
# TensorRT-LLM commit to use for building the trtllm wheel if not provided.
# Important Note: This commit is not used in our CI pipeline. See the CI
# variables to learn how to run a pipeline with a specific commit.
TRTLLM_COMMIT=290649b6aaed5f233b0a0adf50edc1347f8d2b14

# TensorRT-LLM PyPI index URL
TENSORRTLLM_INDEX_URL="https://pypi.python.org/simple"
TENSORRTLLM_PIP_WHEEL=""



VLLM_BASE_IMAGE="nvcr.io/nvidia/cuda-dl-base"
# FIXME: NCCL will hang with 25.03, so use 25.01 for now
# Please check https://github.com/ai-dynamo/dynamo/pull/1065
# for details and reproducer to manually test if the image
# can be updated to later versions.
VLLM_BASE_IMAGE_TAG="25.01-cuda12.8-devel-ubuntu24.04"

NONE_BASE_IMAGE="ubuntu"
NONE_BASE_IMAGE_TAG="24.04"

NIXL_COMMIT=78695c2900cd7fff506764377386592dfc98e87e
NIXL_REPO=ai-dynamo/nixl.git

NO_CACHE=""

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
        --tensorrtllm-pip-wheel-dir)
            if [ "$2" ]; then
                TENSORRTLLM_PIP_WHEEL_DIR=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --tensorrtllm-commit)
            if [ "$2" ]; then
                TRTLLM_COMMIT=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --tensorrtllm-pip-wheel)
            if [ "$2" ]; then
                TENSORRTLLM_PIP_WHEEL=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --tensorrtllm-index-url)
            if [ "$2" ]; then
                TENSORRTLLM_INDEX_URL=$2
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
        echo "   Tensorrtllm_Pip_Wheel: '${TENSORRTLLM_PIP_WHEEL}'"
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
    echo "  [--tensorrtllm-pip-wheel-dir path to tensorrtllm pip wheel directory]"
    echo "  [--tensorrtllm-commit tensorrtllm commit to use for building the trtllm wheel if the wheel is not provided]"
    echo "  [--tensorrtllm-pip-wheel tensorrtllm pip wheel on artifactory]"
    echo "  [--tensorrtllm-index-url tensorrtllm PyPI index URL if providing the wheel from artifactory]"
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
ARCH="amd64"
if [[ "$PLATFORM" == *"linux/arm64"* ]]; then
    ARCH="arm64"
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


check_wheel_file() {
    local wheel_dir="$1"
    # Check if directory exists
    if [ ! -d "$wheel_dir" ]; then
        echo "Error: Directory '$wheel_dir' does not exist"
        return 1
    fi

    # Look for .whl files
    wheel_count=$(find "$wheel_dir" -name "*.whl" | wc -l)

    if [ "$wheel_count" -eq 0 ]; then
        echo "WARN: No .whl files found in '$wheel_dir'"
        return 1
    elif [ "$wheel_count" -gt 1 ]; then
        echo "Warning: Multiple wheel files found in '$wheel_dir'. Will use first one found."
        find "$wheel_dir" -name "*.whl" | head -n 1
        return 0
    else
        echo "Found $wheel_count wheel files in '$wheel_dir'"
        # Check if commit file exists
        commit_file="$wheel_dir/commit.txt"
        if [ ! -f "$commit_file" ]; then
            echo "Error: Commit file '$commit_file' does not exist"
            return 1
        fi

        # Check if commit ID matches, otherwise re-build the wheel
        # Commit ID is of the form <arch>_<commit_id>
        commit_id=$(cat "$commit_file")
        if [ "$commit_id" != "$2" ]; then
            echo "Error: Commit ID mismatch. Expected '$2', got '$commit_id'"
            rm -rf $wheel_dir/*.whl
            return 1
        fi
        return 0
    fi
}

if [[ $FRAMEWORK == "TENSORRTLLM" ]]; then
    if [ -z "${TENSORRTLLM_PIP_WHEEL}" ]; then
        # Use option 1
        if [ ! -d "${TENSORRTLLM_PIP_WHEEL_DIR}" ]; then
            # Create the directory if it doesn't exist
            mkdir -p ${TENSORRTLLM_PIP_WHEEL_DIR}
        fi
        BUILD_ARGS+=" --build-arg HAS_TRTLLM_CONTEXT=1"
        echo "Checking for TensorRT-LLM wheel in ${TENSORRTLLM_PIP_WHEEL_DIR}"
        if ! check_wheel_file "${TENSORRTLLM_PIP_WHEEL_DIR}" "${ARCH}_${TRTLLM_COMMIT}"; then
            echo "WARN: Valid trtllm wheel file not found in ${TENSORRTLLM_PIP_WHEEL_DIR}, attempting to build from source"
            if ! env -i ${SOURCE_DIR}/build_trtllm_wheel.sh -o ${TENSORRTLLM_PIP_WHEEL_DIR} -c ${TRTLLM_COMMIT} -a ${ARCH}; then
                error "ERROR: Failed to build TensorRT-LLM wheel"
            fi
        fi
        echo "Installing TensorRT-LLM from local wheel directory"
        BUILD_CONTEXT_ARG+=" --build-context trtllm_wheel=${TENSORRTLLM_PIP_WHEEL_DIR}"

    else
        BUILD_ARGS+=" --build-arg HAS_TRTLLM_CONTEXT=0"
        BUILD_ARGS+=" --build-arg TENSORRTLLM_PIP_WHEEL=${TENSORRTLLM_PIP_WHEEL}"
        BUILD_ARGS+=" --build-arg TENSORRTLLM_INDEX_URL=${TENSORRTLLM_INDEX_URL}"

        # Create a dummy directory to satisfy the build context requirement
        # There is no way to conditionally copy the build context in dockerfile.
        mkdir -p /tmp/dummy_dir
        BUILD_CONTEXT_ARG+=" --build-context trtllm_wheel=/tmp/dummy_dir"
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

$RUN_PREFIX docker build -f $DOCKERFILE $TARGET_STR $PLATFORM $BUILD_ARGS $CACHE_FROM $CACHE_TO $TAG $LATEST_TAG $BUILD_CONTEXT_ARG $BUILD_CONTEXT $NO_CACHE

{ set +x; } 2>/dev/null

if [ -z "$RUN_PREFIX" ]; then
    set -x
fi

{ set +x; } 2>/dev/null
