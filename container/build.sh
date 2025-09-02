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

declare -A FRAMEWORKS=(["VLLM"]=1 ["TRTLLM"]=2 ["NONE"]=3 ["SGLANG"]=4)

DEFAULT_FRAMEWORK=VLLM

SOURCE_DIR=$(dirname "$(readlink -f "$0")")
DOCKERFILE=${SOURCE_DIR}/Dockerfile
BUILD_CONTEXT=$(dirname "$(readlink -f "$SOURCE_DIR")")

# Base Images
TRTLLM_BASE_IMAGE=nvcr.io/nvidia/pytorch
TRTLLM_BASE_IMAGE_TAG=25.06-py3

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
DEFAULT_EXPERIMENTAL_TRTLLM_COMMIT="a16ba6445c61ed70e7aadfe787d6f316bb422652"
TRTLLM_COMMIT=""
TRTLLM_USE_NIXL_KVCACHE_EXPERIMENTAL="0"
TRTLLM_GIT_URL=""

# TensorRT-LLM PyPI index URL
TENSORRTLLM_INDEX_URL="https://pypi.python.org/simple"
# TODO: Remove the version specification from here and use the ai-dynamo[trtllm] package.
# Need to update the Dockerfile.trtllm to use the ai-dynamo[trtllm] package.
DEFAULT_TENSORRTLLM_PIP_WHEEL="tensorrt-llm==1.0.0rc6"
TENSORRTLLM_PIP_WHEEL=""


VLLM_BASE_IMAGE="nvcr.io/nvidia/cuda-dl-base"
# FIXME: NCCL will hang with 25.03, so use 25.01 for now
# Please check https://github.com/ai-dynamo/dynamo/pull/1065
# for details and reproducer to manually test if the image
# can be updated to later versions.
VLLM_BASE_IMAGE_TAG="25.01-cuda12.8-devel-ubuntu24.04"

NONE_BASE_IMAGE="nvcr.io/nvidia/cuda-dl-base"
NONE_BASE_IMAGE_TAG="25.01-cuda12.8-devel-ubuntu24.04"

SGLANG_BASE_IMAGE="nvcr.io/nvidia/cuda-dl-base"
SGLANG_BASE_IMAGE_TAG="25.01-cuda12.8-devel-ubuntu24.04"

NIXL_REF=0.4.1
NIXL_UCX_REF=v1.19.0
NIXL_UCX_EFA_REF=9d2b88a1f67faf9876f267658bd077b379b8bb76

NO_CACHE=""

# sccache configuration for S3
USE_SCCACHE=""
SCCACHE_BUCKET=""
SCCACHE_REGION=""

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
        --use-default-experimental-tensorrtllm-commit)
            if [ -n "$2" ] && [[ "$2" != --* ]]; then
                echo "ERROR: --use-default-experimental-tensorrtllm-commit does not take any argument"
                exit 1
            fi
            USE_DEFAULT_EXPERIMENTAL_TRTLLM_COMMIT=true
            ;;
        --trtllm-use-nixl-kvcache-experimental)
            if [ -n "$2" ] && [[ "$2" != --* ]]; then
                echo "ERROR: --trtllm-use-nixl-kvcache-experimental does not take any argument"
                exit 1
            fi
            TRTLLM_USE_NIXL_KVCACHE_EXPERIMENTAL="1"
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
        --tensorrtllm-git-url)
            if [ "$2" ]; then
                TRTLLM_GIT_URL=$2
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
        --enable-kvbm)
            ENABLE_KVBM=true
            ;;
        --make-efa)
            NIXL_UCX_REF=$NIXL_UCX_EFA_REF
            ;;
        --use-sccache)
            USE_SCCACHE=true
            ;;
        --sccache-bucket)
            if [ "$2" ]; then
                SCCACHE_BUCKET=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;

        --sccache-region)
            if [ "$2" ]; then
                SCCACHE_REGION=$2
                shift
            else
                missing_requirement "$1"
            fi
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

    # Validate sccache configuration
    if [ "$USE_SCCACHE" = true ]; then
        if [ -z "$SCCACHE_BUCKET" ]; then
            error "ERROR: --sccache-bucket is required when --use-sccache is specified"
        fi
        if [ -z "$SCCACHE_REGION" ]; then
            error "ERROR: --sccache-region is required when --use-sccache is specified"
        fi
    fi
}


show_image_options() {
    echo ""
    echo "Building Dynamo Image: '${TAG}'"
    echo ""
    echo "   Base: '${BASE_IMAGE}'"
    echo "   Base_Image_Tag: '${BASE_IMAGE_TAG}'"
    if [[ $FRAMEWORK == "TRTLLM" ]]; then
        echo "   Tensorrtllm_Pip_Wheel: '${TENSORRTLLM_PIP_WHEEL}'"
    fi
    echo "   Build Context: '${BUILD_CONTEXT}'"
    echo "   Build Arguments: '${BUILD_ARGS}'"
    echo "   Framework: '${FRAMEWORK}'"
    if [ "$USE_SCCACHE" = true ]; then
        echo "   sccache: Enabled"
        echo "   sccache Bucket: '${SCCACHE_BUCKET}'"
        echo "   sccache Region: '${SCCACHE_REGION}'"

        if [ -n "$SCCACHE_S3_KEY_PREFIX" ]; then
            echo "   sccache S3 Key Prefix: '${SCCACHE_S3_KEY_PREFIX}'"
        fi
    fi
    echo ""
}

show_help() {
    echo "usage: build.sh"
    echo "  [--base-image base image]"
    echo "  [--base-image-tag base image tag]"
    echo "  [--platform platform for docker build]"
    echo "  [--framework framework one of ${!FRAMEWORKS[*]}]"
    echo "  [--tensorrtllm-pip-wheel-dir path to tensorrtllm pip wheel directory]"
    echo "  [--tensorrtllm-commit tensorrtllm commit to use for building the trtllm wheel if the wheel is not provided]"
    echo "  [--use-default-experimental-tensorrtllm-commit] Use the default experimental commit (${DEFAULT_EXPERIMENTAL_TRTLLM_COMMIT}) to build TensorRT-LLM. This is a flag (no argument). Do not combine with --tensorrtllm-commit or --tensorrtllm-pip-wheel."
    echo "  [--tensorrtllm-pip-wheel tensorrtllm pip wheel on artifactory]"
    echo "  [--tensorrtllm-index-url tensorrtllm PyPI index URL if providing the wheel from artifactory]"
    echo "  [--tensorrtllm-git-url tensorrtllm git repository URL for cloning]"
    echo "  [--build-arg additional build args to pass to docker build]"
    echo "  [--cache-from cache location to start from]"
    echo "  [--cache-to location where to cache the build output]"
    echo "  [--tag tag for image]"
    echo "  [--no-cache disable docker build cache]"
    echo "  [--dry-run print docker commands without running]"
    echo "  [--build-context name=path to add build context]"
    echo "  [--release-build perform a release build]"
    echo "  [--make-efa Enables EFA support for NIXL]"
    echo "  [--enable-kvbm Enables KVBM support in Python 3.12]"
    echo "  [--trtllm-use-nixl-kvcache-experimental Enables NIXL KVCACHE experimental support for TensorRT-LLM]"
    echo "  [--use-sccache enable sccache for Rust/C/C++ compilation caching]"
    echo "  [--sccache-bucket S3 bucket name for sccache (required with --use-sccache)]"
    echo "  [--sccache-region S3 region for sccache (required with --use-sccache)]"
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
elif [[ $FRAMEWORK == "TRTLLM" ]]; then
    DOCKERFILE=${SOURCE_DIR}/Dockerfile.trtllm
elif [[ $FRAMEWORK == "NONE" ]]; then
    DOCKERFILE=${SOURCE_DIR}/Dockerfile
elif [[ $FRAMEWORK == "SGLANG" ]]; then
    DOCKERFILE=${SOURCE_DIR}/Dockerfile.sglang
fi

# Add NIXL_REF as a build argument
BUILD_ARGS+=" --build-arg NIXL_REF=${NIXL_REF} "

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

if [[ $FRAMEWORK == "TRTLLM" ]]; then
    if [ "$USE_DEFAULT_EXPERIMENTAL_TRTLLM_COMMIT" = true ]; then
        if [ -n "$TRTLLM_COMMIT" ] || [ -n "$TENSORRTLLM_PIP_WHEEL" ]; then
            echo "ERROR: When using --use-default-experimental-trtllm-commit, do not set --tensorrtllm-commit or --tensorrtllm-pip-wheel."
            exit 1
        fi
        TRTLLM_COMMIT="$DEFAULT_EXPERIMENTAL_TRTLLM_COMMIT"
    fi

    if [ -n "${TRTLLM_USE_NIXL_KVCACHE_EXPERIMENTAL}" ]; then
        BUILD_ARGS+=" --build-arg TRTLLM_USE_NIXL_KVCACHE_EXPERIMENTAL=${TRTLLM_USE_NIXL_KVCACHE_EXPERIMENTAL} "
    fi

    # If user didn't set both wheel and commit, use default tensorrt_llm pip wheel
    if [ -z "$TENSORRTLLM_PIP_WHEEL" ] && [ -z "$TRTLLM_COMMIT" ]; then
        TENSORRTLLM_PIP_WHEEL="$DEFAULT_TENSORRTLLM_PIP_WHEEL"
    fi

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
            GIT_URL_ARG=""
            if [ -n "${TRTLLM_GIT_URL}" ]; then
                GIT_URL_ARG="-u ${TRTLLM_GIT_URL}"
            fi
            if ! env -i ${SOURCE_DIR}/build_trtllm_wheel.sh -o ${TENSORRTLLM_PIP_WHEEL_DIR} -c ${TRTLLM_COMMIT} -a ${ARCH} -n ${NIXL_REF} ${GIT_URL_ARG}; then
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

if [[ $FRAMEWORK == "VLLM" ]]; then
    echo "Forcing enable_kvbm to true in vLLM image build"
    ENABLE_KVBM=true
fi

if [  ! -z ${ENABLE_KVBM} ]; then
    echo "Enabling the KVBM in the ai-dynamo-runtime"
    BUILD_ARGS+=" --build-arg ENABLE_KVBM=${ENABLE_KVBM} "
fi

if [ -n "${NIXL_UCX_REF}" ]; then
    BUILD_ARGS+=" --build-arg NIXL_UCX_REF=${NIXL_UCX_REF} "
fi

# Add sccache build arguments
if [ "$USE_SCCACHE" = true ]; then
    BUILD_ARGS+=" --build-arg USE_SCCACHE=true"
    BUILD_ARGS+=" --build-arg SCCACHE_BUCKET=${SCCACHE_BUCKET}"
    BUILD_ARGS+=" --build-arg SCCACHE_REGION=${SCCACHE_REGION}"


fi

LATEST_TAG="--tag dynamo:latest-${FRAMEWORK,,}"
if [ -n "${TARGET}" ]; then
    LATEST_TAG="${LATEST_TAG}-${TARGET}"
fi

show_image_options

if [ -z "$RUN_PREFIX" ]; then
    set -x
fi

# TODO: Follow 2-step build process for all frameworks once necessary changes are made to the sglang and TRT-LLM backend Dockerfiles.
if [[ $FRAMEWORK == "VLLM" ]]; then
    # Define base image tag before using it
    DYNAMO_BASE_IMAGE="dynamo-base:${VERSION}"
    # Start base image build
    echo "======================================"
    echo "Starting Build 1: Base Image"
    echo "======================================"
    $RUN_PREFIX docker build -f "${SOURCE_DIR}/Dockerfile" --target dev $PLATFORM $BUILD_ARGS $CACHE_FROM $CACHE_TO --tag $DYNAMO_BASE_IMAGE $BUILD_CONTEXT_ARG $BUILD_CONTEXT $NO_CACHE
    # Start framework build
    echo "======================================"
    echo "Starting Build 2: Framework Image"
    echo "======================================"
    BUILD_ARGS+=" --build-arg DYNAMO_BASE_IMAGE=${DYNAMO_BASE_IMAGE}"
    $RUN_PREFIX docker build -f $DOCKERFILE $TARGET_STR $PLATFORM $BUILD_ARGS $CACHE_FROM $CACHE_TO $TAG $LATEST_TAG $BUILD_CONTEXT_ARG $BUILD_CONTEXT $NO_CACHE
else
    $RUN_PREFIX docker build -f $DOCKERFILE $TARGET_STR $PLATFORM $BUILD_ARGS $CACHE_FROM $CACHE_TO $TAG $LATEST_TAG $BUILD_CONTEXT_ARG $BUILD_CONTEXT $NO_CACHE
fi


{ set +x; } 2>/dev/null
