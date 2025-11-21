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
commit_id=${commit_id:-$(git rev-parse --short HEAD)}

# if COMMIT_ID matches a TAG use that
current_tag=${current_tag:-$($(git describe --tags --exact-match 2>/dev/null | sed 's/^v//') || true)}

# Get latest TAG and add COMMIT_ID for dev
latest_tag=${latest_tag:-$(git describe --tags --abbrev=0 "$(git rev-list --tags --max-count=1)" | sed 's/^v//' || true)}
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
TRTLLM_BASE_IMAGE_TAG=25.10-py3

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
DEFAULT_TENSORRTLLM_PIP_WHEEL_DIR="/tmp/trtllm_wheel/"

# TensorRT-LLM commit to use for building the trtllm wheel if not provided.
# Important Note: This commit is not used in our CI pipeline. See the CI
# variables to learn how to run a pipeline with a specific commit.
DEFAULT_EXPERIMENTAL_TRTLLM_COMMIT="31116825b39f4e6a6a1e127001f5204b73d1dc32" # 1.2.0rc2
TRTLLM_COMMIT=""
TRTLLM_USE_NIXL_KVCACHE_EXPERIMENTAL="0"
TRTLLM_GIT_URL=""

# TensorRT-LLM PyPI index URL
DEFAULT_TENSORRTLLM_INDEX_URL="https://pypi.nvidia.com/"
# TODO: Remove the version specification from here and use the ai-dynamo[trtllm] package.
# Need to update the Dockerfile.trtllm to use the ai-dynamo[trtllm] package.
DEFAULT_TENSORRTLLM_PIP_WHEEL="tensorrt-llm==1.2.0rc2"
TENSORRTLLM_PIP_WHEEL=""

VLLM_BASE_IMAGE="nvcr.io/nvidia/cuda-dl-base"
# FIXME: OPS-612 NCCL will hang with 25.03, so use 25.01 for now
# Please check https://github.com/ai-dynamo/dynamo/pull/1065
# for details and reproducer to manually test if the image
# can be updated to later versions.
VLLM_BASE_IMAGE_TAG="25.01-cuda12.8-devel-ubuntu24.04"

NONE_BASE_IMAGE="nvcr.io/nvidia/cuda-dl-base"
NONE_BASE_IMAGE_TAG="25.01-cuda12.8-devel-ubuntu24.04"

SGLANG_CUDA_VERSION="12.9.1"
# This is for Dockerfile
SGLANG_BASE_IMAGE="nvcr.io/nvidia/cuda-dl-base"
SGLANG_BASE_IMAGE_TAG="25.01-cuda12.8-devel-ubuntu24.04"
# This is for Dockerfile.sglang. Unlike the other frameworks, it is using a different base image
SGLANG_FRAMEWORK_IMAGE="nvcr.io/nvidia/cuda"
SGLANG_FRAMEWORK_IMAGE_TAG="${SGLANG_CUDA_VERSION}-cudnn-devel-ubuntu24.04"

NIXL_REF=0.7.1
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
            # Note: --base-image cannot be used with --dev-image
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
        --dev-image)
            if [ "$2" ]; then
                DEV_IMAGE_INPUT=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --uid)
            if [ "$2" ]; then
                CUSTOM_UID=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --gid)
            if [ "$2" ]; then
                CUSTOM_GID=$2
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
            DRY_RUN="true"
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
        --vllm-max-jobs)
            # Set MAX_JOBS for vLLM compilation (only used by Dockerfile.vllm)
            if [ "$2" ]; then
                MAX_JOBS=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --no-tag-latest)
            NO_TAG_LATEST=true
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

    # Validate argument combinations
    if [[ -n "${DEV_IMAGE_INPUT:-}" && -n "${BASE_IMAGE:-}" ]]; then
        error "ERROR: --dev-image cannot be used with --base-image. Use --dev-image to build from existing images or --base-image to build new images."
    fi

    # Validate that --target and --dev-image cannot be used together
    if [[ -n "${DEV_IMAGE_INPUT:-}" && -n "${TARGET:-}" ]]; then
        error "ERROR: --target cannot be used with --dev-image. Use --target to build from scratch or --dev-image to build from existing images."
    fi

    # Validate that --uid and --gid are only used with local-dev related options
    if [[ -n "${CUSTOM_UID:-}" || -n "${CUSTOM_GID:-}" ]]; then
        if [[ -z "${DEV_IMAGE_INPUT:-}" && "${TARGET:-}" != "local-dev" ]]; then
            error "ERROR: --uid and --gid can only be used with --dev-image or --target local-dev"
        fi
    fi

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
        if [ -n "${TARGET}" ] && [ "${TARGET}" != "local-dev" ]; then
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
    echo "=========================================="
    echo "🐳 DOCKER BUILD CONFIGURATION"
    echo "=========================================="
    echo ""
    echo "📦 Image Configuration:"
    echo "   Tag: '${TAG}'"
    echo "   Base: '${BASE_IMAGE}'"
    echo "   Base Image Tag: '${BASE_IMAGE_TAG}'"
    echo "   Framework: '${FRAMEWORK}'"
    echo "   Platform: '${PLATFORM}'"
    echo "   Target: '${TARGET_STR}'"
    echo ""
    
    if [[ $FRAMEWORK == "TRTLLM" ]]; then
        echo "🔧 TensorRT-LLM Configuration:"
        echo "   Pip Wheel: '${PRINT_TRTLLM_WHEEL_FILE}'"
        echo "   Intention: '${TRTLLM_INTENTION}'"
        echo ""
    fi
    
    echo "📂 Build Configuration:"
    echo "   Build Context: '${BUILD_CONTEXT}'"
    echo "   Dockerfile: '${DOCKERFILE}'"
    echo "   Architecture: '${ARCH}'"
    echo ""
    
    if [ "$USE_SCCACHE" = true ]; then
        echo "⚡ sccache Configuration:"
        echo "   Status: Enabled"
        echo "   Bucket: '${SCCACHE_BUCKET}'"
        echo "   Region: '${SCCACHE_REGION}'"
        if [ -n "$SCCACHE_S3_KEY_PREFIX" ]; then
            echo "   S3 Key Prefix: '${SCCACHE_S3_KEY_PREFIX}'"
        fi
        echo ""
    else
        echo "⚡ sccache: Disabled"
        echo ""
    fi
    
    if [ -n "$NO_CACHE" ]; then
        echo "🔄 Cache: Disabled (--no-cache)"
        echo ""
    fi
    
    if [ -n "$CACHE_FROM" ]; then
        echo "📥 Cache From: '${CACHE_FROM}'"
        echo ""
    fi
    
    if [ -n "$CACHE_TO" ]; then
        echo "📤 Cache To: '${CACHE_TO}'"
        echo ""
    fi
    
    echo "🔨 Build Arguments:"
    echo "${BUILD_ARGS}" | tr ' ' '\n' | grep -v '^$' | sed 's/^/   /'
    echo ""
    echo "=========================================="
    echo ""
}

show_help() {
    echo "usage: build.sh"
    echo "  [--base-image base image]"
    echo "  [--base-image-tag base image tag]"
    echo "  [--platform platform for docker build]"
    echo "  [--framework framework one of ${!FRAMEWORKS[*]}]"
    echo "  [--tensorrtllm-pip-wheel-dir path to tensorrtllm pip wheel directory]"
    echo "  [--tensorrtllm-commit tensorrtllm commit/tag/branch to use for building the trtllm wheel if the wheel is not provided]"
    echo "  [--tensorrtllm-pip-wheel tensorrtllm pip wheel on artifactory]"
    echo "  [--tensorrtllm-index-url tensorrtllm PyPI index URL if providing the wheel from artifactory]"
    echo "  [--tensorrtllm-git-url tensorrtllm git repository URL for cloning]"
    echo "  [--build-arg additional build args to pass to docker build]"
    echo "  [--cache-from cache location to start from]"
    echo "  [--cache-to location where to cache the build output]"
    echo "  [--tag tag for image]"
    echo "  [--dev-image dev image to build local-dev from]"
    echo "  [--uid user ID for local-dev images (only with --dev-image or --target local-dev)]"
    echo "  [--gid group ID for local-dev images (only with --dev-image or --target local-dev)]"
    echo "  [--no-cache disable docker build cache]"
    echo "  [--dry-run print docker commands without running]"
    echo "  [--build-context name=path to add build context]"
    echo "  [--release-build perform a release build]"
    echo "  [--make-efa Enables EFA support for NIXL]"
    echo "  [--enable-kvbm Enables KVBM support in Python 3.12]"
    echo "  [--use-sccache enable sccache for Rust/C/C++ compilation caching]"
    echo "  [--sccache-bucket S3 bucket name for sccache (required with --use-sccache)]"
    echo "  [--sccache-region S3 region for sccache (required with --use-sccache)]"
    echo "  [--vllm-max-jobs number of parallel jobs for compilation (only used by vLLM framework)]"
    echo "  [--no-tag-latest do not add latest-{framework} tag to built image]"
    echo ""
    echo "  Note: When using --use-sccache, AWS credentials must be set:"
    echo "        export AWS_ACCESS_KEY_ID=your_access_key"
    echo "        export AWS_SECRET_ACCESS_KEY=your_secret_key"
    exit 0
}

missing_requirement() {
    error "ERROR: $1 requires an argument."
}

error() {
    printf '%s %s\n' "$1" "$2" >&2
    exit 1
}

echo "=========================================="
echo "🚀 DYNAMO BUILD SCRIPT STARTING"
echo "=========================================="
echo "Script: $0"
echo "Arguments: $@"
echo "Working Directory: $(pwd)"
echo "User: $(whoami)"
echo "Date: $(date)"
echo "=========================================="
echo ""

get_options "$@"

echo "✅ Command line arguments parsed"
echo ""

# Automatically set ARCH and ARCH_ALT if PLATFORM is linux/arm64
echo "🔍 Detecting architecture from platform..."
ARCH="amd64"
if [[ "$PLATFORM" == *"linux/arm64"* ]]; then
    ARCH="arm64"
    BUILD_ARGS+=" --build-arg ARCH=arm64 --build-arg ARCH_ALT=aarch64 "
    echo "   Detected: arm64 (aarch64)"
else
    echo "   Detected: amd64"
fi
echo ""

# Set the commit sha in the container so we can inspect what build this relates to
echo "📝 Setting build metadata..."
DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA:-$(git rev-parse HEAD)}
BUILD_ARGS+=" --build-arg DYNAMO_COMMIT_SHA=$DYNAMO_COMMIT_SHA "
echo "   Commit SHA: ${DYNAMO_COMMIT_SHA}"
echo ""

# Special handling for vLLM on ARM64 - set required defaults if not already specified by user
if [[ $FRAMEWORK == "VLLM" ]] && [[ "$PLATFORM" == *"linux/arm64"* ]]; then
    echo "🔧 Applying vLLM ARM64-specific configuration..."
    
    # Set base image tag to CUDA 12.9 if using the default value (user didn't override)
    if [ "$BASE_IMAGE_TAG" == "$VLLM_BASE_IMAGE_TAG" ]; then
        BASE_IMAGE_TAG="25.06-cuda12.9-devel-ubuntu24.04"
        echo "   ✓ Base image tag: $BASE_IMAGE_TAG"
    fi

    # Add required build args if not already present
    if [[ "$BUILD_ARGS" != *"RUNTIME_IMAGE_TAG"* ]]; then
        BUILD_ARGS+=" --build-arg RUNTIME_IMAGE_TAG=12.9.0-runtime-ubuntu24.04 "
        echo "   ✓ Runtime image tag: 12.9.0-runtime-ubuntu24.04"
    fi

    if [[ "$BUILD_ARGS" != *"CUDA_VERSION"* ]]; then
        BUILD_ARGS+=" --build-arg CUDA_VERSION=129 "
        echo "   ✓ CUDA version: 129"
    fi

    if [[ "$BUILD_ARGS" != *"TORCH_BACKEND"* ]]; then
        BUILD_ARGS+=" --build-arg TORCH_BACKEND=cu129 "
        echo "   ✓ Torch backend: cu129"
    fi
    
    echo ""
fi

# Update DOCKERFILE if framework is VLLM
echo "📄 Selecting Dockerfile for framework: ${FRAMEWORK}"
if [[ $FRAMEWORK == "VLLM" ]]; then
    DOCKERFILE=${SOURCE_DIR}/Dockerfile.vllm
    echo "   Using: Dockerfile.vllm"
elif [[ $FRAMEWORK == "TRTLLM" ]]; then
    DOCKERFILE=${SOURCE_DIR}/Dockerfile.trtllm
    echo "   Using: Dockerfile.trtllm"
elif [[ $FRAMEWORK == "NONE" ]]; then
    DOCKERFILE=${SOURCE_DIR}/Dockerfile
    echo "   Using: Dockerfile (base)"
elif [[ $FRAMEWORK == "SGLANG" ]]; then
    DOCKERFILE=${SOURCE_DIR}/Dockerfile.sglang
    echo "   Using: Dockerfile.sglang"
fi
echo ""

# Add NIXL_REF as a build argument
BUILD_ARGS+=" --build-arg NIXL_REF=${NIXL_REF} "

# Function to build local-dev image with header
build_local_dev_with_header() {
    local dev_base_image="$1"
    local tags="$2"
    local success_msg="$3"
    local header_title="$4"

    echo "======================================"
    echo "$header_title"
    echo "======================================"

    # Get user info right before using it
    USER_UID=${CUSTOM_UID:-$(id -u)}
    USER_GID=${CUSTOM_GID:-$(id -g)}

    # Set up dockerfile path
    DOCKERFILE_LOCAL_DEV="${SOURCE_DIR}/Dockerfile.local_dev"

    if [[ ! -f "$DOCKERFILE_LOCAL_DEV" ]]; then
        echo "ERROR: Dockerfile.local_dev not found at: $DOCKERFILE_LOCAL_DEV"
        exit 1
    fi

    echo "Building new local-dev image from: $dev_base_image"
    echo "User 'dynamo' will have UID: $USER_UID, GID: $USER_GID"

    # Show the docker command being executed if not in dry-run mode
    if [ -z "$RUN_PREFIX" ]; then
        set -x
    fi

    $RUN_PREFIX docker build \
        --build-arg DEV_BASE="$dev_base_image" \
        --build-arg USER_UID="$USER_UID" \
        --build-arg USER_GID="$USER_GID" \
        --build-arg ARCH="$ARCH" \
        --file "$DOCKERFILE_LOCAL_DEV" \
        $tags \
        "$SOURCE_DIR" || {
        { set +x; } 2>/dev/null
        echo "ERROR: Failed to build local_dev image"
        exit 1
    }

    { set +x; } 2>/dev/null
    echo "$success_msg"

    # Show usage instructions
    echo ""
    echo "To run the local-dev image as the local user ($USER_UID/$USER_GID):"
    # Extract the last tag from the tags string
    last_tag=$(echo "$tags" | grep -o -- '--tag [^ ]*' | tail -1 | cut -d' ' -f2)
    # Calculate relative path to run.sh from current working directory
    # Get the directory where build.sh is located
    build_dir="$(dirname "${BASH_SOURCE[0]}")"
    # Get the absolute path to run.sh (in the same directory as build.sh)
    run_abs_path="$(realpath "$build_dir/run.sh")"
    # Calculate relative path from current PWD to run.sh
    run_path="$(python3 -c "import os; print(os.path.relpath('$run_abs_path', '$PWD'))")"
    echo "  $run_path --image $last_tag --mount-workspace ..."
}


# Handle local-dev target
if [[ $TARGET == "local-dev" ]]; then
    LOCAL_DEV_BUILD=true
    TARGET_STR="--target dev"
fi

# BUILD DEV IMAGE

BUILD_ARGS+=" --build-arg BASE_IMAGE=$BASE_IMAGE --build-arg BASE_IMAGE_TAG=$BASE_IMAGE_TAG"

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
    fi
    echo "Found $wheel_count wheel in $wheel_dir"
    return 0
}

function determine_user_intention_trtllm() {
    # The tensorrt llm installation flags are not quite mutually exclusive
    # since the user should be able to point at a directory of their choosing
    # for storing a trtllm wheel built from source.
    #
    # This function attempts to discern the intention of the user by
    # applying checks, or rules, for each of the scenarios.
    #
    # /return: Calculated intention. One of "download", "install", "build".
    #
    # The three different methods of installing TRTLLM with build.sh are:
    # 1. Download
    # required: --tensorrtllm-pip-wheel
    # optional: --tensorrtllm-index-url
    # optional: --tensorrtllm-commit
    #
    # 2. Install from pre-built
    # required: --tensorrtllm-pip-wheel-dir
    # optional: --tensorrtllm-commit
    #
    # 3. Build from source
    # required: --tensorrtllm-git-url
    # optional: --tensorrtllm-commit
    # optional: --tensorrtllm-pip-wheel-dir
    local intention_download="false"
    local intention_install="false"
    local intention_build="false"
    local intention_count=0
    TRTLLM_INTENTION=${TRTLLM_INTENTION}

    # Install from pre-built
    if [[ -n "$TENSORRTLLM_PIP_WHEEL_DIR"  && ! -n "$TRTLLM_GIT_URL" ]]; then
        intention_install="true";
        intention_count=$((intention_count+1))
        TRTLLM_INTENTION="install"
    fi
    echo "  Intent to Install TRTLLM: $intention_install"

    # Build from source
    if [[ -n "$TRTLLM_GIT_URL" ]]; then
        intention_build="true";
        intention_count=$((intention_count+1))
        TRTLLM_INTENTION="build"
    fi
    echo "  Intent to Build TRTLLM: $intention_build"

    # Download from repository
    if [[ -n "$TENSORRTLLM_INDEX_URL" ]] && [[ -n "$TENSORRTLLM_PIP_WHEEL" ]]; then
        intention_download="true";
        intention_count=$((intention_count+1));
        TRTLLM_INTENTION="download"
        echo "INFO: Installing $TENSORRTLLM_PIP_WHEEL trtllm version from index: $TENSORRTLLM_INDEX_URL"
    elif [[ -n "$TENSORRTLLM_PIP_WHEEL" ]]; then
        intention_download="true";
        intention_count=$((intention_count+1));
        TRTLLM_INTENTION="download"
        echo "INFO: Installing $TENSORRTLLM_PIP_WHEEL trtllm version from default pip index."
    fi

    # If nothing is set then we default to downloading the wheel
    # with the defaults sepcified at the top this file.
    if [[ -z "${TENSORRTLLM_INDEX_URL}" ]] && [[ -z "${TENSORRTLLM_PIP_WHEEL}" ]] && [[ "${intention_count}" -eq 0 ]]; then
        intention_download="true";
        intention_count=$((intention_count+1))
        TRTLLM_INTENTION="download"
        echo "INFO: Inferring download because both TENSORRTLLM_PIP_WHEEL and TENSORRTLLM_INDEX_URL are not set."
    fi
    echo "  Intent to Download TRTLLM: $intention_download"

    if [[ ! "$intention_count" -eq 1 ]]; then
        echo -e "[ERROR] Could not figure out the trtllm installation intent from the current flags. Please check your build.sh command against the following"
        echo -e "  The grouped flags are mutually exclusive:"
        echo -e "  To download and install use both: --tensorrtllm-index-url, --tensorrtllm-pip-wheel"
        echo -e "  To install from a pre-built wheel use: --tensorrtllm-pip-wheel-dir"
        echo -e "  To build from source and install use both: --tensorrtllm-commit, --tensorrtllm-git-url"
        exit 1
    fi
}


if [[ $FRAMEWORK == "TRTLLM" ]]; then
    echo "=========================================="
    echo "🔧 TRTLLM CONFIGURATION"
    echo "=========================================="
    echo "Determining the user's TRTLLM installation intent..."
    echo ""
    determine_user_intention_trtllm   # From this point forward, can assume correct TRTLLM flags
    echo ""

    if [[ "$TRTLLM_INTENTION" == "download" ]]; then
        echo "📥 TRTLLM: Download mode"
        TENSORRTLLM_INDEX_URL=${TENSORRTLLM_INDEX_URL:-$DEFAULT_TENSORRTLLM_INDEX_URL}
        TENSORRTLLM_PIP_WHEEL=${TENSORRTLLM_PIP_WHEEL:-$DEFAULT_TENSORRTLLM_PIP_WHEEL}
        BUILD_ARGS+=" --build-arg HAS_TRTLLM_CONTEXT=0"
        BUILD_ARGS+=" --build-arg TENSORRTLLM_PIP_WHEEL=${TENSORRTLLM_PIP_WHEEL}"
        BUILD_ARGS+=" --build-arg TENSORRTLLM_INDEX_URL=${TENSORRTLLM_INDEX_URL}"
        
        echo "   Index URL: ${TENSORRTLLM_INDEX_URL}"
        echo "   Pip Wheel: ${TENSORRTLLM_PIP_WHEEL}"

        # Create a dummy directory to satisfy the build context requirement
        # There is no way to conditionally copy the build context in dockerfile.
        mkdir -p /tmp/dummy_dir
        BUILD_CONTEXT_ARG+=" --build-context trtllm_wheel=/tmp/dummy_dir"
        PRINT_TRTLLM_WHEEL_FILE=${TENSORRTLLM_PIP_WHEEL}
        echo "   ✓ Dummy wheel context created"
    elif [[ "$TRTLLM_INTENTION" == "install" ]]; then
        echo "📦 TRTLLM: Install from wheel mode"
        echo "   Wheel directory: ${TENSORRTLLM_PIP_WHEEL_DIR}"
        echo "   Checking for TensorRT-LLM wheel..."
        if ! check_wheel_file "${TENSORRTLLM_PIP_WHEEL_DIR}"; then
            echo ""
            echo "❌ ERROR: Valid trtllm wheel file not found in ${TENSORRTLLM_PIP_WHEEL_DIR}"
            echo "   If this is not intended you can try building from source with:"
            echo "   --tensorrtllm-git-url https://github.com/NVIDIA/TensorRT-LLM --tensorrtllm-commit $TRTLLM_COMMIT"
            exit 1
        fi
        echo "   ✓ Wheel found"
        BUILD_ARGS+=" --build-arg HAS_TRTLLM_CONTEXT=1"
        BUILD_CONTEXT_ARG+=" --build-context trtllm_wheel=${TENSORRTLLM_PIP_WHEEL_DIR}"
        PRINT_TRTLLM_WHEEL_FILE=$(find $TENSORRTLLM_PIP_WHEEL_DIR -name "*.whl" | head -n 1)
        echo "   Using: $(basename ${PRINT_TRTLLM_WHEEL_FILE})"
    elif [[ "$TRTLLM_INTENTION" == "build" ]]; then
        echo "🔨 TRTLLM: Build from source mode"
        TENSORRTLLM_PIP_WHEEL_DIR=${TENSORRTLLM_PIP_WHEEL_DIR:=$DEFAULT_TENSORRTLLM_PIP_WHEEL_DIR}
        echo "   Output directory: ${TENSORRTLLM_PIP_WHEEL_DIR}"
        echo "   Commit: ${TRTLLM_COMMIT}"
        echo "   Architecture: ${ARCH}"
        if [ "$DRY_RUN" != "true" ]; then
            GIT_URL_ARG=""
            if [ -n "${TRTLLM_GIT_URL}" ]; then
                GIT_URL_ARG="-u ${TRTLLM_GIT_URL}"
                echo "   Git URL: ${TRTLLM_GIT_URL}"
            fi
            echo ""
            echo "   🚀 Starting TensorRT-LLM wheel build..."
            if ! env -i ${SOURCE_DIR}/build_trtllm_wheel.sh -o ${TENSORRTLLM_PIP_WHEEL_DIR} -c ${TRTLLM_COMMIT} -a ${ARCH} -n ${NIXL_REF} ${GIT_URL_ARG}; then
                error "ERROR: Failed to build TensorRT-LLM wheel"
            fi
            echo "   ✅ Wheel build completed"
            BUILD_ARGS+=" --build-arg HAS_TRTLLM_CONTEXT=1"
            BUILD_CONTEXT_ARG+=" --build-context trtllm_wheel=${TENSORRTLLM_PIP_WHEEL_DIR}"
            PRINT_TRTLLM_WHEEL_FILE=$(find $TENSORRTLLM_PIP_WHEEL_DIR -name "*.whl" | head -n 1)
            echo "   Built: $(basename ${PRINT_TRTLLM_WHEEL_FILE})"
        fi
    else
        echo 'No intention was set. This error should have been detected in "determine_user_intention_trtllm()". Exiting...'
        exit 1
    fi

    # Need to know the commit of TRTLLM so we can determine the
    # TensorRT installation associated with TRTLLM.
    if [[ -z "$TRTLLM_COMMIT" ]]; then
        # Attempt to default since the commit will work with a hash or a tag/branch
        if [[ ! -z "$TENSORRTLLM_PIP_WHEEL" ]]; then
            TRTLLM_COMMIT=$(echo "${TENSORRTLLM_PIP_WHEEL}" | sed -n 's/.*==\([0-9a-zA-Z\.\-]*\).*/\1/p')
            echo "Attempting to default TRTLLM_COMMIT to \"$TRTLLM_COMMIT\" for installation of TensorRT."
        else
            echo -e "[ERROR] TRTLLM framework was set as a target but the TRTLLM_COMMIT variable was not set."
            echo -e "  Could not find a suitible default by infering from TENSORRTLLM_PIP_WHEEL."
            echo -e "  TRTLLM_COMMIT is needed to install the correct version of TensorRT associated with TensorRT-LLM."
            exit 1
        fi
    fi
    BUILD_ARGS+=" --build-arg GITHUB_TRTLLM_COMMIT=${TRTLLM_COMMIT}"


fi

# ENABLE_KVBM: Used in base Dockerfile for block-manager feature.
#              Declared but not currently used in Dockerfile.{vllm,trtllm}.
if [[ $FRAMEWORK == "VLLM" ]] || [[ $FRAMEWORK == "TRTLLM" ]]; then
    echo "🔧 Forcing KVBM enabled for ${FRAMEWORK} framework"
    ENABLE_KVBM=true
else
    ENABLE_KVBM=false
fi

if [  ! -z ${ENABLE_KVBM} ]; then
    echo "   ✓ KVBM enabled in the dynamo image"
    BUILD_ARGS+=" --build-arg ENABLE_KVBM=${ENABLE_KVBM} "
fi
echo ""

# NIXL_UCX_REF: Used in base Dockerfile only.
#               Passed to framework Dockerfile.{vllm,sglang,...} where it's NOT used.
if [ -n "${NIXL_UCX_REF}" ]; then
    BUILD_ARGS+=" --build-arg NIXL_UCX_REF=${NIXL_UCX_REF} "
fi

# MAX_JOBS is only used by Dockerfile.vllm
if [ -n "${MAX_JOBS}" ]; then
    BUILD_ARGS+=" --build-arg MAX_JOBS=${MAX_JOBS} "
fi

if [[ $FRAMEWORK == "SGLANG" ]]; then
    echo "Customizing Python, CUDA, and framework images for sglang images"
    BUILD_ARGS+=" --build-arg PYTHON_VERSION=3.10"
    BUILD_ARGS+=" --build-arg CUDA_VERSION=${SGLANG_CUDA_VERSION}"
    # Unlike the other two frameworks, SGLang's framework image is different from the base image, so we need to set it explicitly.
    BUILD_ARGS+=" --build-arg FRAMEWORK_IMAGE=${SGLANG_FRAMEWORK_IMAGE}"
    BUILD_ARGS+=" --build-arg FRAMEWORK_IMAGE_TAG=${SGLANG_FRAMEWORK_IMAGE_TAG}"
else
    BUILD_ARGS+=" --build-arg PYTHON_VERSION=3.12"
fi
# Add sccache build arguments
if [ "$USE_SCCACHE" = true ]; then
    echo "⚡ Configuring sccache build arguments..."
    BUILD_ARGS+=" --build-arg USE_SCCACHE=true"
    BUILD_ARGS+=" --build-arg SCCACHE_BUCKET=${SCCACHE_BUCKET}"
    BUILD_ARGS+=" --build-arg SCCACHE_REGION=${SCCACHE_REGION}"
    BUILD_ARGS+=" --secret id=aws-key-id,env=AWS_ACCESS_KEY_ID"
    BUILD_ARGS+=" --secret id=aws-secret-id,env=AWS_SECRET_ACCESS_KEY"
    echo "   ✓ sccache arguments added"
    echo ""
fi
if [[ "$PLATFORM" == *"linux/arm64"* && "${FRAMEWORK}" == "SGLANG" ]]; then
    # Add arguments required for sglang blackwell build
    echo "🔧 Configuring SGLang for Grace Blackwell (ARM64)..."
    BUILD_ARGS+=" --build-arg GRACE_BLACKWELL=true --build-arg BUILD_TYPE=blackwell_aarch64"
    echo "   ✓ Grace Blackwell arguments added"
    echo ""
fi
LATEST_TAG=""
if [ -z "${NO_TAG_LATEST}" ]; then
    LATEST_TAG="--tag dynamo:latest-${FRAMEWORK,,}"
    if [ -n "${TARGET}" ] && [ "${TARGET}" != "local-dev" ]; then
        LATEST_TAG="${LATEST_TAG}-${TARGET}"
    fi
fi

show_image_options

if [ -z "$RUN_PREFIX" ]; then
    set -x
fi


# Skip Build 1 and Build 2 if DEV_IMAGE_INPUT is set (we'll handle it at the bottom)
if [[ -z "${DEV_IMAGE_INPUT:-}" ]]; then
    # Follow 2-step build process for all frameworks
    if [[ $FRAMEWORK != "NONE" ]]; then
        # Define base image tag with framework suffix to prevent clobbering
        # Different frameworks require different base configurations:
        # - VLLM: Python 3.12, ENABLE_KVBM=true, BASE_IMAGE=cuda-dl-base
        # - SGLANG: Python 3.10, BASE_IMAGE=cuda-dl-base
        # - TRTLLM: Python 3.12, ENABLE_KVBM=true, BASE_IMAGE=pytorch
        # Without unique tags, building different frameworks would overwrite each other's names
        DYNAMO_BASE_IMAGE="dynamo-base:${VERSION}-${FRAMEWORK,,}"
        # Start base image build
        echo "=========================================="
        echo "🏗️  BUILD STEP 1: BASE IMAGE"
        echo "=========================================="
        echo "Building base image: ${DYNAMO_BASE_IMAGE}"
        echo "Dockerfile: ${SOURCE_DIR}/Dockerfile"
        echo "Target: dev"
        echo ""
        
        # Create build log directory if it doesn't exist
        BUILD_LOG_DIR="${BUILD_CONTEXT}/build-logs"
        mkdir -p "${BUILD_LOG_DIR}"
        BASE_BUILD_LOG="${BUILD_LOG_DIR}/base-image-build.log"
        echo "📝 Capturing build output to: ${BASE_BUILD_LOG}"
        echo ""
        
        BUILD_START=$(date +%s)
        
        # Capture build output to log file while showing on console
        # Use BuildKit for enhanced metadata (docker buildx or DOCKER_BUILDKIT=1)
        if [ -z "$RUN_PREFIX" ]; then
            # Check if buildx is available
            if docker buildx version &>/dev/null; then
                echo "   Using BuildKit via docker buildx..."
                docker buildx build --progress=plain --load -f "${SOURCE_DIR}/Dockerfile" --target dev $PLATFORM $BUILD_ARGS $CACHE_FROM $CACHE_TO --tag $DYNAMO_BASE_IMAGE $BUILD_CONTEXT_ARG $BUILD_CONTEXT $NO_CACHE 2>&1 | tee "${BASE_BUILD_LOG}"
                BUILD_EXIT_CODE=${PIPESTATUS[0]}
            else
                echo "   Using BuildKit via DOCKER_BUILDKIT=1..."
                DOCKER_BUILDKIT=1 docker build --progress=plain -f "${SOURCE_DIR}/Dockerfile" --target dev $PLATFORM $BUILD_ARGS $CACHE_FROM $CACHE_TO --tag $DYNAMO_BASE_IMAGE $BUILD_CONTEXT_ARG $BUILD_CONTEXT $NO_CACHE 2>&1 | tee "${BASE_BUILD_LOG}"
                BUILD_EXIT_CODE=${PIPESTATUS[0]}
            fi
        else
            $RUN_PREFIX docker build -f "${SOURCE_DIR}/Dockerfile" --target dev $PLATFORM $BUILD_ARGS $CACHE_FROM $CACHE_TO --tag $DYNAMO_BASE_IMAGE $BUILD_CONTEXT_ARG $BUILD_CONTEXT $NO_CACHE
            BUILD_EXIT_CODE=$?
        fi
        
        BUILD_END=$(date +%s)
        BUILD_DURATION=$((BUILD_END - BUILD_START))
        
        if [ ${BUILD_EXIT_CODE} -ne 0 ]; then
            echo ""
            echo "❌ Base image build FAILED with exit code ${BUILD_EXIT_CODE}"
            echo "   Build log saved to: ${BASE_BUILD_LOG}"
            exit ${BUILD_EXIT_CODE}
        fi
        
        echo ""
        echo "✅ Base image build completed in ${BUILD_DURATION} seconds"
        
        # Extract layer statistics from build log
        if [ -f "${BASE_BUILD_LOG}" ]; then
            CACHED_LAYERS=$(grep -c "CACHED" "${BASE_BUILD_LOG}" 2>/dev/null || echo "0")
            TOTAL_STEPS=$(grep -c "^Step [0-9]" "${BASE_BUILD_LOG}" 2>/dev/null || echo "0")
            echo "📊 Build statistics:"
            echo "   Total steps: ${TOTAL_STEPS}"
            echo "   Cached layers: ${CACHED_LAYERS}"
            if [ ${TOTAL_STEPS} -gt 0 ]; then
                CACHE_RATE=$(awk "BEGIN {printf \"%.1f\", (${CACHED_LAYERS} / ${TOTAL_STEPS}) * 100}")
                echo "   Cache hit rate: ${CACHE_RATE}%"
            fi
        fi
        echo ""
        
        # Start framework build
        echo "=========================================="
        echo "🏗️  BUILD STEP 2: FRAMEWORK IMAGE"
        echo "=========================================="
        echo "Building framework image for: ${FRAMEWORK}"
        echo "Dockerfile: ${DOCKERFILE}"
        echo "Target: ${TARGET_STR}"
        echo "Base image: ${DYNAMO_BASE_IMAGE}"
        echo ""
        
        # Determine framework build log name
        FRAMEWORK_BUILD_LOG="${BUILD_LOG_DIR}/framework-${FRAMEWORK,,}-build.log"
        echo "📝 Capturing build output to: ${FRAMEWORK_BUILD_LOG}"
        echo ""
        
        BUILD_START=$(date +%s)
        
        BUILD_ARGS+=" --build-arg DYNAMO_BASE_IMAGE=${DYNAMO_BASE_IMAGE}"
        
        # Capture build output to log file while showing on console
        # Use BuildKit for enhanced metadata (docker buildx or DOCKER_BUILDKIT=1)
        if [ -z "$RUN_PREFIX" ]; then
            # Check if buildx is available
            if docker buildx version &>/dev/null; then
                echo "   Using BuildKit via docker buildx..."
                docker buildx build --progress=plain --load -f $DOCKERFILE $TARGET_STR $PLATFORM $BUILD_ARGS $CACHE_FROM $CACHE_TO $TAG $LATEST_TAG $BUILD_CONTEXT_ARG $BUILD_CONTEXT $NO_CACHE 2>&1 | tee "${FRAMEWORK_BUILD_LOG}"
                BUILD_EXIT_CODE=${PIPESTATUS[0]}
            else
                echo "   Using BuildKit via DOCKER_BUILDKIT=1..."
                DOCKER_BUILDKIT=1 docker build --progress=plain -f $DOCKERFILE $TARGET_STR $PLATFORM $BUILD_ARGS $CACHE_FROM $CACHE_TO $TAG $LATEST_TAG $BUILD_CONTEXT_ARG $BUILD_CONTEXT $NO_CACHE 2>&1 | tee "${FRAMEWORK_BUILD_LOG}"
                BUILD_EXIT_CODE=${PIPESTATUS[0]}
            fi
        else
            $RUN_PREFIX docker build -f $DOCKERFILE $TARGET_STR $PLATFORM $BUILD_ARGS $CACHE_FROM $CACHE_TO $TAG $LATEST_TAG $BUILD_CONTEXT_ARG $BUILD_CONTEXT $NO_CACHE
            BUILD_EXIT_CODE=$?
        fi
        
        BUILD_END=$(date +%s)
        BUILD_DURATION=$((BUILD_END - BUILD_START))
        
        if [ ${BUILD_EXIT_CODE} -ne 0 ]; then
            echo ""
            echo "❌ Framework image build FAILED with exit code ${BUILD_EXIT_CODE}"
            echo "   Build log saved to: ${FRAMEWORK_BUILD_LOG}"
            exit ${BUILD_EXIT_CODE}
        fi
        
        echo ""
        echo "✅ Framework image build completed in ${BUILD_DURATION} seconds"
        
        # Extract layer statistics from build log
        if [ -f "${FRAMEWORK_BUILD_LOG}" ]; then
            CACHED_LAYERS=$(grep -c "CACHED" "${FRAMEWORK_BUILD_LOG}" 2>/dev/null || echo "0")
            TOTAL_STEPS=$(grep -c "^Step [0-9]" "${FRAMEWORK_BUILD_LOG}" 2>/dev/null || echo "0")
            PULL_COMPLETE=$(grep -c "Pull complete" "${FRAMEWORK_BUILD_LOG}" 2>/dev/null || echo "0")
            echo "📊 Build statistics:"
            echo "   Total steps: ${TOTAL_STEPS}"
            echo "   Cached layers: ${CACHED_LAYERS}"
            echo "   Layers pulled: ${PULL_COMPLETE}"
            if [ ${TOTAL_STEPS} -gt 0 ]; then
                CACHE_RATE=$(awk "BEGIN {printf \"%.1f\", (${CACHED_LAYERS} / ${TOTAL_STEPS}) * 100}")
                echo "   Cache hit rate: ${CACHE_RATE}%"
            fi
        fi
        echo ""
    else
        echo "=========================================="
        echo "🏗️  SINGLE-STAGE BUILD"
        echo "=========================================="
        echo "Building image (no framework)..."
        echo "Dockerfile: ${DOCKERFILE}"
        echo ""
        
        # Create build log directory if it doesn't exist
        BUILD_LOG_DIR="${BUILD_CONTEXT}/build-logs"
        mkdir -p "${BUILD_LOG_DIR}"
        SINGLE_BUILD_LOG="${BUILD_LOG_DIR}/single-stage-build.log"
        echo "📝 Capturing build output to: ${SINGLE_BUILD_LOG}"
        echo ""
        
        BUILD_START=$(date +%s)
        
        # Capture build output to log file while showing on console
        # Use BuildKit for enhanced metadata (docker buildx or DOCKER_BUILDKIT=1)
        if [ -z "$RUN_PREFIX" ]; then
            # Check if buildx is available
            if docker buildx version &>/dev/null; then
                echo "   Using BuildKit via docker buildx..."
                docker buildx build --progress=plain --load -f $DOCKERFILE $TARGET_STR $PLATFORM $BUILD_ARGS $CACHE_FROM $CACHE_TO $TAG $LATEST_TAG $BUILD_CONTEXT_ARG $BUILD_CONTEXT $NO_CACHE 2>&1 | tee "${SINGLE_BUILD_LOG}"
                BUILD_EXIT_CODE=${PIPESTATUS[0]}
            else
                echo "   Using BuildKit via DOCKER_BUILDKIT=1..."
                DOCKER_BUILDKIT=1 docker build --progress=plain -f $DOCKERFILE $TARGET_STR $PLATFORM $BUILD_ARGS $CACHE_FROM $CACHE_TO $TAG $LATEST_TAG $BUILD_CONTEXT_ARG $BUILD_CONTEXT $NO_CACHE 2>&1 | tee "${SINGLE_BUILD_LOG}"
                BUILD_EXIT_CODE=${PIPESTATUS[0]}
            fi
        else
            $RUN_PREFIX docker build -f $DOCKERFILE $TARGET_STR $PLATFORM $BUILD_ARGS $CACHE_FROM $CACHE_TO $TAG $LATEST_TAG $BUILD_CONTEXT_ARG $BUILD_CONTEXT $NO_CACHE
            BUILD_EXIT_CODE=$?
        fi
        
        BUILD_END=$(date +%s)
        BUILD_DURATION=$((BUILD_END - BUILD_START))
        
        if [ ${BUILD_EXIT_CODE} -ne 0 ]; then
            echo ""
            echo "❌ Image build FAILED with exit code ${BUILD_EXIT_CODE}"
            echo "   Build log saved to: ${SINGLE_BUILD_LOG}"
            exit ${BUILD_EXIT_CODE}
        fi
        
        echo ""
        echo "✅ Image build completed in ${BUILD_DURATION} seconds"
        
        # Extract layer statistics from build log
        if [ -f "${SINGLE_BUILD_LOG}" ]; then
            CACHED_LAYERS=$(grep -c "CACHED" "${SINGLE_BUILD_LOG}" 2>/dev/null || echo "0")
            TOTAL_STEPS=$(grep -c "^Step [0-9]" "${SINGLE_BUILD_LOG}" 2>/dev/null || echo "0")
            echo "📊 Build statistics:"
            echo "   Total steps: ${TOTAL_STEPS}"
            echo "   Cached layers: ${CACHED_LAYERS}"
            if [ ${TOTAL_STEPS} -gt 0 ]; then
                CACHE_RATE=$(awk "BEGIN {printf \"%.1f\", (${CACHED_LAYERS} / ${TOTAL_STEPS}) * 100}")
                echo "   Cache hit rate: ${CACHE_RATE}%"
            fi
        fi
        echo ""
    fi
fi

# Handle --dev-image option (build local-dev from existing dev image)
if [[ -n "${DEV_IMAGE_INPUT:-}" ]]; then
    # Validate that the dev image is not already a local-dev image
    if [[ "$DEV_IMAGE_INPUT" == *"-local-dev" ]]; then
        echo "ERROR: Cannot use local-dev image as dev image input: '$DEV_IMAGE_INPUT'"
        exit 1
    fi

    # Build tag arguments - always add -local-dev suffix for --dev-image
    # Generate local-dev tag from input image
    if [[ "$DEV_IMAGE_INPUT" == *:* ]]; then
        LOCAL_DEV_TAG="--tag ${DEV_IMAGE_INPUT}-local-dev"
    else
        LOCAL_DEV_TAG="--tag ${DEV_IMAGE_INPUT}:latest-local-dev"
    fi

    build_local_dev_with_header "$DEV_IMAGE_INPUT" "$LOCAL_DEV_TAG" "Successfully built local-dev image: ${LOCAL_DEV_TAG#--tag }" "Building Local-Dev Image"
elif [[ "${LOCAL_DEV_BUILD:-}" == "true" ]]; then
    # Use the first tag name (TAG) if available, otherwise use latest
    if [[ -n "$TAG" ]]; then
        DEV_IMAGE=$(echo "$TAG" | sed 's/--tag //' | sed 's/-local-dev$//')
    else
        DEV_IMAGE="dynamo:latest-${FRAMEWORK,,}"
    fi

    # Build local-dev tags from existing tags
    LOCAL_DEV_TAGS=""
    if [[ -n "$TAG" ]]; then
        # Extract tag name, remove any existing -local-dev suffix, then add -local-dev
        TAG_NAME=$(echo "$TAG" | sed 's/--tag //' | sed 's/-local-dev$//')
        LOCAL_DEV_TAGS+=" --tag ${TAG_NAME}-local-dev"
    fi

    if [[ -n "$LATEST_TAG" ]]; then
        # Extract tag name, remove any existing -local-dev suffix, then add -local-dev
        LATEST_TAG_NAME=$(echo "$LATEST_TAG" | sed 's/--tag //' | sed 's/-local-dev$//')
        LOCAL_DEV_TAGS+=" --tag ${LATEST_TAG_NAME}-local-dev"
    fi

    build_local_dev_with_header "$DEV_IMAGE" "$LOCAL_DEV_TAGS" "Successfully built local-dev images" "Starting Build 3: Local-Dev Image"
fi


{ set +x; } 2>/dev/null

# Final build summary
echo ""
echo "=========================================="
echo "🎉 BUILD COMPLETED SUCCESSFULLY"
echo "=========================================="
echo ""
echo "📦 Built Images:"
if [ -n "$TAG" ]; then
    FINAL_TAG=$(echo "$TAG" | sed 's/--tag //')
    echo "   ${FINAL_TAG}"
    
    # Show image size if possible
    if command -v docker &> /dev/null && [ -z "$RUN_PREFIX" ]; then
        IMAGE_SIZE=$(docker image inspect "${FINAL_TAG}" --format='{{.Size}}' 2>/dev/null || echo "")
        if [ -n "$IMAGE_SIZE" ]; then
            IMAGE_SIZE_HUMAN=$(numfmt --to=iec-i --suffix=B ${IMAGE_SIZE} 2>/dev/null || echo "${IMAGE_SIZE} bytes")
            echo "   Size: ${IMAGE_SIZE_HUMAN}"
        fi
    fi
fi
if [ -n "$LATEST_TAG" ]; then
    echo "   $(echo "$LATEST_TAG" | sed 's/--tag //')"
fi
echo ""

# Show build logs location if they exist
if [ -d "${BUILD_LOG_DIR:-}" ]; then
    echo "📝 Build Logs:"
    find "${BUILD_LOG_DIR}" -name "*.log" -type f 2>/dev/null | while read -r log_file; do
        LOG_SIZE=$(du -h "${log_file}" | cut -f1)
        LOG_LINES=$(wc -l < "${log_file}")
        echo "   $(basename ${log_file}): ${LOG_SIZE} (${LOG_LINES} lines)"
    done
    echo ""
    
    # Generate BuildKit reports if Python is available
    if command -v python3 &>/dev/null; then
        BUILDKIT_PARSER="${SOURCE_DIR}/../.github/scripts/parse_buildkit_output.py"
        if [ -f "${BUILDKIT_PARSER}" ]; then
            echo "📊 Generating BuildKit Reports..."
            echo ""
            
            # Create reports directory
            REPORTS_DIR="${BUILD_LOG_DIR}/reports"
            mkdir -p "${REPORTS_DIR}"
            
            # Process each build log
            for log_file in $(find "${BUILD_LOG_DIR}" -name "*.log" -type f 2>/dev/null); do
                LOG_BASENAME=$(basename "${log_file}" .log)
                REPORT_FILE="${REPORTS_DIR}/${LOG_BASENAME}-report.txt"
                JSON_FILE="${REPORTS_DIR}/${LOG_BASENAME}-data.json"
                
                echo "   Processing: $(basename ${log_file})"
                
                # Parse and generate report
                if python3 "${BUILDKIT_PARSER}" "${log_file}" "${JSON_FILE}" "${REPORT_FILE}" 2>/dev/null; then
                    echo "   ✅ Report: $(basename ${REPORT_FILE})"
                    echo "   ✅ Data:   $(basename ${JSON_FILE})"
                else
                    echo "   ⚠️  Failed to generate report"
                fi
            done
            
            echo ""
            echo "📊 Build Reports generated in: ${REPORTS_DIR}"
            echo ""
            
            # Display the most relevant report (framework build if available, otherwise latest)
            MAIN_REPORT=""
            if [ -n "${FRAMEWORK:-}" ]; then
                MAIN_REPORT=$(find "${REPORTS_DIR}" -name "*framework-${FRAMEWORK,,}*-report.txt" | head -n 1)
            fi
            if [ -z "${MAIN_REPORT}" ]; then
                MAIN_REPORT=$(find "${REPORTS_DIR}" -name "*-report.txt" | tail -n 1)
            fi
            
            if [ -n "${MAIN_REPORT}" ] && [ -f "${MAIN_REPORT}" ]; then
                echo ""
                cat "${MAIN_REPORT}"
                echo ""
            fi
        else
            echo "   ⚠️  BuildKit parser not found at: ${BUILDKIT_PARSER}"
        fi
    else
        echo "   ⚠️  Python3 not available for BuildKit report generation"
    fi
fi

echo "✅ All builds completed successfully!"
echo "=========================================="
echo ""
