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

RUN_PREFIX=

# Frameworks
#
# Each framework has a corresponding base image.  Additional
# dependencies are specified in the /container/deps folder and
# installed within framework specific sections of the Dockerfile.

declare -A FRAMEWORKS=(["VLLM"]=1 ["TRTLLM"]=2 ["NONE"]=3 ["SGLANG"]=4)

DEFAULT_FRAMEWORK=VLLM

SOURCE_DIR=$(dirname "$(readlink -f "$0")")

IMAGE=
TARGET="dev"
HF_CACHE=
DEFAULT_HF_CACHE=${SOURCE_DIR}/.cache/huggingface
GPUS="all"
PRIVILEGED=
VOLUME_MOUNTS=
MOUNT_WORKSPACE=
DEV_MODE=
ENVIRONMENT_VARIABLES=
REMAINING_ARGS=
INTERACTIVE=
USE_NIXL_GDS=
RUNTIME=nvidia
WORKDIR=/workspace

get_options() {
    while :; do
        case $1 in
        -h | -\? | --help)
            show_help
            exit
            ;;
        --framework)
            if [ "$2" ]; then
                FRAMEWORK=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --image)
            if [ "$2" ]; then
                IMAGE=$2
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
        --name)
            if [ "$2" ]; then
                NAME=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --hf-cache)
            if [ "$2" ]; then
                HF_CACHE=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;

        --gpus)
            if [ "$2" ]; then
                GPUS=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --runtime)
            if [ "$2" ]; then
                RUNTIME=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --entrypoint)
            if [ "$2" ]; then
                ENTRYPOINT=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --workdir)
            if [ "$2" ]; then
                WORKDIR="$2"
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --privileged)
            if [ "$2" ]; then
                PRIVILEGED=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        --rm)
            if [ "$2" ]; then
                RM=$2
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        -v)
            if [ "$2" ]; then
                VOLUME_MOUNTS+=" -v $2 "
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        -e)
            if [ "$2" ]; then
                ENVIRONMENT_VARIABLES+=" -e $2 "
                shift
            else
                missing_requirement "$1"
            fi
            ;;
        -it)
            INTERACTIVE=" -it "
            ;;
        --mount-workspace)
            MOUNT_WORKSPACE=TRUE
            ;;

        --use-nixl-gds)
            USE_NIXL_GDS=TRUE
            ;;
        --dry-run)
            RUN_PREFIX="echo"
            echo ""
            echo "=============================="
            echo "DRY RUN: COMMANDS PRINTED ONLY"
            echo "=============================="
            echo ""
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
    fi

    if [ -z "$IMAGE" ]; then
        IMAGE="dynamo:latest-${FRAMEWORK,,}"
        if [ -n "${TARGET}" ]; then
            IMAGE="${IMAGE}-${TARGET}"
        fi
    fi

    if [[ ${GPUS^^} == "NONE" ]]; then
        GPU_STRING=""
    else
        GPU_STRING="--gpus ${GPUS}"
    fi

    if [[ ${NAME^^} == "" ]]; then
        NAME_STRING=""
    else
        NAME_STRING="--name ${NAME}"
    fi

    if [[ ${ENTRYPOINT^^} == "" ]]; then
        ENTRYPOINT_STRING=""
    else
        ENTRYPOINT_STRING="--entrypoint ${ENTRYPOINT}"
    fi

    if [[ ${HF_CACHE^^} == "NONE" ]]; then
        HF_CACHE=
    fi

    # HF_CACHE mounting will be handled in workspace section

    if [ -z "${PRIVILEGED}" ]; then
        PRIVILEGED="FALSE"
    fi

    if [ -z "${RM}" ]; then
        RM="TRUE"
    fi

    # Initialize PRIVILEGED_STRING
    PRIVILEGED_STRING=""
    if [[ ${PRIVILEGED^^} != "FALSE" ]]; then
        PRIVILEGED_STRING="--privileged"
    fi

    if [[ ${RM^^} == "FALSE" ]]; then
        RM_STRING=""
    else
        RM_STRING=" --rm "
    fi

    if [ -n "$USE_NIXL_GDS" ]; then
        VOLUME_MOUNTS+=" -v /run/udev:/run/udev:ro "
        NIXL_GDS_CAPS="--cap-add=IPC_LOCK"
        # NOTE(jthomson04): In the KVBM disk pools, we currently allocate our files in /tmp.
        # For some arcane reason, GDS requires that /tmp be mounted.
        # This is already handled for us if we set --mount-workspace
        # If we aren't mounting our workspace but need GDS, we need to mount /tmp.
        if [ -z "$MOUNT_WORKSPACE" ]; then
            VOLUME_MOUNTS+=" -v /tmp:/tmp "
        fi
    else
        NIXL_GDS_CAPS=""
    fi
    if [[ "$GPUS" == "none" || "$GPUS" == "NONE" ]]; then
            RUNTIME=""
    fi

    # Auto-enable DEV_MODE for vllm dev images
    # TODO(keivenc): Currently only Dockerfile.vllm has proper permissions to run as ubuntu user.
    # Other Dockerfiles (trtllm, sglang, etc.) still require root access.
    if [[ "$IMAGE" == *"-vllm-dev" ]]; then
        DEV_MODE=TRUE
        MOUNT_WORKSPACE=TRUE
        # Interactive mode is implied when MOUNT_WORKSPACE is TRUE
    fi

    REMAINING_ARGS=("$@")
}

show_help() {
    echo "usage: run.sh"
    echo "  [--image image]"
    echo "  [--framework framework one of ${!FRAMEWORKS[*]}]"
    echo "  [--target target stage to use, default is 'dev']"
    echo "  [--name name for launched container, default NONE]"
    echo "  [--privileged whether to launch in privileged mode, default FALSE unless mounting workspace]"
    echo "  [--dry-run print docker commands without running]"
    echo "  [--hf-cache directory to volume mount as the hf cache, default is NONE unless mounting workspace]"
    echo "  [--gpus gpus to enable, default is 'all', 'none' disables gpu support]"
    echo "  [--use-nixl-gds add volume mounts and capabilities needed for NVIDIA GPUDirect Storage]"
    echo "  [-v add volume mount]"
    echo "  [-e add environment variable]"
    echo "  [--mount-workspace set up for local development]"
    echo "  [-- stop processing and pass remaining args as command to docker run]"
    echo "  [--workdir set the working directory inside the container]"
    echo "  [--runtime add runtime variables]"
    echo "  [--entrypoint override container entrypoint]"
    echo "  [-h, --help show this help]"
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

# Process workspace mounting after auto-detection
if [ -n "$MOUNT_WORKSPACE" ]; then
    HOME_PATH="/home/ubuntu"

    # Common workspace setup
    VOLUME_MOUNTS+=" -v $(dirname "${SOURCE_DIR}"):/workspace "
    VOLUME_MOUNTS+=" -v /tmp:/tmp "
    VOLUME_MOUNTS+=" -v /mnt/:/mnt "
    WORKDIR=/workspace
    INTERACTIVE=" -it "

    # Set default HF_CACHE if not specified
    if [ -z "$HF_CACHE" ]; then
        HF_CACHE=$DEFAULT_HF_CACHE
    fi

    # Environment variables for all workspace modes
    ENVIRONMENT_VARIABLES+=" -e HF_TOKEN"
    ENVIRONMENT_VARIABLES+=" -e GITHUB_TOKEN"
    ENVIRONMENT_VARIABLES+=" -e HOME=$HOME_PATH"

    # Mount HF_CACHE to user's home cache directory
    if [ -n "$HF_CACHE" ]; then
        mkdir -p "$HF_CACHE"
        VOLUME_MOUNTS+=" -v $HF_CACHE:$HOME_PATH/.cache/huggingface"
    fi

    if [ -n "$DEV_MODE" ]; then
        # Dev Container-specific setup - the Dockerfile handles UID/GID mapping via build args
        # This currently only works with Dockerfile.vllm which has proper ubuntu user setup.
        echo "Dev Container mode enabled - using ubuntu user with host UID/GID"
        # Use ubuntu user (with correct UID/GID baked into image)
        PRIVILEGED_STRING+=" --user ubuntu"
    else
        # Standard workspace mode - enable privileged mode
        # TODO(keivenc): Security risk, remove soon. Dockerfiles (trtllm, sglang) still need to run as root.
        if [ -z "${PRIVILEGED}" ]; then
            PRIVILEGED_STRING="--privileged"
        fi
    fi
fi

# RUN the image
if [ -z "$RUN_PREFIX" ]; then
    set -x
fi

${RUN_PREFIX} docker run \
    ${GPU_STRING} \
    ${INTERACTIVE} \
    ${RM_STRING} \
    --network host \
    ${RUNTIME:+--runtime "$RUNTIME"} \
    --shm-size=10G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ulimit nofile=65536:65536 \
    ${ENVIRONMENT_VARIABLES} \
    ${VOLUME_MOUNTS} \
    -w "$WORKDIR" \
    --cap-add CAP_SYS_PTRACE \
    ${NIXL_GDS_CAPS} \
    --ipc host \
    ${PRIVILEGED_STRING} \
    ${NAME_STRING} \
    ${ENTRYPOINT_STRING} \
    ${IMAGE} \
    "${REMAINING_ARGS[@]}"

{ set +x; } 2>/dev/null
