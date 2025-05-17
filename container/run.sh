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

declare -A FRAMEWORKS=(["VLLM"]=1 ["TENSORRTLLM"]=2)
DEFAULT_FRAMEWORK=VLLM

SOURCE_DIR=$(dirname "$(readlink -f "$0")")

IMAGE=
HF_CACHE=
DEFAULT_HF_CACHE=${SOURCE_DIR}/.cache/huggingface
GPUS="all"
PRIVILEGED=
VOLUME_MOUNTS=
MOUNT_WORKSPACE=
ENVIRONMENT_VARIABLES=
REMAINING_ARGS=
INTERACTIVE=
USE_NIXL_GDS=

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
	--entrypoint)
            if [ "$2" ]; then
                ENTRYPOINT=$2
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

    if [ -n "$MOUNT_WORKSPACE" ]; then
	VOLUME_MOUNTS+=" -v ${SOURCE_DIR}/..:/workspace "
	VOLUME_MOUNTS+=" -v /tmp:/tmp "
	VOLUME_MOUNTS+=" -v /mnt/:/mnt "

	if [ -z "$HF_CACHE" ]; then
	    HF_CACHE=$DEFAULT_HF_CACHE
	fi

	if [ -z "${PRIVILEGED}" ]; then
	    PRIVILEGED="TRUE"
	fi

	ENVIRONMENT_VARIABLES+=" -e HF_TOKEN"

	INTERACTIVE=" -it "
    fi

    if [[ ${HF_CACHE^^} == "NONE" ]]; then
	HF_CACHE=
    fi

    if [ -n "$HF_CACHE" ]; then
	mkdir -p "$HF_CACHE"
	VOLUME_MOUNTS+=" -v $HF_CACHE:/root/.cache/huggingface"
    fi

    if [ -z "${PRIVILEGED}" ]; then
	PRIVILEGED="FALSE"
    fi

    if [ -z "${RM}" ]; then
	RM="TRUE"
    fi

    if [[ ${PRIVILEGED^^} == "FALSE" ]]; then
	PRIVILEGED_STRING=""
    else
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
    else
        NIXL_GDS_CAPS=""
    fi

    REMAINING_ARGS=("$@")
}

show_help() {
    echo "usage: run.sh"
    echo "  [--image image]"
    echo "  [--framework framework one of ${!FRAMEWORKS[*]}]"
    echo "  [--name name for launched container, default NONE] "
    echo "  [--privileged whether to launch in privileged mode, default FALSE unless mounting workspace]"
    echo "  [--dry-run print docker commands without running]"
    echo "  [--hf-cache directory to volume mount as the hf cache, default is NONE unless mounting workspace]"
    echo "  [--gpus gpus to enable, default is 'all', 'none' disables gpu support]"
    echo "  [--use-nixl-gds add volume mounts and capabilities needed for NVIDIA GPUDirect Storage]"
    echo "  [-v add volume mount]"
    echo "  [-e add environment variable]"
    echo "  [--mount-workspace set up for local development]"
    echo "  [-- stop processing and pass remaining args as command to docker run]"
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

# RUN the image

if [ -z "$RUN_PREFIX" ]; then
    set -x
fi

${RUN_PREFIX} docker run \
    ${GPU_STRING} \
    ${INTERACTIVE} \
    ${RM_STRING} \
    --network host \
    --shm-size=10G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ulimit nofile=65536:65536 \
    ${ENVIRONMENT_VARIABLES} \
    ${VOLUME_MOUNTS} \
    -w /workspace \
    --cap-add CAP_SYS_PTRACE \
    ${NIXL_GDS_CAPS} \
    --ipc host \
    ${PRIVILEGED_STRING} \
    ${NAME_STRING} \
    ${ENTRYPOINT_STRING} \
    ${IMAGE} \
    "${REMAINING_ARGS[@]}"

{ set +x; } 2>/dev/null
