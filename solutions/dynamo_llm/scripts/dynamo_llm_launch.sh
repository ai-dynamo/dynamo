#!/bin/bash -e
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

RUN_PREFIX=

# Frameworks
#
# Each framework has a corresponding base image.  Additional
# dependencies are specified in the /container/deps folder and
# installed within framework specific sections of the Dockerfile.

declare -A FRAMEWORKS=(["TENSORRTLLM"]=2 ["VLLM"]=3 ["VLLM_NIXL"]=4)
DEFAULT_FRAMEWORK=VLLM

SOURCE_DIR=$(dirname "$(readlink -f "$0")")

MOUNT_WORKSPACE=
REMAINING_ARGS=
LEADER=
FOLLOWING=

get_options() {
    while :; do
        case $1 in
        -h | -\? | --help)
            show_help
            exit
            ;;

	--following)
	    if [ "$2" ]; then
		FOLLOWING=$2
		NATS_SERVER="-e NATS_SERVER=nats://${FOLLOWING}:4222"
		ETCD_ENDPOINTS="-e ETCD_ENPOINTS=http://${FOLLOWING}:2379"
                shift
            else
		missing_requirement $1
            fi
            ;;
	--leader)
	    LEADER=TRUE
	    ;;
	--mount-workspace)
	    MOUNT_WORKSPACE=TRUE
	    ;;
        --dry-run)
	    DRY_RUN=" --dry-run "
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
	    error 'ERROR: Unknown option: ' $1
            ;;
	 ?*)
	    error 'ERROR: Unknown option: ' $1
            ;;
        *)
            break
            ;;
        esac

        shift
    done

    if [ -z "$LEADER" ] && [ -z "$FOLLOWING" ]; then
	LEADER=TRUE
    fi

    if [ ! -z "$LEADER" ] && [ ! -z "$FOLLOWING" ]; then
       error "Need to follow or lead, not both";
    fi


    REMAINING_ARGS=("$@")
}

show_help() {
    echo "usage: run.sh"
    echo "  [--mount-workspace set up for local development]"
    echo "  [--leader run infrastructure services]"
    echo "  [--following <IP> attach to a leader]"
    ${SOURCE_DIR}/../../../container/run.sh -it --mount-workspace ${NATS_SERVER} ${ETCD_ENDPOINTS} -- dynamo-llm --help ;
    exit 0
}

missing_requirement() {
    error "ERROR: $1 requires an argument."
}

error() {
    printf '%s %s\n' "$1" "$2" >&2
    exit 1
}

GLOO_SOCKET_IFNAME=" -e GLOO_SOCKET_IFNAME=lo "


get_options "$@"

# RUN the image


if [ ! -z "$LEADER" ]; then

    if [ -z "$RUN_PREFIX" ]; then
	set -x
    fi

    docker compose -f ${SOURCE_DIR}/../../../deploy/docker-compose.yml down
    docker compose -f ${SOURCE_DIR}/../../../deploy/docker-compose.yml up -d
fi;

{ set +x; } 2>/dev/null


if [ ! -z "$MOUNT_WORKSPACE" ]; then
    if [ -z "$RUN_PREFIX" ]; then
	set -x
    fi

    ${SOURCE_DIR}/../../../container/run.sh -it --mount-workspace ${NATS_SERVER} ${ETCD_ENDPOINTS} ${GLOO_SOCKET_IFNAME} -- dynamo-llm $DRY_RUN "${REMAINING_ARGS[@]}" ;
else

    if [ -z "$RUN_PREFIX" ]; then
	set -x
    fi

    ${SOURCE_DIR}/../../../container/run.sh -it ${NATS_SERVER} ${ETCD_ENDPOINTS} ${GLOO_SOCKET_IFNAME} -- dynamo-llm $DRY_RUN "${REMAINING_ARGS[@]}" ;
fi;

{ set +x; } 2>/dev/null

if [ ! -z "$LEADER" ]; then

    if [ -z "$RUN_PREFIX" ]; then
	set -x
    fi

    docker compose -f ${SOURCE_DIR}/../../../deploy/docker-compose.yml down
fi;

