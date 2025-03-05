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

TENSORRTLLM_BACKEND_REPO_TAG=
TENSORRTLLM_BACKEND_REBUILD=
DYNEMO_LLM_PATH=
GIT_TOKEN=
GIT_REPO=

get_options() {
    while :; do
        case $1 in
        -h | -\? | --help)
            show_help
            exit
            ;;
    --tensorrtllm-backend-repo-tag)
            if [ "$2" ]; then
                TENSORRTLLM_BACKEND_REPO_TAG=$2
                shift
            else
		missing_requirement $1
            fi
            ;;
    --tensorrtllm-backend-rebuild)
            if [ "$2" ]; then
                TENSORRTLLM_BACKEND_REBUILD=$2
                shift
            else
		missing_requirement $1
            fi
            ;;
    --dynemo-llm-path)
            if [ "$2" ]; then
                DYNEMO_LLM_PATH=$2
                shift
            else
		missing_requirement $1
            fi
            ;;
    --git-token)
            if [ "$2" ]; then
                GIT_TOKEN=$2
                shift
            else
		missing_requirement $1
            fi
            ;;
    --git-repo)
            if [ "$2" ]; then
                GIT_REPO=$2
                shift
            else
		missing_requirement $1
            fi
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
}

show_options() {
    echo ""
    echo "Getting TENSORRTLLM Backend Repo"
    echo ""
    echo "   Tensorrtllm Backend Repo Tag: '${TENSORRTLLM_BACKEND_REPO_TAG}'"
    echo "   Tensorrtllm Backend Rebuild: '${TENSORRTLLM_BACKEND_REBUILD}'"
    echo ""
}


show_help() {
    echo "usage: clone_tensorrtllm.sh"
    echo "  [--tensorrtllm-backend-repo-tag commit]"
    echo "  [--tensorrtllm-backend-rebuild whether to rebuild backend]"
    echo "  [--git-token git-token]"
    echo "  [--git-repo git-repo]"
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

if [ -z ${GIT_REPO} ]; then
       GIT_REPO="github.com/triton-inference-server/tensorrtllm_backend"
fi

if [ ! -z ${GIT_TOKEN} ]; then
    GIT_REPO="https://oauth2:${GIT_TOKEN}@${GIT_REPO}"
else
    GIT_REPO="https://${GIT_REPO}"
fi

show_options

git clone ${GIT_REPO}
cd tensorrtllm_backend
git checkout ${TENSORRTLLM_BACKEND_REPO_TAG}
git submodule update --init --recursive
git lfs install
git lfs pull

if [ ! -z ${TENSORRTLLM_BACKEND_REBUILD} ]; then
    # Install cmake
    apt update -q=2 \
	    && apt install -y gpg wget \
        && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - |  tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
	    && . /etc/os-release \
	    && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $UBUNTU_CODENAME main" | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
	    && apt-get update -q=2 \
	    && apt-get install -y --no-install-recommends cmake=3.28.3* cmake-data=3.28.3* \
        && cmake --version

    # Install rapidjson
    apt install -y rapidjson-dev

    # Build the backend
    (cd inflight_batcher_llm/src \
        && cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DUSE_CXX11_ABI=1 -DDYNEMO_LLM_PATH=$DYNEMO_LLM_PATH .. \
        && make install \
        && cp libdynemo_tensorrtllm.so /opt/tritonserver/backends/tensorrtllm/ \
        && cp trtllmExecutorWorker /opt/tritonserver/backends/tensorrtllm/ \
    )
fi
cd ..
mv tensorrtllm_backend /
