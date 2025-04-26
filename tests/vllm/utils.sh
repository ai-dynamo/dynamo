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


wait_for_server() {
    local PREV_SET_X=""
    if [[ $- == *x* ]]; then
        PREV_SET_X="set -x"
        set +x  # Disable set -x
    fi

    local host_port=$1
    local model=$2
    local max_attempts=${3:-20}  # Default to 20 attempts (5min with 15s delay)
    local attempt=1

    local host=$(echo $host_port | cut -d ':' -f 1)
    local port=$(echo $host_port | cut -d ':' -f 2)

    echo "Waiting for server to be ready at $host_port..."
    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt"
        echo "Trying to connect to $host_port with model $model"
        if curl -s -X POST http://$host_port/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{
                "model": '$model',
                "messages": [
                    {
                        "role": "user",
                        "content": "test"
                    }
                ],
                "stream": false,
                "max_tokens": 5
            }' | grep -q "content"; then
            echo "Server is ready!"

            if [[ -n "$PREV_SET_X" ]]; then
                $PREV_SET_X
            fi

            return 0
        fi

        sleep 15
        attempt=$((attempt + 1))
    done

    echo "Server failed to start after $max_attempts attempts"

    if [[ -n "$PREV_SET_X" ]]; then
        $PREV_SET_X
    fi

    return 1
}

kill_tree() {
    local parent=$1
    local children=$(ps -o pid= --ppid $parent)
    for child in $children; do
        kill_tree $child
    done
    echo "Killing process $parent"
    kill -9 $parent
}


setup_cleanup_trap() {
  # Sets up a cleanup trap to ensure that all subprocesses are terminated when the script exits.
  # It creates a new process group for this script so that all spawned child processes can be killed together.
  # NOTE: If SIGINT has been inherited as ignored from the parent process, we explicitly reset it with 'trap - SIGINT'
  # to ensure the SIGINT handler registered here works as intended.
  # The trap is configured to catch EXIT, HUP, INT, and TERM signals. When triggered, it resets the traps,
  # kills the entire process group, waits for all child processes to exit, and logs a cleanup completion message.
  if [ -z "$CLEANUP_TRAP_SETUP" ]; then
    trap - SIGINT
    trap 'echo "Caught exit signal. Killing all subprocesses..."; kill_tree $(pgrep circusd); kill -int $(jobs -p) 2>/dev/null || true; wait; echo "Cleanup complete."' EXIT HUP INT TERM

    export CLEANUP_TRAP_SETUP=1
    echo "Process cleanup trap has been set up"
  fi
}

# Function to extract the first node from SLURM_NODELIST
# This handles various SLURM formats like: node[1-4], node1,node2, gpu-node[01-04,07,09-12]
get_master_node() {
  if [ -n "$SLURM_NODELIST" ]; then
    # If SLURM_NODELIST contains brackets (node ranges)
    if [[ "$SLURM_NODELIST" == *"["* ]]; then
      # Extract the prefix before the bracket
      prefix=$(echo $SLURM_NODELIST | sed 's/\[.*//')
      # Extract the first item in the range
      first_item=$(echo $SLURM_NODELIST | sed 's/.*\[\([^,]*\).*/\1/' | sed 's/-.*$//')
      echo "$prefix$first_item"
    else
      # For comma-separated lists without brackets, take the first entry
      echo $(echo $SLURM_NODELIST | cut -d',' -f1)
    fi
  else
    # Fallback if SLURM_NODELIST is not set
    hostname
  fi
}

wait_for_etcd() {
  local ETCD_ENDPOINTS="$1"
  echo "Waiting for etcd to be available at $ETCD_ENDPOINTS..."
  timeout 1m bash -c "
    until curl -s $ETCD_ENDPOINTS/health > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


wait_for_nats() {
  local NATS_URL="$1"
  echo "Waiting for NATS to be available at $NATS_URL..."
  timeout 1m bash -c "
    until curl -s $NATS_URL/healthz > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}
