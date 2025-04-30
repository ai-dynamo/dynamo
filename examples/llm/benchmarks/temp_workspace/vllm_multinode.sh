#/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# start nats and etcd
if [[ -z "${HEAD_NODE_IP}" ]]; then
    nats-server -js &
    etcd --advertise-client-urls http://0.0.0.0:2379 --listen-client-urls http://0.0.0.0:2379 &
    HEAD_NODE_IP=`hostname -i`
else
    export NATS_SERVER=nats://${HEAD_NODE_IP}:4222
    export ETCD_ENDPOINTS=${HEAD_NODE_IP}:2379
fi

# start ray cluster
if [[ -z "${RAY_LEADER_NODE_IP}" ]]; then
    ray start --head --port=6379 --disable-usage-stats
    RAY_LEADER_NODE_IP=`hostname -i`
else
    ray start --address=${RAY_LEADER_NODE_IP}:6379
fi

echo "HEAD_NODE_IP=${HEAD_NODE_IP} RAY_LEADER_NODE_IP=${RAY_LEADER_NODE_IP=} source ${BASH_SOURCE[0]}"
