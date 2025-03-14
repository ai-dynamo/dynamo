#!/bin/bash

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
# ACTION REQUIRED: Export your Kubernetes namespace as $KUBE_NS
# and update the ns.yaml file with the same value
export KUBE_NS=$KUBE_NS
kubectl apply -f testing/ns.yaml

# Export your ngc api key

curl -X POST \
     -H "Content-Type: application/json" \
     https://${NAMESPACE}.dev.aire.nvidia.com/api/v1/clusters \
     -d '{
       "name": "default",
       "description": "Default cluster",
       "kube_config": ""
     }' | jq '.'

# check out ui at https://${NAMESPACE}.dev.aire.nvidia.com
