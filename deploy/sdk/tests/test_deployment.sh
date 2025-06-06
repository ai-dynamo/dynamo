#!/bin/bash
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


set -euo pipefail

export DYNAMO_CLOUD="${DYNAMO_CLOUD:-http://dynamo-cloud}"
export DYNAMO_IMAGE="${DYNAMO_IMAGE:-dynamo-base:latest}"
export DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-ci-hw}"

cd /workspace/examples/hello_world

# Step.1: Login to  dynamo cloud
dynamo cloud login $DYNAMO_CLOUD

# Step.2:  build a dynamo nim with framework-less base
DYNAMO_TAG=$(dynamo build hello_world:Frontend | grep "Successfully built" | awk -F"\"" '{ print $2 }')

# Step.3: Deploy!
echo $DYNAMO_TAG
dynamo deployment create $DYNAMO_TAG --no-wait -n $DEPLOYMENT_NAME
