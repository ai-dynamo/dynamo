#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Deploy the Dynamo SGLang rollout graph used by the Slime external example.

set -euo pipefail

: "${KUBE_CONTEXT:?Set KUBE_CONTEXT to the target Kubernetes context}"
: "${NAMESPACE:?Set NAMESPACE to the target Kubernetes namespace}"
: "${DYNAMO_IMAGE:?Set DYNAMO_IMAGE to an image containing the SGLang /generate facade}"
: "${MODEL_PATH:=Qwen/Qwen3-0.6B}"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
export DYNAMO_IMAGE MODEL_PATH

envsubst '${DYNAMO_IMAGE} ${MODEL_PATH}' <"${SCRIPT_DIR}/dynamo.yaml" |
    kubectl --context="${KUBE_CONTEXT}" --namespace="${NAMESPACE}" apply -f -
