#!/usr/bin/env bash

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

set -euo pipefail
trap 'echo "Error at line $LINENO. Exiting."' ERR


MODEL_NAMESPACE=my-model
kubectl create namespace $MODEL_NAMESPACE || true

# Install the Gateway API
GATEWAY_API_VERSION=v1.3.0
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/$GATEWAY_API_VERSION/standard-install.yaml


# Install the Inference Extension CRDs
INFERENCE_EXTENSION_VERSION=v0.5.1
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/$INFERENCE_EXTENSION_VERSION/manifests.yaml -n  $MODEL_NAMESPACE


# Install and upgrade Kgateway (includes CRDs)
KGATEWAY_VERSION=v2.0.3
KGATEWAY_SYSTEM_NAMESPACE=kgateway-system
kubectl create namespace $KGATEWAY_SYSTEM_NAMESPACE || true

helm upgrade -i --create-namespace --namespace $KGATEWAY_SYSTEM_NAMESPACE --version $KGATEWAY_VERSION kgateway-crds oci://cr.kgateway.dev/kgateway-dev/charts/kgateway-crds

helm upgrade -i --namespace $KGATEWAY_SYSTEM_NAMESPACE --version $KGATEWAY_VERSION kgateway oci://cr.kgateway.dev/kgateway-dev/charts/kgateway --set inferenceExtension.enabled=true


# Deploy the Gateway Instance
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/v1.0.0/config/manifests/gateway/kgateway/gateway.yaml -n $MODEL_NAMESPACE
