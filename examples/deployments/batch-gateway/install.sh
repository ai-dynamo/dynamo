#!/bin/bash
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

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly BATCH_GATEWAY_CHART="oci://ghcr.io/llm-d/charts/batch-gateway"
readonly BATCH_GATEWAY_VERSION="0.3.0"
readonly ASYNC_CHART="oci://ghcr.io/llm-d/charts/llm-d-async"
readonly ASYNC_VERSION="v0.9.0"
readonly MODEL="Qwen/Qwen3-0.6B"
readonly DGD_NAME="qwen3-0-6b-batch"
readonly DEFAULT_PROMETHEUS_URL="http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090"

NAMESPACE="dynamo-batch-example"
PROMETHEUS_URL="${PROMETHEUS_URL:-${DEFAULT_PROMETHEUS_URL}}"
DRY_RUN=false

usage() {
    cat <<'EOF'
Install the experimental per-DGD batch example.

Usage:
  ./install.sh [--namespace NAMESPACE] [--dry-run]

Options:
  --namespace NAMESPACE  Target namespace (default: dynamo-batch-example).
  --dry-run              Print the mutating commands without running them.
  -h, --help             Show this help.

Environment:
  PROMETHEUS_URL         Prometheus base URL visible from the cluster.
EOF
}

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" == "false" ]]; then
        "$@"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --namespace)
            [[ $# -ge 2 ]] || fail "--namespace requires a value"
            NAMESPACE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "unknown option: $1"
            ;;
    esac
done

[[ -n "${NAMESPACE}" ]] || fail "namespace must not be empty"
require_command kubectl
require_command helm

if [[ "${DRY_RUN}" == "false" ]]; then
    kubectl get crd dynamographdeployments.nvidia.com >/dev/null 2>&1 \
        || fail "DynamoGraphDeployment CRD not found; install the Dynamo platform first"

    printf 'Installing in Kubernetes context %s, namespace %s\n' \
        "$(kubectl config current-context)" \
        "${NAMESPACE}"

    if [[ "${PROMETHEUS_URL}" == "${DEFAULT_PROMETHEUS_URL}" ]]; then
        kubectl get service \
            --namespace monitoring \
            prometheus-kube-prometheus-prometheus >/dev/null 2>&1 \
            || fail "default Prometheus service not found; install Dynamo observability or set PROMETHEUS_URL"
    fi

    if ! kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1; then
        run kubectl create namespace "${NAMESPACE}"
    fi
else
    run kubectl create namespace "${NAMESPACE}"
fi

run kubectl apply \
    --namespace "${NAMESPACE}" \
    --filename "${SCRIPT_DIR}/batch-infra.yaml"
run kubectl rollout status \
    --namespace "${NAMESPACE}" \
    statefulset/batch-gateway-valkey \
    --timeout 180s

run kubectl apply \
    --namespace "${NAMESPACE}" \
    --filename "${SCRIPT_DIR}/dynamo.yaml"
run kubectl wait \
    --namespace "${NAMESPACE}" \
    --for condition=Ready \
    "dynamographdeployment/${DGD_NAME}" \
    --timeout 900s

readiness_query="min(dynamo_frontend_model_ready{model=\"${MODEL}\"\\,namespace=\"${NAMESPACE}\"})"

run helm upgrade --install async-dispatch \
    "${ASYNC_CHART}" \
    --version "${ASYNC_VERSION}" \
    --namespace "${NAMESPACE}" \
    --values "${SCRIPT_DIR}/llm-d-async-values.yaml" \
    --set-string "ap.prometheusURL=${PROMETHEUS_URL}" \
    --set-string "ap.redis.gateParams.query=${readiness_query}" \
    --atomic \
    --wait \
    --timeout 5m

run helm upgrade --install batch-gateway \
    "${BATCH_GATEWAY_CHART}" \
    --version "${BATCH_GATEWAY_VERSION}" \
    --namespace "${NAMESPACE}" \
    --values "${SCRIPT_DIR}/batch-gateway-values.yaml" \
    --values "${SCRIPT_DIR}/batch-gateway-async-values.yaml" \
    --atomic \
    --wait \
    --timeout 5m

if [[ "${DRY_RUN}" == "false" ]]; then
    cat <<EOF

Batch Gateway is ready. In one terminal, forward the Batch API:

  kubectl port-forward --namespace ${NAMESPACE} service/batch-gateway-apiserver 8001:8000

Then run the lifecycle check from ${SCRIPT_DIR}:

  python3 run_example.py --base-url http://127.0.0.1:8001
EOF
fi
