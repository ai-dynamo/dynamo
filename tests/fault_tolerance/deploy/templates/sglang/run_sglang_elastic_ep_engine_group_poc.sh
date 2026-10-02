#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Attach the running EP1 SGLang DGD worker to an Engine Group, request EP2
# through the Kubernetes Scale subresource, and verify committed serving.

set -euo pipefail

NS="${1:-default}"
DEPLOYMENT_NAME="${2:-sglang-elastic-ep-poc}"
GROUP_NAME="${3:-sglang-elastic-ep-poc}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V2-Lite}"

worker_pod() {
  kubectl get pods -n "$NS" \
    -l "nvidia.com/dynamo-component=SGLangDecodeWorker,nvidia.com/dynamo-graph-deployment-name=$DEPLOYMENT_NAME" \
    --field-selector=status.phase=Running \
    --sort-by=.metadata.name \
    -o jsonpath='{.items[0].metadata.name}'
}

frontend_pod() {
  kubectl get pods -n "$NS" \
    -l "nvidia.com/dynamo-component=Frontend,nvidia.com/dynamo-graph-deployment-name=$DEPLOYMENT_NAME" \
    --field-selector=status.phase=Running \
    --sort-by=.metadata.name \
    -o jsonpath='{.items[0].metadata.name}'
}

echo "Waiting for the EP1 primary and frontend"
kubectl wait -n "$NS" --for=condition=Ready pod \
  -l "nvidia.com/dynamo-component=SGLangDecodeWorker,nvidia.com/dynamo-graph-deployment-name=$DEPLOYMENT_NAME" \
  --timeout=1200s
kubectl wait -n "$NS" --for=condition=Ready pod \
  -l "nvidia.com/dynamo-component=Frontend,nvidia.com/dynamo-graph-deployment-name=$DEPLOYMENT_NAME" \
  --timeout=1200s

PRIMARY="$(worker_pod)"
FRONTEND_IP="$(kubectl get pod -n "$NS" "$(frontend_pod)" -o jsonpath='{.status.podIP}')"

echo "Binding $PRIMARY to Engine Group $GROUP_NAME"
kubectl label pod -n "$NS" "$PRIMARY" --overwrite \
  "nvidia.com/dynamo-engine-group=$GROUP_NAME" \
  "nvidia.com/dynamo-engine-group-replica=replica-0" \
  "nvidia.com/dynamo-engine-group-slot=slot-0" \
  "nvidia.com/dynamo-engine-group-role=primary" \
  "nvidia.com/dynamo-scale-representative=true"

kubectl apply -n "$NS" -f - <<EOF
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentEngineGroup
metadata:
  name: $GROUP_NAME
  labels:
    nvidia.com/dynamo-engine-group-runtime: sglang-elastic-ep
  annotations:
    nvidia.com/dynamo-engine-group-control-port: "9090"
    nvidia.com/dynamo-engine-group-verify-url: http://$FRONTEND_IP:8000/v1/completions
    nvidia.com/dynamo-engine-group-verify-model: $MODEL
spec:
  replicas: 1
  policy:
    minReplicas: 1
    maxReplicas: 2
EOF

echo "Waiting for the controller to adopt the EP1 baseline"
for _ in $(seq 1 120); do
  active="$(kubectl get dynamographdeploymentenginegroup -n "$NS" "$GROUP_NAME" -o jsonpath='{.status.activeNativeMemberCount}' 2>/dev/null || true)"
  if [[ "$active" == "1" ]]; then
    break
  fi
  sleep 2
done
[[ "${active:-}" == "1" ]] || { echo "Engine Group did not observe EP1" >&2; exit 1; }

echo "Requesting EP2 through the Scale subresource"
kubectl scale -n "$NS" dynamographdeploymentenginegroup/"$GROUP_NAME" --replicas=2

for _ in $(seq 1 600); do
  active="$(kubectl get dynamographdeploymentenginegroup -n "$NS" "$GROUP_NAME" -o jsonpath='{.status.activeNativeMemberCount}' 2>/dev/null || true)"
  available="$(kubectl get dynamographdeploymentenginegroup -n "$NS" "$GROUP_NAME" -o jsonpath='{.status.availableReplicas}' 2>/dev/null || true)"
  reached="$(kubectl get dynamographdeploymentenginegroup -n "$NS" "$GROUP_NAME" -o jsonpath='{.status.conditions[?(@.type=="TargetReached")].status}' 2>/dev/null || true)"
  if [[ "$active" == "2" && "$available" == "2" && "$reached" == "True" ]]; then
    break
  fi
  sleep 2
done
if [[ "${active:-}" != "2" || "${available:-}" != "2" || "${reached:-}" != "True" ]]; then
  kubectl get dynamographdeploymentenginegroup -n "$NS" "$GROUP_NAME" -o yaml
  kubectl get pods -n "$NS" -l "nvidia.com/dynamo-engine-group=$GROUP_NAME" -o wide
  echo "Engine Group did not converge to EP2" >&2
  exit 1
fi

echo "EP1 -> EP2 completed"
kubectl get dynamographdeploymentenginegroup -n "$NS" "$GROUP_NAME"
kubectl get pods -n "$NS" -l "nvidia.com/dynamo-engine-group=$GROUP_NAME" -o wide
