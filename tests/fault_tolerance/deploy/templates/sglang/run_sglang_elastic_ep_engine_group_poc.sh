#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Observe the DGD-created Engine Group, request EP3 through its Scale subresource,
# and verify committed serving plus preservation of the initial Grove allocations.

set -euo pipefail

NS="${1:-default}"
DEPLOYMENT_NAME="${2:-sglang-elastic-ep-poc}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V2-Lite}"

# Child discovery proves the DGD lifecycle rather than creating or manually binding a world.
echo "Waiting for the DGD to create its Worker Engine Group"
for _ in $(seq 1 120); do
  GROUP_NAME="$(kubectl get dynamographdeploymentenginegroups -n "$NS" \
    -l "nvidia.com/dynamo-graph-deployment-name=$DEPLOYMENT_NAME,nvidia.com/dynamo-component=Worker" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  [[ -n "$GROUP_NAME" ]] && break
  sleep 2
done
[[ -n "${GROUP_NAME:-}" ]] || { echo "DGD did not create its Engine Group" >&2; exit 1; }
DGD_UID="$(kubectl get dynamographdeployment -n "$NS" "$DEPLOYMENT_NAME" -o jsonpath='{.metadata.uid}')"
OWNER_UID="$(kubectl get dynamographdeploymentenginegroup -n "$NS" "$GROUP_NAME" -o jsonpath='{.metadata.ownerReferences[?(@.controller==true)].uid}')"
[[ "$OWNER_UID" == "$DGD_UID" ]] || { echo "Engine Group is not a DGD-owned child" >&2; exit 1; }

primary_pod() {
  kubectl get pods -n "$NS" \
    -l "nvidia.com/dynamo-engine-group=$GROUP_NAME,grove.io/podclique-pod-index=0" \
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

verify_serving() {
  kubectl exec -i -n "$NS" "$(frontend_pod)" -- python3 - "$MODEL" <<'PY'
import sys
import time

import requests

model = sys.argv[1]
deadline = time.monotonic() + 1200
while time.monotonic() < deadline:
    response = requests.get("http://localhost:8000/v1/models", timeout=10)
    response.raise_for_status()
    if any(item["id"] == model for item in response.json()["data"]):
        break
    time.sleep(2)
else:
    raise RuntimeError("The frontend did not discover the initial engine world")

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={"model": model, "prompt": "The capital of France is", "max_tokens": 8,
          "temperature": 0, "stream": False},
    timeout=180,
)
response.raise_for_status()
assert response.json()["choices"][0]["text"].strip(), "Empty completion"
print(response.text)
PY
}

echo "Waiting for the EP2 initial world and frontend"
# Child creation precedes Grove's asynchronous Pod creation; wait for both initial allocations.
for _ in $(seq 1 120); do
  INITIAL_PARTICIPANT="$(kubectl get pods -n "$NS" \
    -l "nvidia.com/dynamo-engine-group=$GROUP_NAME,grove.io/podclique-pod-index=1" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  INITIAL_FRONTEND="$(frontend_pod 2>/dev/null || true)"
  [[ -n "$INITIAL_PARTICIPANT" && -n "$INITIAL_FRONTEND" ]] && break
  sleep 2
done
[[ -n "${INITIAL_PARTICIPANT:-}" && -n "${INITIAL_FRONTEND:-}" ]] || {
  echo "Grove did not create the initial allocations and frontend" >&2
  exit 1
}
kubectl wait -n "$NS" --for=condition=Ready pod \
  -l "nvidia.com/dynamo-engine-group=$GROUP_NAME" \
  --timeout=1200s
kubectl wait -n "$NS" --for=condition=Ready pod \
  -l "nvidia.com/dynamo-component=Frontend,nvidia.com/dynamo-graph-deployment-name=$DEPLOYMENT_NAME" \
  --timeout=1200s

PRIMARY="$(primary_pod)"
PRIMARY_UID="$(kubectl get pod -n "$NS" "$PRIMARY" -o jsonpath='{.metadata.uid}')"
PARTICIPANT="$(kubectl get pods -n "$NS" \
  -l "nvidia.com/dynamo-engine-group=$GROUP_NAME,grove.io/podclique-pod-index=1" \
  -o jsonpath='{.items[0].metadata.name}')"
PARTICIPANT_UID="$(kubectl get pod -n "$NS" "$PARTICIPANT" -o jsonpath='{.metadata.uid}')"
CLIQUE="$(kubectl get pod -n "$NS" "$PRIMARY" -o jsonpath='{.metadata.labels.grove\.io/podclique}')"
CLIQUE_UID="$(kubectl get podclique -n "$NS" "$CLIQUE" -o jsonpath='{.metadata.uid}')"
WORLD="$(kubectl get podclique -n "$NS" "$CLIQUE" -o jsonpath='{.metadata.ownerReferences[?(@.controller==true)].name}')"

echo "Verifying EP2 baseline inference"
verify_serving

echo "Waiting for the controller to adopt the EP2 baseline"
for _ in $(seq 1 120); do
  active="$(kubectl get dynamographdeploymentenginegroup -n "$NS" "$GROUP_NAME" -o jsonpath='{.status.activeNativeMemberCount}' 2>/dev/null || true)"
  if [[ "$active" == "2" ]]; then
    break
  fi
  sleep 2
done
[[ "${active:-}" == "2" ]] || { echo "Engine Group did not observe EP2" >&2; exit 1; }
BOUND_UID="$(kubectl get dynamographdeploymentenginegroup -n "$NS" "$GROUP_NAME" -o jsonpath='{.metadata.annotations.nvidia\.com/dynamo-engine-group-pod-clique-uid}')"
[[ "$BOUND_UID" == "$CLIQUE_UID" ]] || { echo "DGD did not bind the exact member clique UID" >&2; exit 1; }

echo "Requesting EP3 through the Scale subresource"
kubectl scale -n "$NS" dynamographdeploymentenginegroup/"$GROUP_NAME" --replicas=3

for _ in $(seq 1 600); do
  active="$(kubectl get dynamographdeploymentenginegroup -n "$NS" "$GROUP_NAME" -o jsonpath='{.status.activeNativeMemberCount}' 2>/dev/null || true)"
  available="$(kubectl get dynamographdeploymentenginegroup -n "$NS" "$GROUP_NAME" -o jsonpath='{.status.availableReplicas}' 2>/dev/null || true)"
  reached="$(kubectl get dynamographdeploymentenginegroup -n "$NS" "$GROUP_NAME" -o jsonpath='{.status.conditions[?(@.type=="TargetReached")].status}' 2>/dev/null || true)"
  if [[ "$active" == "3" && "$available" == "3" && "$reached" == "True" ]]; then
    break
  fi
  sleep 2
done
if [[ "${active:-}" != "3" || "${available:-}" != "3" || "${reached:-}" != "True" ]]; then
  kubectl get dynamographdeploymentenginegroup -n "$NS" "$GROUP_NAME" -o yaml
  kubectl get pods -n "$NS" -l "nvidia.com/dynamo-engine-group=$GROUP_NAME" -o wide
  echo "Engine Group did not converge to EP3" >&2
  exit 1
fi

echo "EP2 -> EP3 completed"
# Growth must preserve the primary incarnation and scale capacity, not world count.
CURRENT_PRIMARY_UID="$(kubectl get pod -n "$NS" "$(primary_pod)" -o jsonpath='{.metadata.uid}')"
CURRENT_PARTICIPANT_UID="$(kubectl get pod -n "$NS" "$PARTICIPANT" -o jsonpath='{.metadata.uid}')"
CLIQUE_SIZE="$(kubectl get podclique -n "$NS" "$CLIQUE" -o jsonpath='{.spec.replicas}')"
WORLD_COUNT="$(kubectl get podcliquescalinggroup -n "$NS" "$WORLD" -o jsonpath='{.spec.replicas}')"
[[ "$CURRENT_PRIMARY_UID" == "$PRIMARY_UID" && "$CURRENT_PARTICIPANT_UID" == "$PARTICIPANT_UID" && "$CLIQUE_SIZE" == "3" && "$WORLD_COUNT" == "1" ]] || {
  echo "Grove growth changed an initial Pod or the world-count dimension" >&2
  exit 1
}
echo "Verifying EP3 inference after committed growth"
verify_serving

# A frontend-only spec change triggers a real parent reconcile without changing the world profile.
kubectl patch -n "$NS" dynamographdeployment/"$DEPLOYMENT_NAME" --type=json \
  -p='[{"op":"replace","path":"/spec/components/0/replicas","value":2}]'
GENERATION="$(kubectl get dynamographdeployment -n "$NS" "$DEPLOYMENT_NAME" -o jsonpath='{.metadata.generation}')"
for _ in $(seq 1 120); do
  OBSERVED="$(kubectl get dynamographdeployment -n "$NS" "$DEPLOYMENT_NAME" -o jsonpath='{.status.observedGeneration}')"
  [[ "$OBSERVED" == "$GENERATION" ]] && break
  sleep 2
done
[[ "${OBSERVED:-}" == "$GENERATION" ]] || { echo "Parent did not reconcile its frontend update" >&2; exit 1; }
kubectl wait -n "$NS" --for=condition=Ready dynamographdeployment/"$DEPLOYMENT_NAME" --timeout=120s
[[ "$(kubectl get dynamographdeploymentenginegroup -n "$NS" "$GROUP_NAME" -o jsonpath='{.spec.replicas}')" == "3" ]]
[[ "$(kubectl get podclique -n "$NS" "$CLIQUE" -o jsonpath='{.spec.replicas}')" == "3" ]]
kubectl patch -n "$NS" dynamographdeployment/"$DEPLOYMENT_NAME" --type=json \
  -p='[{"op":"replace","path":"/spec/components/0/replicas","value":1}]'
kubectl get dynamographdeploymentenginegroup -n "$NS" "$GROUP_NAME"
kubectl get podclique -n "$NS" "$CLIQUE"
kubectl get pods -n "$NS" -l "nvidia.com/dynamo-engine-group=$GROUP_NAME" -o wide
