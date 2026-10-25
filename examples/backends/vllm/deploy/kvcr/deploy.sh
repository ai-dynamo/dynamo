#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

: "${DYNAMO_VLLM_IMAGE:?set DYNAMO_VLLM_IMAGE to a KVCR-capable image}"
: "${DYNAMO_UCX_NET_DEVICES:?set DYNAMO_UCX_NET_DEVICES to the GPU-local UCX device}"

DYNAMO_RDMA_RESOURCE=${DYNAMO_RDMA_RESOURCE:-rdma/shared_ib}
KVCR_MEMORY_SERVICE_ENABLED=${KVCR_MEMORY_SERVICE_ENABLED:-false}
NAMESPACE=${NAMESPACE:-default}
export DYNAMO_KVCR_COMPATIBILITY_DIGEST DYNAMO_RDMA_RESOURCE
export DYNAMO_UCX_NET_DEVICES DYNAMO_VLLM_IMAGE

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
case "$KVCR_MEMORY_SERVICE_ENABLED" in
  false)
    manifest=$script_dir/agg.yaml
    deployment_name=vllm-agg-kvcr
    ;;
  true)
    manifest=$script_dir/agg-memory-service.yaml
    deployment_name=vllm-agg-kvcr-memory-service
    : "${DYNAMO_KVCR_COMPATIBILITY_DIGEST:?set a versioned KVCR compatibility value}"
    case "$DYNAMO_KVCR_COMPATIBILITY_DIGEST" in
      *[!A-Za-z0-9._:/-]*)
        echo "DYNAMO_KVCR_COMPATIBILITY_DIGEST contains an unsupported character" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "KVCR_MEMORY_SERVICE_ENABLED must be true or false" >&2
    exit 2
    ;;
esac

# Keep startup code and an optional development wheel out of the base image.
# Content-addressed, immutable ConfigMaps cannot change under a live engine.
startup_configmap() {
  python3 - "$script_dir" "${KVCR_WHEEL:-}" "${1:-}" <<'PY'
import base64
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
data = {
    "services.sh": (root / "services.sh").read_text(),
    "launch_utils.sh": (root.parents[3] / "common/launch_utils.sh").read_text(),
}
binary = {}
if sys.argv[2]:
    binary["kvcr.whl"] = base64.b64encode(pathlib.Path(sys.argv[2]).read_bytes()).decode()
payload = json.dumps({"data": data, "binaryData": binary}, sort_keys=True)
if len(payload.encode()) > 1024 * 1024:
    sys.exit("KVCR startup bundle exceeds 1 MiB; use a smaller pure-Python wheel")
name = "kvcr-startup-" + hashlib.sha256(payload.encode()).hexdigest()[:16]
print(name if sys.argv[3] == "name" else json.dumps({
    "apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": name},
    "immutable": True, "data": data, "binaryData": binary,
}))
PY
}

if [ "$KVCR_MEMORY_SERVICE_ENABLED" = true ]; then
  KVCR_STARTUP_CONFIG_MAP=$(startup_configmap name)
  export KVCR_STARTUP_CONFIG_MAP
fi

render() {
  substitution_vars='${DYNAMO_VLLM_IMAGE} ${DYNAMO_RDMA_RESOURCE}'
  substitution_vars="$substitution_vars "'${DYNAMO_UCX_NET_DEVICES}'
  substitution_vars="$substitution_vars "'${DYNAMO_KVCR_COMPATIBILITY_DIGEST}'
  substitution_vars="$substitution_vars "'${KVCR_STARTUP_CONFIG_MAP}'
  envsubst "$substitution_vars" < "$manifest"
}

case "${1:-}" in
  --render-only)
    render
    ;;
  --render-configmap)
    if [ "$KVCR_MEMORY_SERVICE_ENABLED" != true ]; then
      echo "--render-configmap requires KVCR_MEMORY_SERVICE_ENABLED=true" >&2
      exit 2
    fi
    startup_configmap
    ;;
  "")
    if ! kubectl api-resources --api-group=grove.io -o name 2>/dev/null \
      | grep -qx 'podcliquesets.grove.io'; then
      echo "The KVCR examples require a Dynamo operator with Grove enabled" >&2
      exit 1
    fi
    if [ "$KVCR_MEMORY_SERVICE_ENABLED" = true ]; then
      startup_configmap | kubectl apply -n "$NAMESPACE" -f -
    fi
    render | kubectl apply -n "$NAMESPACE" -f -
    provider=
    for _ in $(seq 1 20); do
      provider=$(kubectl get dynamographdeployment "$deployment_name" \
        -n "$NAMESPACE" \
        -o jsonpath='{.metadata.annotations.nvidia\.com/workload-provider}' \
        2>/dev/null || true)
      [ -z "$provider" ] || break
      sleep 1
    done
    if [ "$provider" != grove ]; then
      echo "The operator selected '${provider:-unknown}', not Grove;" \
        "DGD $NAMESPACE/$deployment_name remains applied" >&2
      exit 1
    fi
    ;;
  *)
    echo "usage: $0 [--render-only|--render-configmap]" >&2
    exit 2
    ;;
esac
