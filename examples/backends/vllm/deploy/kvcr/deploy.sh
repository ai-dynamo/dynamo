#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

render() {
  substitution_vars='${DYNAMO_VLLM_IMAGE} ${DYNAMO_RDMA_RESOURCE}'
  substitution_vars="$substitution_vars "'${DYNAMO_UCX_NET_DEVICES}'
  substitution_vars="$substitution_vars "'${DYNAMO_KVCR_COMPATIBILITY_DIGEST}'
  envsubst "$substitution_vars" < "$manifest"
}

case "${1:-}" in
  --render-only)
    render
    ;;
  "")
    if ! kubectl api-resources --api-group=grove.io -o name 2>/dev/null \
      | grep -qx 'podcliquesets.grove.io'; then
      echo "The KVCR examples require a Dynamo operator with Grove enabled" >&2
      exit 1
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
    echo "usage: $0 [--render-only]" >&2
    exit 2
    ;;
esac
