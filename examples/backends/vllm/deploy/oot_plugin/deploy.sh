#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

: "${DYNAMO_VLLM_IMAGE:?set DYNAMO_VLLM_IMAGE to the custom runtime image built from ./Dockerfile}"
: "${OOT_MODEL:?set OOT_MODEL to the Hugging Face model id, e.g. XHToken/Spark-X2.6-4B}"
: "${OOT_SERVED_MODEL_NAME:?set OOT_SERVED_MODEL_NAME to the public model name, e.g. spark-x2.6}"

OOT_MAX_MODEL_LEN=${OOT_MAX_MODEL_LEN:-131072}
NAMESPACE=${NAMESPACE:-default}
export DYNAMO_VLLM_IMAGE OOT_MODEL OOT_SERVED_MODEL_NAME OOT_MAX_MODEL_LEN

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
manifest=$script_dir/agg_oot_plugin.yaml
deployment_name=vllm-agg-oot-plugin

render() {
  envsubst '${DYNAMO_VLLM_IMAGE} ${OOT_MODEL} ${OOT_SERVED_MODEL_NAME} ${OOT_MAX_MODEL_LEN}' \
    < "$manifest"
}

case "${1:-}" in
  --render-only)
    render
    ;;
  "")
    render | kubectl apply -n "$NAMESPACE" -f -
    echo "Deployed $deployment_name in namespace $NAMESPACE."
    echo "Watch rollout: kubectl get pods -n $NAMESPACE -l nvidia.com/dynamo-graph-deployment-name=$deployment_name -w"
    ;;
  *)
    echo "usage: $0 [--render-only]" >&2
    exit 2
    ;;
esac
