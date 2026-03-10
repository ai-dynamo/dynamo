#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Usage:
#   ./run-benchmark.sh on   # benchmark with embedding cache ON (4GB)
#   ./run-benchmark.sh off  # benchmark with embedding cache OFF
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NAMESPACE="${NAMESPACE:dynamo}"

if [[ $# -ne 1 ]] || [[ "$1" != "on" && "$1" != "off" ]]; then
  echo "Usage: $0 <on|off>"
  exit 1
fi

MODE="$1"

if [[ "${MODE}" == "on" ]]; then
  CACHE_GB="4"
  CACHE_MODE="cache_on"
else
  CACHE_GB="0"
  CACHE_MODE="cache_off"
fi

echo "==> Embedding cache: ${MODE} (${CACHE_GB}GB)"

# Patch deploy.yaml: replace MM_EMBEDDING_CACHE_GB value
sed 's/value: "[0-9]*"  # MM_EMBEDDING_CACHE_GB/value: "'"${CACHE_GB}"'"  # MM_EMBEDDING_CACHE_GB/' \
  "${SCRIPT_DIR}/deploy.yaml" | \
  kubectl apply -f - -n "${NAMESPACE}"

echo "==> Waiting for worker to be ready..."
kubectl wait --for=condition=Ready \
  dynamographdeployment/qwen3-vl-agg \
  -n "${NAMESPACE}" --timeout=600s

# Delete old benchmark pod if exists
kubectl delete pod qwen3-vl-agg-benchmark \
  -n "${NAMESPACE}" --ignore-not-found

# Patch perf.yaml: replace CACHE_MODE value
sed 's/value: cache_o[nf]*/value: '"${CACHE_MODE}"'/' \
  "${SCRIPT_DIR}/perf.yaml" | \
  kubectl apply -f - -n "${NAMESPACE}"

echo "==> Benchmark pod launched (cache ${MODE})"
echo "    Monitor with: kubectl logs -f qwen3-vl-agg-benchmark -n ${NAMESPACE}"