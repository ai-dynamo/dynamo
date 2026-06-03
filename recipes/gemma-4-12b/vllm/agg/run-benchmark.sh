#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Deploy the Gemma 4 12B aggregated graph, wait for it to be Ready, then launch
# the aiperf benchmark pod.
#
# Usage:
#   NAMESPACE=dynamo ./run-benchmark.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NAMESPACE="${NAMESPACE:-dynamo}"

echo "==> Applying deployment"
kubectl apply -f "${SCRIPT_DIR}/deploy.yaml" -n "${NAMESPACE}"

echo "==> Waiting for worker to be ready..."
kubectl wait --for=condition=Ready \
  dynamographdeployment/gemma4-12b-agg \
  -n "${NAMESPACE}" --timeout=1800s

# Delete old benchmark pod if it exists.
kubectl delete pod gemma4-12b-agg-benchmark \
  -n "${NAMESPACE}" --ignore-not-found

echo "==> Launching benchmark pod"
kubectl apply -f "${SCRIPT_DIR}/perf.yaml" -n "${NAMESPACE}"

echo "==> Benchmark pod launched"
echo "    Monitor with: kubectl logs -f gemma4-12b-agg-benchmark -n ${NAMESPACE}"
