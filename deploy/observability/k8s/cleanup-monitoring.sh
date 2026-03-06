#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script to cleanup/undo monitoring stack installation

set -e

echo "=========================================="
echo "Cleaning up Prometheus & Grafana for Dynamo"
echo "=========================================="

# Step 1: Delete Grafana dashboard ConfigMap
echo ""
echo "Step 1: Removing Grafana dashboard ConfigMap..."
kubectl delete configmap grafana-disagg-dashboard -n monitoring --ignore-not-found=true

# Step 2: Revert DCGM custom metrics configuration
echo ""
echo "Step 2: Reverting DCGM custom metrics to default..."
kubectl delete configmap dcgm-exporter-metrics-config -n gpu-operator --ignore-not-found=true

echo "Adding required Helm repositories..."
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia 2>/dev/null || true
helm repo add nvidia-dynamo https://helm.ngc.nvidia.com/nvidia/ai-dynamo 2>/dev/null || true
helm repo update

echo "Reverting GPU Operator DCGM settings to default..."
helm upgrade gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --reuse-values \
  --set dcgmExporter.config.name=""

echo "Restarting DCGM exporter to apply default metrics..."
kubectl rollout restart daemonset nvidia-dcgm-exporter -n gpu-operator
kubectl rollout status daemonset nvidia-dcgm-exporter -n gpu-operator --timeout=180s

# Step 3: Revert Dynamo operator Prometheus endpoint
echo ""
echo "Step 3: Removing Prometheus endpoint from Dynamo operator..."
DYNAMO_VERSION=$(helm list -n dynamo -o json | jq -r '.[] | select(.name=="dynamo-platform") | .chart' | sed 's/dynamo-platform-//')
echo "Detected Dynamo Platform version: ${DYNAMO_VERSION}"

# Delete the conflicting secret (grove-operator will recreate it)
echo "Removing grove-webhook-server-cert to avoid conflict..."
kubectl delete secret grove-webhook-server-cert -n dynamo --ignore-not-found=true

helm upgrade dynamo-platform nvidia-dynamo/dynamo-platform \
  --version "${DYNAMO_VERSION}" \
  --namespace dynamo \
  --reuse-values \
  --set prometheusEndpoint=""

# Step 4: Uninstall kube-prometheus-stack
echo ""
echo "Step 4: Uninstalling kube-prometheus-stack..."
helm uninstall prometheus -n monitoring || echo "Prometheus stack not found, skipping..."

echo "Deleting kube-prometheus-stack CRDs (Helm doesn't remove these automatically)..."
kubectl delete $(kubectl get crd -o name | grep monitoring.coreos.com) --ignore-not-found=true

# Step 5: Delete monitoring namespace (optional)
echo ""
read -p "Delete monitoring namespace? This will remove all monitoring data (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Deleting monitoring namespace..."
    kubectl delete namespace monitoring --ignore-not-found=true
else
    echo "Keeping monitoring namespace"
fi

echo ""
echo "=========================================="
echo "✅ Cleanup Complete!"
echo "=========================================="
echo ""
echo "You can now run setup-monitoring.sh to reinstall with a clean slate"
echo ""
