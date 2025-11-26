#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script to install Prometheus and Grafana monitoring stack for Dynamo
# Following the official Dynamo Kubernetes observability guide:
# https://github.com/ai-dynamo/dynamo/blob/main/docs/kubernetes/observability/metrics.md

set -e

echo "=========================================="
echo "Installing Prometheus & Grafana for Dynamo"
echo "=========================================="

# Step 1: Add Helm repository
echo ""
echo "Step 1: Adding prometheus-community Helm repository..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts

# Step 2: Update Helm repositories
echo ""
echo "Step 2: Updating Helm repositories..."
helm repo update

# Step 3: Install kube-prometheus-stack
echo ""
echo "Step 3: Installing kube-prometheus-stack..."
echo "This includes: Prometheus Operator, Prometheus, Grafana, Alertmanager"
helm install prometheus -n monitoring --create-namespace prometheus-community/kube-prometheus-stack \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
  --set-json 'prometheus.prometheusSpec.podMonitorNamespaceSelector={}' \
  --set-json 'prometheus.prometheusSpec.probeNamespaceSelector={}'

# Step 4: Wait for pods to be ready
echo ""
echo "Step 4: Waiting for monitoring stack pods to be ready..."
echo "This may take 1-2 minutes..."
kubectl wait --for=condition=ready pod -l "release=prometheus" -n monitoring --timeout=180s

# Step 5: Verify installation
echo ""
echo "Step 5: Verifying installation..."
kubectl get pods -n monitoring

# Step 6: Update Dynamo operator with Prometheus endpoint
echo ""
echo "Step 6: Updating Dynamo operator with Prometheus endpoint..."
helm upgrade dynamo-platform nvidia-dynamo/dynamo-platform \
  --namespace dynamo \
  --reuse-values \
  --set prometheusEndpoint=http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090

# Step 7: Configure DCGM custom metrics for NVLink profiling
echo ""
echo "Step 7: Configuring DCGM custom metrics for NVLink profiling..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
kubectl create configmap dcgm-exporter-metrics-config \
  --from-file=dcgm-metrics.csv="$SCRIPT_DIR/dcgm-metrics-with-nvlink.csv" \
  --namespace=gpu-operator \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Updating GPU Operator to use custom DCGM metrics..."
helm upgrade gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --reuse-values \
  --set dcgmExporter.config.name=dcgm-exporter-metrics-config

echo "Restarting DCGM exporter to apply new metrics configuration..."
kubectl rollout restart daemonset nvidia-dcgm-exporter -n gpu-operator
kubectl rollout status daemonset nvidia-dcgm-exporter -n gpu-operator --timeout=60s

echo "✅ DCGM custom metrics configured with NVLink profiling support"

# Step 8: Verify DCGM ServiceMonitor exists (created by GPU operator during cluster setup)
echo ""
echo "Step 8: Verifying DCGM exporter ServiceMonitor..."
if kubectl get servicemonitor -n gpu-operator nvidia-dcgm-exporter &>/dev/null; then
    echo "✅ DCGM ServiceMonitor found - GPU metrics will be available in Prometheus/Grafana"
else
    echo "⚠️  DCGM ServiceMonitor not found."
    echo "The GPU operator should have been installed with serviceMonitor enabled in createEks.sh"
    echo "Please verify the cluster was created with the updated createEks.sh that includes:"
    echo "  --set dcgmExporter.serviceMonitor.enabled=true"
    echo "  --set dcgmExporter.serviceMonitor.additionalLabels.release=prometheus"
fi

# Step 9: Get Grafana credentials
echo ""
echo "Step 9: Retrieving Grafana credentials..."
GRAFANA_USER=$(kubectl get secret -n monitoring prometheus-grafana -o jsonpath="{.data.admin-user}" | base64 --decode)
GRAFANA_PASSWORD=$(kubectl get secret -n monitoring prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 --decode)

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "Grafana Access:"
echo "  Username: $GRAFANA_USER"
echo "  Password: $GRAFANA_PASSWORD"
echo ""
echo "To access Grafana:"
echo "  kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring"
echo "  Then visit: http://localhost:3000"
echo ""
echo "To access Prometheus:"
echo "  kubectl port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090 -n monitoring"
echo "  Then visit: http://localhost:9090"
echo ""
echo "Next Steps:"
echo "  1. Deploy the Grafana dashboard ConfigMap:"
echo "     kubectl apply -f \"\$SCRIPT_DIR/../k8s/grafana-disagg-dashboard-configmap.yaml\""
echo "  2. Deploy or redeploy your DynamoGraphDeployment with DYN_SYSTEM_ENABLED=true"
echo "  3. The Dynamo operator will automatically create PodMonitors for metrics collection"
echo "  4. View metrics in Grafana under Dashboards → General → Dynamo Disaggregated Analysis"
echo ""
