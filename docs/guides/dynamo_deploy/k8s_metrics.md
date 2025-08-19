# Kubernetes Metrics Setup

This guide covers setting up comprehensive metrics collection for Dynamo deployments on Kubernetes using Prometheus and Grafana.

## Overview

The Dynamo Kubernetes Platform provides extensive telemetry through:
- **Application metrics**: Request latency, throughput, error rates
- **Infrastructure metrics**: GPU utilization, memory usage, network I/O
- **Kubernetes metrics**: Pod status, resource consumption, cluster health
- **Custom Dynamo metrics**: KV cache efficiency, disaggregation performance

## Prerequisites

- Kubernetes cluster with Dynamo deployed
- Cluster admin permissions for installing monitoring stack
- Sufficient storage for metrics retention (recommended: 50GB+ for production)

## Quick Setup with Helm

### 1. Install Prometheus Stack

```bash
# Add prometheus community helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install kube-prometheus-stack
helm install prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set grafana.adminPassword=admin123
```

### 2. Configure Dynamo Service Monitors

Create ServiceMonitor resources to scrape Dynamo metrics:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: dynamo-frontend-metrics
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: dynamo-frontend
  endpoints:
  - port: metrics
    path: /metrics
    interval: 15s
  namespaceSelector:
    matchNames:
    - default
    - dynamo-system
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: dynamo-worker-metrics
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: dynamo-worker
  endpoints:
  - port: metrics
    path: /metrics
    interval: 15s
  namespaceSelector:
    matchNames:
    - default
    - dynamo-system
```

### 3. Access Grafana Dashboard

```bash
# Port forward to Grafana
kubectl port-forward -n monitoring svc/prometheus-stack-grafana 3000:80

# Access at http://localhost:3000
# Username: admin
# Password: admin123
```

## Custom Metrics Configuration

### Enable Metrics in Dynamo Components

Update your Dynamo deployment to expose metrics endpoints:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dynamo-frontend
spec:
  template:
    spec:
      containers:
      - name: frontend
        image: nvcr.io/nvidia/ai-dynamo/frontend:latest
        ports:
        - name: http
          containerPort: 8000
        - name: metrics  # Add metrics port
          containerPort: 9090
        env:
        - name: ENABLE_METRICS
          value: "true"
        - name: METRICS_PORT
          value: "9090"
---
apiVersion: v1
kind: Service
metadata:
  name: dynamo-frontend
  labels:
    app: dynamo-frontend
spec:
  ports:
  - name: http
    port: 8000
    targetPort: 8000
  - name: metrics  # Add metrics service port
    port: 9090
    targetPort: 9090
  selector:
    app: dynamo-frontend
```

### GPU Metrics with DCGM

Install NVIDIA DCGM for detailed GPU metrics:

```bash
# Install NVIDIA GPU Operator with DCGM enabled
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set dcgmExporter.enabled=true \
  --set dcgmExporter.serviceMonitor.enabled=true
```

## Grafana Dashboards

### Import Dynamo Dashboard

1. Download the Dynamo dashboard JSON from: `deploy/monitoring/grafana-dashboard.json`
2. In Grafana, go to **Dashboards** > **Import**
3. Upload the JSON file or paste the content
4. Configure data source as your Prometheus instance

### Key Metrics to Monitor

**Request Metrics:**
- `dynamo_requests_total` - Total request count
- `dynamo_request_duration_seconds` - Request latency percentiles
- `dynamo_active_requests` - Currently processing requests

**Inference Metrics:**
- `dynamo_ttft_seconds` - Time to First Token
- `dynamo_itl_seconds` - Inter-Token Latency
- `dynamo_tokens_per_second` - Generation throughput

**KV Cache Metrics:**
- `dynamo_kv_cache_hit_rate` - Cache hit percentage
- `dynamo_kv_cache_memory_usage_bytes` - Cache memory consumption
- `dynamo_kv_transfers_total` - Cross-worker KV transfers

**GPU Metrics (via DCGM):**
- `DCGM_FI_DEV_GPU_UTIL` - GPU utilization
- `DCGM_FI_DEV_MEM_COPY_UTIL` - Memory bandwidth utilization
- `DCGM_FI_DEV_FB_USED` - GPU memory usage

## Alerting Rules

Create alerting rules for critical conditions:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: dynamo-alerts
  namespace: monitoring
spec:
  groups:
  - name: dynamo.rules
    rules:
    - alert: HighRequestLatency
      expr: histogram_quantile(0.95, dynamo_request_duration_seconds_bucket) > 2
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High request latency detected"
        description: "95th percentile latency is {{ $value }}s"
        
    - alert: LowKVCacheHitRate
      expr: dynamo_kv_cache_hit_rate < 0.7
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "KV cache hit rate is low"
        description: "Hit rate is {{ $value | humanizePercentage }}"
        
    - alert: GPUMemoryHigh
      expr: DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL > 0.9
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "GPU memory usage is high"
        description: "GPU {{ $labels.gpu }} memory usage is {{ $value | humanizePercentage }}"
```

## Performance Tuning

### Prometheus Configuration

Optimize retention and storage for your workload:

```yaml
# values.yaml for prometheus-stack helm chart
prometheus:
  prometheusSpec:
    retention: 30d
    retentionSize: 50GB
    storageSpec:
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 100Gi
          storageClassName: fast-ssd
```

### Scrape Interval Tuning

Adjust scrape intervals based on your monitoring needs:

```yaml
# High frequency for critical metrics
- job_name: 'dynamo-critical'
  scrape_interval: 5s
  metrics_path: '/metrics'
  static_configs:
  - targets: ['dynamo-frontend:9090']

# Lower frequency for resource metrics  
- job_name: 'dynamo-resources'
  scrape_interval: 30s
  metrics_path: '/metrics'
  static_configs:
  - targets: ['node-exporter:9100']
```

## Troubleshooting

### Common Issues

**Metrics not appearing:**
- Verify ServiceMonitor is in correct namespace
- Check Prometheus targets: `http://prometheus:9090/targets`
- Ensure firewall/network policies allow scraping

**High cardinality warnings:**
- Review metric labels and limit dynamic labels
- Use recording rules for complex queries
- Configure metric relabeling to drop unnecessary labels

**Grafana connection issues:**
- Verify Prometheus data source URL
- Check authentication credentials
- Ensure network connectivity between Grafana and Prometheus

### Debugging Commands

```bash
# Check Prometheus configuration
kubectl exec -n monitoring prometheus-prometheus-stack-kube-prom-prometheus-0 \
  -- cat /etc/prometheus/config_out/prometheus.env.yaml

# View ServiceMonitor status
kubectl get servicemonitor -n monitoring -o yaml

# Check DCGM exporter status
kubectl logs -n gpu-operator -l app=nvidia-dcgm-exporter
```

For advanced configuration and troubleshooting, see the [Prometheus Operator Documentation](https://prometheus-operator.dev/).