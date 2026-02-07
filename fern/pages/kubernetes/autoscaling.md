---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

This guide explains how to configure autoscaling for DynamoGraphDeployment (DGD) services using the `sglang-agg` example from `examples/backends/sglang/deploy/agg.yaml`.

## Example DGD

All examples in this guide use the following DGD:

```yaml
# examples/backends/sglang/deploy/agg.yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: sglang-agg
  namespace: default
spec:
  services:
    Frontend:
      dynamoNamespace: sglang-agg
      componentType: frontend
      replicas: 1

    decode:
      dynamoNamespace: sglang-agg
      componentType: worker
      replicas: 1
      resources:
        limits:
          gpu: "1"
```

**Key identifiers:**
- **DGD name**: `sglang-agg`
- **Namespace**: `default`
- **Services**: `Frontend`, `decode`
- **dynamo_namespace label**: `default-sglang-agg` (used for metric filtering)

## Overview

Dynamo provides flexible autoscaling through the `DynamoGraphDeploymentScalingAdapter` (DGDSA) resource. When you deploy a DGD, the operator automatically creates one adapter per service (unless explicitly disabled). These adapters implement the Kubernetes [Scale subresource](https://kubernetes.io/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/#scale-subresource), enabling integration with:

| Autoscaler | Description | Best For |
|------------|-------------|----------|
| **KEDA** | Event-driven autoscaling (recommended) | Most use cases |
| **Kubernetes HPA** | Native horizontal pod autoscaling | Simple CPU/memory-based scaling |
| **Dynamo Planner** | LLM-aware autoscaling with SLA optimization | Production LLM workloads |
| **Custom Controllers** | Any scale-subresource-compatible controller | Custom requirements |

> **⚠️ Deprecation Notice**: The `spec.services[X].autoscaling` field in DGD is **deprecated and ignored**. Use DGDSA with HPA, KEDA, or Planner instead. If you have existing DGDs with `autoscaling` configured, you'll see a warning. Remove the field to silence the warning.

## Architecture

```
┌──────────────────────────────────┐          ┌─────────────────────────────────────┐
│   DynamoGraphDeployment          │          │   Scaling Adapters (auto-created)   │
│   "sglang-agg"                   │          │   (one per service)                 │
├──────────────────────────────────┤          ├─────────────────────────────────────┤
│                                  │          │                                     │
│  spec.services:                  │          │  ┌─────────────────────────────┐    │      ┌──────────────────┐
│                                  │          │  │ sglang-agg-frontend         │◄───┼──────│   Autoscalers    │
│    ┌────────────────────────┐◄───┼──────────┼──│ spec.replicas: 1            │    │      │                  │
│    │ Frontend: 1 replica    │    │          │  └─────────────────────────────┘    │      │  • KEDA          │
│    └────────────────────────┘    │          │                                     │      │  • HPA           │
│                                  │          │  ┌─────────────────────────────┐    │      │  • Planner       │
│    ┌────────────────────────┐◄───┼──────────┼──│ sglang-agg-decode           │◄───┼──────│  • Custom        │
│    │ decode:   1 replica    │    │          │  │ spec.replicas: 1            │    │      │                  │
│    └────────────────────────┘    │          │  └─────────────────────────────┘    │      └──────────────────┘
│                                  │          │                                     │
└──────────────────────────────────┘          └─────────────────────────────────────┘
```

**How it works:**

1. You deploy a DGD with services (Frontend, decode)
2. The operator auto-creates one DGDSA per service
3. Autoscalers (KEDA, HPA, Planner) target the adapters via `/scale` subresource
4. Adapter controller syncs replica changes to the DGD
5. DGD controller reconciles the underlying pods

## Viewing Scaling Adapters

After deploying the `sglang-agg` DGD, verify the auto-created adapters:

```bash
kubectl get dgdsa -n default

# Example output:
# NAME                  DGD         SERVICE    REPLICAS   AGE
# sglang-agg-frontend   sglang-agg  Frontend   1          5m
# sglang-agg-decode     sglang-agg  decode     1          5m
```

## Replica Ownership Model

When DGDSA is enabled (the default), it becomes the **source of truth** for replica counts. This follows the same pattern as Kubernetes Deployments owning ReplicaSets.

### How It Works

1. **DGDSA owns replicas**: Autoscalers (HPA, KEDA, Planner) update the DGDSA's `spec.replicas`
2. **DGDSA syncs to DGD**: The DGDSA controller writes the replica count to the DGD's service
3. **Direct DGD edits blocked**: A validating webhook prevents users from directly editing `spec.services[X].replicas` in the DGD
4. **Controllers allowed**: Only authorized controllers (operator, Planner) can modify DGD replicas

### Manual Scaling with DGDSA Enabled

When DGDSA is enabled, use `kubectl scale` on the adapter (not the DGD):

```bash
# ✅ Correct - scale via DGDSA
kubectl scale dgdsa sglang-agg-decode --replicas=3

# ❌ Blocked - direct DGD edit rejected by webhook
kubectl patch dgd sglang-agg --type=merge -p '{"spec":{"services":{"decode":{"replicas":3}}}}'
# Error: spec.services[decode].replicas cannot be modified directly when scaling adapter is enabled;
#        use 'kubectl scale dgdsa/sglang-agg-decode --replicas=3' or update the DynamoGraphDeploymentScalingAdapter instead
```

## Enabling DGDSA for a Service

By default, no DGDSA is created for services, allowing direct replica management via the DGD. To enable autoscaling via HPA, KEDA, or Planner, explicitly enable the scaling adapter:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: sglang-agg
spec:
  services:
    Frontend:
      replicas: 2        # ← No DGDSA by default, direct edits allowed

    decode:
      replicas: 1
      scalingAdapter:
        enabled: true    # ← DGDSA created, managed via adapter
```

**When to enable DGDSA:**
- You want to use HPA, KEDA, or Planner for autoscaling
- You want a clear separation between "desired scale" (adapter) and "deployment config" (DGD)
- You want protection against accidental direct replica edits

**When to keep DGDSA disabled (default):**
- You want simple, manual replica management
- You don't need autoscaling for that service
- You prefer direct DGD edits over adapter-based scaling

## Autoscaling with Dynamo Planner

The Dynamo Planner is an LLM-aware autoscaler that optimizes scaling decisions based on inference-specific metrics like Time To First Token (TTFT), Inter-Token Latency (ITL), and KV cache utilization.

**When to use Planner:**
- You want LLM-optimized autoscaling out of the box
- You need coordinated scaling across prefill/decode services
- You want SLA-driven scaling (e.g., target TTFT \< 500ms)

**How Planner works:**

Planner is deployed as a service component within your DGD. It:
1. Queries Prometheus for frontend metrics (request rate, latency, etc.)
2. Uses profiling data to predict optimal replica counts
3. Scales prefill/decode workers to meet SLA targets

**Deployment:**

The recommended way to deploy Planner is via `DynamoGraphDeploymentRequest` (DGDR). See the [SLA Planner Quick Start](../components/planner/planner-guide.md) for complete instructions.

Example configurations with Planner:
- `examples/backends/vllm/deploy/disagg_planner.yaml`
- `examples/backends/sglang/deploy/disagg_planner.yaml`
- `examples/backends/trtllm/deploy/disagg_planner.yaml`

For more details, see the [SLA Planner documentation](../components/planner/planner-guide.md).

## Autoscaling with Kubernetes HPA

The Horizontal Pod Autoscaler (HPA) is Kubernetes' native autoscaling solution.

**When to use HPA:**
- You have simple, predictable scaling requirements
- You want to use standard Kubernetes tooling
- You need CPU or memory-based scaling

<Note>
For custom metrics (like TTFT or queue depth), consider using [KEDA](#autoscaling-with-keda-recommended) instead - it's simpler to configure.
</Note>

### Basic HPA (CPU-based)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sglang-agg-frontend-hpa
  namespace: default
spec:
  scaleTargetRef:
    apiVersion: nvidia.com/v1alpha1
    kind: DynamoGraphDeploymentScalingAdapter
    name: sglang-agg-frontend
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
    scaleUp:
      stabilizationWindowSeconds: 0
```

### HPA with Dynamo Metrics

Dynamo exports several metrics useful for autoscaling. These are available at the `/metrics` endpoint on each frontend pod.

> **See also**: For a complete list of all Dynamo metrics, see the [Metrics Reference](../observability/metrics.md). For Prometheus and Grafana setup, see the [Prometheus and Grafana Setup Guide](../observability/prometheus-grafana.md).

#### Available Dynamo Metrics

| Metric | Type | Description | Good for scaling |
|--------|------|-------------|------------------|
| `dynamo_frontend_queued_requests` | Gauge | Requests waiting in HTTP queue | ✅ Workers |
| `dynamo_frontend_inflight_requests` | Gauge | Concurrent requests to engine | ✅ All services |
| `dynamo_frontend_time_to_first_token_seconds` | Histogram | TTFT latency | ✅ Workers |
| `dynamo_frontend_inter_token_latency_seconds` | Histogram | ITL latency | ✅ Decode |
| `dynamo_frontend_request_duration_seconds` | Histogram | Total request duration | ⚠️ General |

#### Metric Labels

Dynamo metrics include these labels for filtering:

| Label | Description | Example |
|-------|-------------|---------|
| `dynamo_namespace` | Unique DGD identifier (`{k8s-namespace}-{dynamoNamespace}`) | `default-sglang-agg` |
| `model` | Model being served | `Qwen/Qwen3-0.6B` |

<Note>
When you have multiple DGDs in the same namespace, use `dynamo_namespace` to filter metrics for a specific DGD.
</Note>

#### Example: Scale Decode Service Based on TTFT

Using HPA with Prometheus Adapter requires configuring external metrics.

**Step 1: Configure Prometheus Adapter**

Add this to your Helm values file (e.g., `prometheus-adapter-values.yaml`):

```yaml
# prometheus-adapter-values.yaml
prometheus:
  url: http://prometheus-kube-prometheus-prometheus.monitoring.svc
  port: 9090

rules:
  external:
  # TTFT p95 from frontend - used to scale decode
  - seriesQuery: 'dynamo_frontend_time_to_first_token_seconds_bucket{namespace!=""}'
    resources:
      overrides:
        namespace: {resource: "namespace"}
    name:
      as: "dynamo_ttft_p95_seconds"
    metricsQuery: |
      histogram_quantile(0.95,
        sum(rate(dynamo_frontend_time_to_first_token_seconds_bucket{<<.LabelMatchers>>}[5m]))
        by (le, namespace, dynamo_namespace)
      )
```

**Step 2: Install Prometheus Adapter**

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm upgrade --install prometheus-adapter prometheus-community/prometheus-adapter \
  -n monitoring --create-namespace \
  -f prometheus-adapter-values.yaml
```

**Step 3: Verify the metric is available**

```bash
kubectl get --raw "/apis/external.metrics.k8s.io/v1beta1/namespaces/<your-namespace>/dynamo_ttft_p95_seconds" | jq
```

**Step 4: Create the HPA**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sglang-agg-decode-hpa
spec:
  scaleTargetRef:
    apiVersion: nvidia.com/v1alpha1
    kind: DynamoGraphDeploymentScalingAdapter
    name: sglang-agg-decode              # ← DGD name + service name (lowercase)
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: External
    external:
      metric:
        name: dynamo_ttft_p95_seconds
        selector:
          matchLabels:
            dynamo_namespace: "default-sglang-agg"  # ← {namespace}-{dynamoNamespace}
      target:
        type: Value
        value: "500m"  # Scale up when TTFT p95 > 500ms
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 60    # Wait 1 min before scaling down
      policies:
      - type: Pods
        value: 1
        periodSeconds: 30
    scaleUp:
      stabilizationWindowSeconds: 0      # Scale up immediately
      policies:
      - type: Pods
        value: 2
        periodSeconds: 30
```

**How it works:**
1. Frontend pods export `dynamo_frontend_time_to_first_token_seconds` histogram
2. Prometheus Adapter calculates p95 TTFT per `dynamo_namespace`
3. HPA monitors this metric filtered by `dynamo_namespace: "default-sglang-agg"`
4. When TTFT p95 > 500ms, HPA scales up the `sglang-agg-decode` adapter
5. Adapter controller syncs the replica count to the DGD's `decode` service
6. More decode workers are created, reducing TTFT

#### Example: Scale Based on Queue Depth

Add this rule to your `prometheus-adapter-values.yaml` (alongside the TTFT rule):

```yaml
# Add to rules.external in prometheus-adapter-values.yaml
- seriesQuery: 'dynamo_frontend_queued_requests{namespace!=""}'
  resources:
    overrides:
      namespace: {resource: "namespace"}
  name:
    as: "dynamo_queued_requests"
  metricsQuery: |
    sum(<<.Series>>{<<.LabelMatchers>>}) by (namespace, dynamo_namespace)
```

Then create the HPA:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sglang-agg-decode-queue-hpa
  namespace: default
spec:
  scaleTargetRef:
    apiVersion: nvidia.com/v1alpha1
    kind: DynamoGraphDeploymentScalingAdapter
    name: sglang-agg-decode
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: External
    external:
      metric:
        name: dynamo_queued_requests
        selector:
          matchLabels:
            dynamo_namespace: "default-sglang-agg"
      target:
        type: Value
        value: "10"  # Scale up when queue > 10 requests
```

## Autoscaling with KEDA (Recommended)

KEDA (Kubernetes Event-driven Autoscaling) extends Kubernetes with event-driven autoscaling, supporting 50+ scalers including Prometheus.

**Advantages over HPA + Prometheus Adapter:**
- No Prometheus Adapter configuration needed
- PromQL queries are defined in the ScaledObject itself (declarative, per-deployment)
- Easy to update - just `kubectl apply` the ScaledObject
- Can scale to zero when idle
- Supports multiple triggers per object

**When to use KEDA:**
- You want simpler configuration (no Prometheus Adapter to manage)
- You need event-driven scaling (e.g., queue depth, Kafka, etc.)
- You want to scale to zero when idle

### Installing KEDA

```bash
# Add KEDA Helm repo
helm repo add kedacore https://kedacore.github.io/charts
helm repo update

# Install KEDA
helm install keda kedacore/keda \
  --namespace keda \
  --create-namespace

# Verify installation
kubectl get pods -n keda
```

<Note>
If you have Prometheus Adapter installed, either uninstall it first (`helm uninstall prometheus-adapter -n monitoring`) or install KEDA with `--set metricsServer.enabled=false` to avoid API conflicts.
</Note>

### Example: Scale Decode Based on TTFT

Using the `sglang-agg` DGD from `examples/backends/sglang/deploy/agg.yaml`:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: sglang-agg-decode-scaler
  namespace: default
spec:
  scaleTargetRef:
    apiVersion: nvidia.com/v1alpha1
    kind: DynamoGraphDeploymentScalingAdapter
    name: sglang-agg-decode
  minReplicaCount: 1
  maxReplicaCount: 10
  pollingInterval: 15      # Check metrics every 15 seconds
  cooldownPeriod: 60       # Wait 60s before scaling down
  triggers:
  - type: prometheus
    metadata:
      # Update this URL to match your Prometheus service
      serverAddress: http://prometheus-kube-prometheus-prometheus.monitoring.svc:9090
      metricName: dynamo_ttft_p95
      query: |
        histogram_quantile(0.95,
          sum(rate(dynamo_frontend_time_to_first_token_seconds_bucket{dynamo_namespace="default-sglang-agg"}[5m]))
          by (le)
        )
      threshold: "0.5"              # Scale up when TTFT p95 > 500ms (0.5 seconds)
      activationThreshold: "0.1"    # Start scaling when TTFT > 100ms
```

Apply it:

```bash
kubectl apply -f sglang-agg-decode-scaler.yaml
```

### Verify KEDA Scaling

```bash
# Check ScaledObject status
kubectl get scaledobject -n default

# KEDA creates an HPA under the hood - you can see it
kubectl get hpa -n default

# Example output:
# NAME                                REFERENCE                                              TARGETS      MINPODS   MAXPODS   REPLICAS
# keda-hpa-sglang-agg-decode-scaler   DynamoGraphDeploymentScalingAdapter/sglang-agg-decode  45m/500m     1         10        1

# Get detailed status
kubectl describe scaledobject sglang-agg-decode-scaler -n default
```

### Example: Scale Based on Queue Depth

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: sglang-agg-decode-queue-scaler
  namespace: default
spec:
  scaleTargetRef:
    apiVersion: nvidia.com/v1alpha1
    kind: DynamoGraphDeploymentScalingAdapter
    name: sglang-agg-decode
  minReplicaCount: 1
  maxReplicaCount: 10
  pollingInterval: 15
  cooldownPeriod: 60
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus-kube-prometheus-prometheus.monitoring.svc:9090
      metricName: dynamo_queued_requests
      query: |
        sum(dynamo_frontend_queued_requests{dynamo_namespace="default-sglang-agg"})
      threshold: "10"    # Scale up when queue > 10 requests
```

### How KEDA Works

KEDA creates and manages an HPA under the hood:

```
┌──────────────────────────────────────────────────────────────────────┐
│  You create: ScaledObject                                            │
│    - scaleTargetRef: sglang-agg-decode                               │
│    - triggers: prometheus query                                      │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  KEDA Operator automatically creates: HPA                            │
│    - name: keda-hpa-sglang-agg-decode-scaler                         │
│    - scaleTargetRef: sglang-agg-decode                               │
│    - metrics: External (from KEDA metrics server)                    │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  DynamoGraphDeploymentScalingAdapter: sglang-agg-decode              │
│    - spec.replicas: updated by HPA                                   │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  DynamoGraphDeployment: sglang-agg                                   │
│    - spec.services.decode.replicas: synced from adapter              │
└──────────────────────────────────────────────────────────────────────┘
```

## Mixed Autoscaling

For disaggregated deployments (prefill + decode), you can use different autoscaling strategies for different services:

```yaml
---
# HPA for Frontend (CPU-based)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sglang-agg-frontend-hpa
  namespace: default
spec:
  scaleTargetRef:
    apiVersion: nvidia.com/v1alpha1
    kind: DynamoGraphDeploymentScalingAdapter
    name: sglang-agg-frontend
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70

---
# KEDA for Decode (TTFT-based)
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: sglang-agg-decode-scaler
  namespace: default
spec:
  scaleTargetRef:
    apiVersion: nvidia.com/v1alpha1
    kind: DynamoGraphDeploymentScalingAdapter
    name: sglang-agg-decode
  minReplicaCount: 1
  maxReplicaCount: 10
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus-kube-prometheus-prometheus.monitoring.svc:9090
      query: |
        histogram_quantile(0.95,
          sum(rate(dynamo_frontend_time_to_first_token_seconds_bucket{dynamo_namespace="default-sglang-agg"}[5m]))
          by (le)
        )
      threshold: "0.5"
```

## Manual Scaling

### With DGDSA Enabled (Default)

When DGDSA is enabled (the default), scale via the adapter:

```bash
kubectl scale dgdsa sglang-agg-decode -n default --replicas=3
```

Verify the scaling:

```bash
kubectl get dgdsa sglang-agg-decode -n default

# Output:
# NAME                DGD         SERVICE   REPLICAS   AGE
# sglang-agg-decode   sglang-agg  decode    3          10m
```

<Note>
If an autoscaler (KEDA, HPA, Planner) is managing the adapter, your change will be overwritten on the next evaluation cycle.
</Note>

### With DGDSA Disabled

If you've disabled the scaling adapter for a service, edit the DGD directly:

```bash
kubectl patch dgd sglang-agg --type=merge -p '{"spec":{"services":{"decode":{"replicas":3}}}}'
```

Or edit the YAML (no `scalingAdapter.enabled: true` means direct edits are allowed):

```yaml
spec:
  services:
    decode:
      replicas: 3
      # No scalingAdapter.enabled means replicas can be edited directly
```

## Best Practices

### 1. Choose One Autoscaler Per Service

Avoid configuring multiple autoscalers for the same service:

| Configuration | Status |
|---------------|--------|
| HPA for frontend, Planner for prefill/decode | ✅ Good |
| KEDA for all services | ✅ Good |
| Planner only (default) | ✅ Good |
| HPA + Planner both targeting decode | ❌ Bad - they will fight |

### 2. Use Appropriate Metrics

| Service Type | Recommended Metrics | Dynamo Metric |
|--------------|---------------------|---------------|
| Frontend | CPU utilization, request rate | `dynamo_frontend_requests_total` |
| Prefill | Queue depth, TTFT | `dynamo_frontend_queued_requests`, `dynamo_frontend_time_to_first_token_seconds` |
| Decode | ITL | `dynamo_frontend_inter_token_latency_seconds` |

### 3. Configure Stabilization Windows

Prevent thrashing with appropriate stabilization:

```yaml
# HPA
behavior:
  scaleDown:
    stabilizationWindowSeconds: 300  # Wait 5 min before scaling down
  scaleUp:
    stabilizationWindowSeconds: 0    # Scale up immediately

# KEDA
spec:
  cooldownPeriod: 300
```

### 4. Set Sensible Min/Max Replicas

Always configure minimum and maximum replicas in your HPA/KEDA to prevent:
- Scaling to zero (unless intentional)
- Unbounded scaling that exhausts cluster resources

## Troubleshooting

### Adapters Not Created

```bash
# Check DGD status
kubectl describe dgd sglang-agg -n default

# Check operator logs
kubectl logs -n dynamo-system deployment/dynamo-operator
```

### Scaling Not Working

```bash
# Check adapter status
kubectl describe dgdsa sglang-agg-decode -n default

# Check HPA/KEDA status
kubectl describe hpa sglang-agg-decode-hpa -n default
kubectl describe scaledobject sglang-agg-decode-scaler -n default

# Verify metrics are available in Kubernetes metrics API
kubectl get --raw /apis/external.metrics.k8s.io/v1beta1
```

### Metrics Not Available

If HPA/KEDA shows `<unknown>` for metrics:

```bash
# Check if Dynamo metrics are being scraped
kubectl port-forward -n default svc/sglang-agg-frontend 8000:8000
curl http://localhost:8000/metrics | grep dynamo_frontend

# Example output:
# dynamo_frontend_queued_requests{model="Qwen/Qwen3-0.6B"} 2
# dynamo_frontend_inflight_requests{model="Qwen/Qwen3-0.6B"} 5

# Verify Prometheus is scraping the metrics
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
# Then query: dynamo_frontend_time_to_first_token_seconds_bucket

# Check KEDA operator logs
kubectl logs -n keda deployment/keda-operator
```

### Rapid Scaling Up and Down

If you see unstable scaling:

1. Check if multiple autoscalers are targeting the same adapter
2. Increase `cooldownPeriod` in KEDA ScaledObject
3. Increase `stabilizationWindowSeconds` in HPA behavior

## References

- [Kubernetes HPA Documentation](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [KEDA Documentation](https://keda.sh/)
- [Prometheus Adapter](https://github.com/kubernetes-sigs/prometheus-adapter)
- [Planner Documentation](../components/planner/planner-guide.md)
- [Dynamo Metrics Reference](../observability/metrics.md)
- [Prometheus and Grafana Setup](../observability/prometheus-grafana.md)

