# Autoscaling

This guide explains how to configure autoscaling for DynamoGraphDeployment (DGD) services. Dynamo supports multiple autoscaling strategies to meet different use cases, from simple CPU-based scaling to sophisticated LLM-aware optimization.

## Overview

Dynamo provides flexible autoscaling through the `DynamoGraphDeploymentScalingAdapter` (DGDSA) resource. When you deploy a DGD, the operator automatically creates one adapter per service. These adapters implement the Kubernetes [Scale subresource](https://kubernetes.io/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/#scale-subresource), enabling integration with:

| Autoscaler | Description | Best For |
|------------|-------------|----------|
| **Dynamo Planner** | LLM-aware autoscaling with SLA optimization | Production LLM workloads |
| **Kubernetes HPA** | Native horizontal pod autoscaling | Simple CPU/memory-based scaling |
| **KEDA** | Event-driven autoscaling | Queue-based or external metrics |
| **Custom Controllers** | Any scale-subresource-compatible controller | Custom requirements |

## Architecture

```
┌──────────────────────────────────┐          ┌─────────────────────────────────────┐
│   DynamoGraphDeployment          │          │   Scaling Adapters (auto-created)   │
│   "my-llm-deployment"            │          │   (one per service)                 │
├──────────────────────────────────┤          ├─────────────────────────────────────┤
│                                  │          │                                     │
│  spec.services:                  │          │  ┌─────────────────────────────┐    │      ┌──────────────────┐
│                                  │          │  │ my-llm-deployment-frontend  │◄───┼──────│   Autoscalers    │
│    ┌────────────────────────┐◄───┼──────────┼──│ spec.replicas: 2            │    │      │                  │
│    │ frontend:  2 replicas  │    │          │  └─────────────────────────────┘    │      │  • Planner       │
│    └────────────────────────┘    │          │                                     │      │  • HPA           │
│                                  │          │  ┌─────────────────────────────┐    │      │  • KEDA          │
│    ┌────────────────────────┐◄───┼──────────┼──│ my-llm-deployment-prefill   │◄───┼──────│  • Custom        │
│    │ prefill:   4 replicas  │    │          │  │ spec.replicas: 4            │    │      │                  │
│    └────────────────────────┘    │          │  └─────────────────────────────┘    │      └──────────────────┘
│                                  │          │                                     │
│    ┌────────────────────────┐◄───┼──────────┼──┌─────────────────────────────┐    │
│    │ decode:    8 replicas  │    │          │  │ my-llm-deployment-decode    │◄───┼──────
│    └────────────────────────┘    │          │  │ spec.replicas: 8            │    │
│                                  │          │  └─────────────────────────────┘    │
└──────────────────────────────────┘          └─────────────────────────────────────┘
```

**How it works:**

1. You deploy a DGD with services (frontend, prefill, decode, etc.)
2. The operator auto-creates one DGDSA per service
3. Autoscalers (HPA, KEDA, Planner) target the adapters via `/scale` subresource
4. Adapter controller syncs replica changes to the DGD
5. DGD controller reconciles the underlying pods

## Viewing Scaling Adapters

After deploying a DGD, verify the auto-created adapters:

```bash
kubectl get dgdsa -n <namespace>

# Example output:
# NAME                          DGD               SERVICE    REPLICAS   AGE
# my-llm-deployment-frontend    my-llm-deployment frontend   2          5m
# my-llm-deployment-prefill     my-llm-deployment prefill    4          5m
# my-llm-deployment-decode      my-llm-deployment decode     8          5m
```

## Autoscaling with Dynamo Planner

The Dynamo Planner is an LLM-aware autoscaler that optimizes scaling decisions based on inference-specific metrics like Time To First Token (TTFT), Inter-Token Latency (ITL), and KV cache utilization.

**When to use Planner:**
- You want LLM-optimized autoscaling out of the box
- You need coordinated scaling across prefill/decode services
- You want SLA-driven scaling (e.g., target TTFT < 500ms)

**How Planner works:**

Planner is deployed as a service component within your DGD. It:
1. Queries Prometheus for frontend metrics (request rate, latency, etc.)
2. Uses profiling data to predict optimal replica counts
3. Scales prefill/decode workers to meet SLA targets

**Deployment:**

The recommended way to deploy Planner is via `DynamoGraphDeploymentRequest` (DGDR), which automatically:
1. Profiles your model to find optimal configurations
2. Generates a DGD with Planner included
3. Deploys the optimized configuration

See the [SLA Planner Quick Start](../planner/sla_planner_quickstart.md) for complete instructions.

**Manual Planner deployment:**

You can also manually add Planner to your DGD. Example configurations are available in:
- `examples/backends/vllm/deploy/disagg_planner.yaml`
- `examples/backends/sglang/deploy/disagg_planner.yaml`
- `examples/backends/trtllm/deploy/disagg_planner.yaml`

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-llm-deployment
  namespace: llm-serving
spec:
  backendFramework: vllm
  services:
    frontend:
      replicas: 2
      componentType: frontend
    prefill:
      replicas: 4
      componentType: worker
      subComponentType: prefill
    decode:
      replicas: 8
      componentType: worker
      subComponentType: decode
    # Planner service
    planner:
      replicas: 1
      componentType: planner
      # Planner requires profiling data and Prometheus access
      # See examples/backends/*/deploy/disagg_planner.yaml for full configuration
```

For more details, see the [SLA Planner documentation](../planner/sla_planner.md).

## Autoscaling with Kubernetes HPA

The Horizontal Pod Autoscaler (HPA) is Kubernetes' native autoscaling solution.

**When to use HPA:**
- You have simple, predictable scaling requirements
- You want to use standard Kubernetes tooling
- You need CPU or memory-based scaling

### Basic HPA (CPU-based)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: frontend-hpa
  namespace: llm-serving
spec:
  scaleTargetRef:
    apiVersion: nvidia.com/v1alpha1
    kind: DynamoGraphDeploymentScalingAdapter
    name: my-llm-deployment-frontend
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

### HPA with Custom Metrics

To use LLM-specific metrics, you need [Prometheus Adapter](https://github.com/kubernetes-sigs/prometheus-adapter) or similar:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: decode-hpa
  namespace: llm-serving
spec:
  scaleTargetRef:
    apiVersion: nvidia.com/v1alpha1
    kind: DynamoGraphDeploymentScalingAdapter
    name: my-llm-deployment-decode
  minReplicas: 2
  maxReplicas: 20
  metrics:
  # Scale based on KV cache utilization
  - type: Pods
    pods:
      metric:
        name: vllm_gpu_cache_usage_perc
      target:
        type: AverageValue
        averageValue: "70"
  # Also consider queue depth
  - type: External
    external:
      metric:
        name: vllm_num_requests_waiting
        selector:
          matchLabels:
            service: decode
      target:
        type: AverageValue
        averageValue: "5"
```

### HPA with Dynamo Metrics

Dynamo exports several metrics useful for autoscaling. These are available at the `/metrics` endpoint on each frontend pod.

> **See also**: For a complete list of all Dynamo metrics, see the [Metrics Reference](../observability/metrics.md). For Prometheus and Grafana setup, see the [Prometheus and Grafana Setup Guide](../observability/prometheus-grafana.md).

#### Available Dynamo Metrics

| Metric | Type | Description | Good for scaling |
|--------|------|-------------|------------------|
| `dynamo_frontend_queued_requests` | Gauge | Requests waiting in HTTP queue | ✅ Prefill |
| `dynamo_frontend_inflight_requests` | Gauge | Concurrent requests to engine | ✅ All services |
| `dynamo_frontend_time_to_first_token_seconds` | Histogram | TTFT latency | ✅ Prefill |
| `dynamo_frontend_inter_token_latency_seconds` | Histogram | ITL latency | ✅ Decode |
| `dynamo_frontend_request_duration_seconds` | Histogram | Total request duration | ⚠️ General |
| `kvstats_gpu_cache_usage_percent` | Gauge | GPU KV cache usage (0-1) | ✅ Decode |

#### Metric Labels

Dynamo metrics include these labels for filtering:

| Label | Description | Example |
|-------|-------------|---------|
| `dynamo_namespace` | Unique DGD identifier (`{k8s-namespace}-{dgd-name}`) | `llm-serving-my-deployment` |
| `model` | Model being served | `meta-llama/Llama-3-70B` |

> **Note**: When you have multiple DGDs in the same namespace, use `dynamo_namespace` to filter metrics for a specific DGD.

#### Example: Scale Prefill Based on TTFT

This example scales **Prefill workers** when Time To First Token (TTFT) exceeds 500ms. Note that TTFT is measured at the Frontend, but reflects Prefill performance.

First, configure Prometheus Adapter to expose the TTFT metric:

```yaml
# Prometheus Adapter ConfigMap (add to your existing config)
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-adapter-config
  namespace: monitoring
data:
  config.yaml: |
    rules:
    # TTFT p95 from frontend - used to scale prefill
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

Then create the HPA targeting the Prefill adapter:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: prefill-ttft-hpa
  namespace: llm-serving
spec:
  scaleTargetRef:
    apiVersion: nvidia.com/v1alpha1
    kind: DynamoGraphDeploymentScalingAdapter
    name: my-llm-deployment-prefill      # ← Target: PREFILL adapter
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: External
    external:
      metric:
        name: dynamo_ttft_p95_seconds
        selector:
          matchLabels:
            # Filter by DGD using dynamo_namespace label
            dynamo_namespace: "llm-serving-my-llm-deployment"
      target:
        type: Value
        value: "500m"  # Scale up when TTFT p95 > 500ms
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300    # Wait 5 min before scaling down
      policies:
      - type: Pods
        value: 1
        periodSeconds: 60
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
3. HPA monitors this metric for your specific DGD
4. When TTFT p95 > 500ms, HPA scales up the Prefill adapter
5. Adapter controller syncs the replica count to the DGD
6. More Prefill workers are created, reducing TTFT

#### Example: Scale Decode Based on Queue Depth

```yaml
# Prometheus Adapter rule
rules:
- seriesQuery: 'dynamo_frontend_queued_requests{namespace!=""}'
  resources:
    overrides:
      namespace: {resource: "namespace"}
  name:
    as: "dynamo_queued_requests"
  metricsQuery: |
    sum(<<.Series>>{<<.LabelMatchers>>}) by (namespace, dynamo_namespace)

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: decode-queue-hpa
  namespace: llm-serving
spec:
  scaleTargetRef:
    apiVersion: nvidia.com/v1alpha1
    kind: DynamoGraphDeploymentScalingAdapter
    name: my-llm-deployment-decode
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: External
    external:
      metric:
        name: dynamo_queued_requests
        selector:
          matchLabels:
            dynamo_namespace: "llm-serving-my-llm-deployment"
      target:
        type: Value
        value: "10"  # Scale up when queue > 10 requests
```

## Autoscaling with KEDA

KEDA extends Kubernetes with event-driven autoscaling, supporting 50+ scalers.

**When to use KEDA:**
- You need event-driven scaling (e.g., queue depth)
- You want to scale to zero when idle
- You need complex scaling triggers

### KEDA with Prometheus

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: prefill-scaledobject
  namespace: llm-serving
spec:
  scaleTargetRef:
    apiVersion: nvidia.com/v1alpha1
    kind: DynamoGraphDeploymentScalingAdapter
    name: my-llm-deployment-prefill
  minReplicaCount: 1
  maxReplicaCount: 15
  pollingInterval: 15
  cooldownPeriod: 120
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus-server.monitoring.svc.cluster.local:9090
      metricName: vllm_queue_depth
      query: |
        sum(vllm_num_requests_waiting{
          namespace="llm-serving",
          dynamo_graph_deployment="my-llm-deployment",
          service="prefill"
        })
      threshold: "10"
```

## Mixed Autoscaling

You can use different autoscaling strategies for different services:

```yaml
# DGD with three services
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-llm-deployment
  namespace: llm-serving
spec:
  services:
    frontend:
      replicas: 2        # Managed by HPA (CPU-based)
    prefill:
      replicas: 3        # Managed by KEDA (queue-based)
    decode:
      replicas: 6        # Managed by Planner (LLM-optimized)

---
# HPA for Frontend
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: frontend-hpa
  namespace: llm-serving
spec:
  scaleTargetRef:
    apiVersion: nvidia.com/v1alpha1
    kind: DynamoGraphDeploymentScalingAdapter
    name: my-llm-deployment-frontend
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70

---
# KEDA for Prefill
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: prefill-scaledobject
  namespace: llm-serving
spec:
  scaleTargetRef:
    apiVersion: nvidia.com/v1alpha1
    kind: DynamoGraphDeploymentScalingAdapter
    name: my-llm-deployment-prefill
  minReplicaCount: 1
  maxReplicaCount: 12
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus-server.monitoring.svc.cluster.local:9090
      query: sum(vllm_num_requests_waiting{service="prefill"})
      threshold: "10"

# Decode is managed by Planner (no additional config needed)
```

## Manual Scaling

You can manually scale a service by patching the adapter:

```bash
kubectl patch dgdsa my-llm-deployment-decode -n llm-serving \
  --type='json' -p='[{"op": "replace", "path": "/spec/replicas", "value": 10}]'
```

> **Note**: If an autoscaler is managing the adapter, your change will be overwritten on the next evaluation cycle.

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
| Decode | KV cache utilization, ITL | `kvstats_gpu_cache_usage_percent`, `dynamo_frontend_inter_token_latency_seconds` |

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
kubectl describe dgd my-llm-deployment -n llm-serving

# Check operator logs
kubectl logs -n dynamo-system deployment/dynamo-operator
```

### Scaling Not Working

```bash
# Check adapter status
kubectl describe dgdsa my-llm-deployment-decode -n llm-serving

# Check HPA status
kubectl describe hpa decode-hpa -n llm-serving

# Verify metrics are available in Kubernetes metrics API
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1
kubectl get --raw /apis/external.metrics.k8s.io/v1beta1
```

### Metrics Not Available

If HPA shows `<unknown>` for metrics:

```bash
# Check if Dynamo metrics are being scraped
kubectl port-forward -n llm-serving pod/<frontend-pod> 8000:8000
curl http://localhost:8000/metrics | grep dynamo_frontend

# Example output:
# dynamo_frontend_queued_requests{model="meta-llama/Llama-3-70B"} 2
# dynamo_frontend_inflight_requests{model="meta-llama/Llama-3-70B"} 5

# Verify Prometheus is scraping the metrics
kubectl port-forward -n monitoring svc/prometheus-server 9090:9090
# Then query: dynamo_frontend_time_to_first_token_seconds_bucket

# Check Prometheus Adapter logs
kubectl logs -n monitoring deployment/prometheus-adapter
```

### Rapid Scaling Up and Down

If you see unstable scaling:

1. Check if multiple autoscalers are targeting the same adapter
2. Increase stabilization window in HPA behavior
3. Increase cooldown period in KEDA ScaledObject

## References

- [Kubernetes HPA Documentation](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [KEDA Documentation](https://keda.sh/)
- [Prometheus Adapter](https://github.com/kubernetes-sigs/prometheus-adapter)
- [Planner Documentation](../planner/sla_planner.md)
- [Dynamo Metrics Reference](../observability/metrics.md)
- [Prometheus and Grafana Setup](../observability/prometheus-grafana.md)

