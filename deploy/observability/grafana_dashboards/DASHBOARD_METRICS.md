# Grafana Dashboard Metrics Documentation

This document explains where each panel in the `disagg-dashboard.json` gets its data and how it's displayed.

## Dashboard Organization

The dashboard is organized in **logical request flow order** (21 panels across 6 rows):

**Row 1: Frontend Health** (User-facing metrics - y=0)
- Frontend Requests/Sec (x=0), Avg TTFT (x=8), Avg Request Duration (x=16)

**Row 2: Frontend Details** (y=8)
- Avg Inter-Token Latency (x=0), Avg ISL/OSL (x=8), **Queued Requests** ⭐ (x=16)

**Row 3: Prefill Workers** (The typical bottleneck! - y=16)
- Prefill Worker Processing Time ⭐ (x=0), Prefill Worker Throughput (x=8), Component Latency Comparison (x=16)

**Row 4: Decode Workers** (y=24)
- Request Throughput (x=0), Avg Request Duration (x=8), KV Cache Utilization (%) (x=16)

**Row 5: KV Cache + GPU** (y=32)
- KV Cache Blocks (Active/Total) ⭐ (x=0), GPU Compute Utilization (x=8), GPU Memory Used (x=16)

**Row 6: NIXL Transfer Metrics** (y=40)
- GPU Memory Bandwidth (x=0), NVLink Bandwidth (GB/s) (x=8), Worker CPU Usage (x=16)

**Row 7: Node + Worker** (y=48)
- Node CPU Utilization (x=0), Worker Request Throughput (x=8), Worker Data Transfer (x=16)

⭐ = Key metrics for diagnosing TTFT bottlenecks

## Metric Sources

### Frontend Metrics (from Frontend Pod)
These metrics come from the `dynamo_frontend_*` namespace and are collected from the frontend deployment pod.

| Panel | Metric | Formula | Description |
|-------|--------|---------|-------------|
| **Frontend Requests / Sec** | `dynamo_frontend_requests_total` | `rate(...[30s])` | Rate of requests per second hitting the frontend, broken down by request_type and status |
| **Frontend Avg Time to First Token** ⭐ | `dynamo_frontend_time_to_first_token_seconds_{sum,count}` | `1000 * (rate(sum[5m]) / rate(count[5m]))` | Average time (in ms) from request arrival to first token over the last 5 minutes. Includes queue wait, prefill compute, and NIXL transfer time. **The primary performance metric** |
| **Frontend Avg Request Duration** | `dynamo_frontend_request_duration_seconds_{sum,count}` | `1000 * (rate(sum[5m]) / rate(count[5m]))` | Total end-to-end request duration in milliseconds over the last 5 minutes |
| **Frontend Avg Inter-Token Latency** | `dynamo_frontend_inter_token_latency_seconds_{sum,count}` | `1000 * (rate(sum[5m]) / rate(count[5m]))` | Average time (in ms) between token generations during decode phase over the last 5 minutes |
| **Frontend Avg Input/Output Sequence Length** | `dynamo_frontend_input_sequence_tokens_{sum,count}` & `dynamo_frontend_output_sequence_tokens_{sum,count}` | `rate(sum[5m]) / rate(count[5m])` for each | Average input prompt length (ISL) and output generation length (OSL) in tokens over the last 5 minutes |
| **Frontend Queued Requests** ⭐⭐⭐ | `dynamo_frontend_queued_requests` | Raw value | Number of requests waiting in queue. **THE key metric for diagnosing worker saturation.** High values (>10) indicate workers cannot keep up with load. Yellow threshold at 10, red at 50 |

### GPU Metrics (from DCGM Exporter)
These metrics come from the DCGM (Data Center GPU Manager) exporter running as a DaemonSet in the `gpu-operator` namespace. DCGM collects hardware-level GPU metrics.

| Panel | Metric | Formula | Description |
|-------|--------|---------|-------------|
| **GPU Compute Utilization** | `DCGM_FI_DEV_GPU_UTIL` | Raw value | GPU compute utilization percentage (0-100) for each GPU. Prefill workers show high utilization during prefill phase |
| **GPU Memory Bandwidth** | `DCGM_FI_DEV_MEM_COPY_UTIL` | Raw value | GPU memory copy bandwidth utilization percentage (0-100). **Spikes indicate KV cache transfers over NIXL**. On single-node deployments, NIXL uses CUDA IPC (GPU→Host→Host→GPU) not direct GPU-to-GPU. Yellow threshold at 60%, red at 80% |
| **NVLink Bandwidth (GB/s)** | `DCGM_FI_PROF_NVLINK_TX_BYTES` & `DCGM_FI_PROF_NVLINK_RX_BYTES` | `(rate(TX_BYTES[1m]) + rate(RX_BYTES[1m])) / 1e9` | NVLink transfer bandwidth in GB/s (rate of change) per GPU, measured from DCGM profiling metrics. Shows total bidirectional bandwidth (TX + RX). This includes intra-pod TP communication (TP=2 for prefill, TP=4 for decode). Low bandwidth (<1 GB/s) indicates inter-pod NIXL KV cache transfers may be using host memory copies instead of direct NVLink/GPUDirect. Yellow threshold at 5 GB/s, red at 10 GB/s |
| **GPU Memory Used** | `DCGM_FI_DEV_FB_USED` | `value / 1024` | GPU framebuffer memory used in GB. Prefill workers allocate KV blocks on decode workers via NIXL |

### Prefill Worker Metrics (from Prefill Worker Pods)
These metrics come from the prefill worker pods' system endpoints (port 9090). They track request processing for prefill operations.

| Panel | Metric | Formula | Description |
|-------|--------|---------|-------------|
| **Prefill Worker Processing Time** | `dynamo_component_request_duration_seconds_{sum,count,bucket}{dynamo_component="prefill",dynamo_endpoint="generate"}` | `1000 * rate(sum[5m]) / rate(count[5m])` for avg, `histogram_quantile(0.99, ...)` for P99 | Average and P99 time (in ms) spent processing prefill requests. **Includes prefill computation AND KV cache transfer over NIXL** |
| **Prefill Worker Throughput** | `dynamo_component_requests_total{dynamo_component="prefill",dynamo_endpoint="generate"}` | `rate(...[5m])` | Rate of prefill requests being processed in requests/second |

### Decode Worker Metrics (from Decode Worker Pods)
These metrics come from the decode worker pods' system endpoints (port 9090). In disaggregated mode, decode workers receive KV cache from prefill workers and perform token generation.

| Panel | Metric | Formula | Description |
|-------|--------|---------|-------------|
| **Component Latency - Prefill vs Decode** | `dynamo_component_request_duration_seconds_{sum,count}{dynamo_component="prefill",dynamo_endpoint="generate"}` & `{dynamo_component="backend",dynamo_endpoint="generate"}` | `rate(sum[5m]) / rate(count[5m])` | Average request duration for prefill workers (includes NIXL transfer) vs decode workers (entire decode session for all output tokens) over the last 5 minutes. **Note**: Decode worker latency measures the FULL decode session duration, not just time to first token. Only shows `generate` endpoint (filters out `clear_kv_blocks` maintenance operations) |
| **Decode Worker - Request Throughput** | `dynamo_component_requests_total{dynamo_component="backend"}` | `rate(...[5m])` | Rate of requests processed by decode workers in requests/second |
| **Decode Worker - Avg Request Duration** | `dynamo_component_request_duration_seconds_{sum,count}{dynamo_component="backend"}` | `rate(sum[5m]) / rate(count[5m])` | Average time decode workers spend processing requests (decode phase only) over the last 5 minutes |
| **KV Cache Utilization** | `dynamo_component_kvstats_gpu_cache_usage_percent` | Raw value (0-100%) | GPU memory utilization for KV cache storage of active requests. High values (>90%) indicate workers are at capacity and requests are queueing. **Note**: Only available for decode workers - prefill workers in disaggregated mode don't expose this metric. Monitor Prefill Worker Processing Time instead for prefill capacity |
| **KV Cache Blocks (Active/Total)** ⭐ | `dynamo_component_kvstats_active_blocks` & `dynamo_component_kvstats_total_blocks` | Raw values | Number of KV cache blocks in use vs total available for decode workers. When active approaches total, decode workers are at capacity. Shows numeric values (e.g., 2048/5297). **Note**: Only for decode workers |

### CPU Metrics (from cAdvisor and Node Exporter)
These metrics come from Kubernetes cAdvisor (container metrics) and Node Exporter (node-level metrics). CPU bottlenecks can impact prefill/decode performance.

| Panel | Metric | Formula | Description |
|-------|--------|---------|-------------|
| **Worker CPU Usage** | `container_cpu_usage_seconds_total{namespace="robert",pod=~".*worker.*",container="main"}` | `rate(...[5m])` | CPU cores used by worker pods. Value shows actual CPU consumption (e.g., 2.5 = 2.5 cores). Yellow at 30 cores, red at 50 cores |
| **Node CPU Utilization** | `node_cpu_seconds_total{mode="idle"}` | `100 - (avg(rate(idle)) * 100)` | Overall node CPU utilization percentage. Shows aggregate CPU usage across all cores |

### Worker Metrics (from Worker Pods)
These metrics track request processing across all worker pods (prefill and decode).

| Panel | Metric | Formula | Description |
|-------|--------|---------|-------------|
| **Worker Request Throughput** | `dynamo_component_requests_total{dynamo_endpoint="generate"}` | `rate(...[5m])` | Requests per second processed by each worker, broken down by component type (prefill, backend). Shows overall system throughput |
| **Worker Data Transfer** | `dynamo_component_request_bytes_total` & `dynamo_component_response_bytes_total` | `rate(...[5m])` | Bytes per second transferred in requests (IN) and responses (OUT). Shows data throughput across worker pods |

## Metric Label Filters

### Component Name Filtering
- **Prefill workers**: `dynamo_component="prefill"`
- **Decode workers**: `dynamo_component="backend"`
- **All workers**: Filter by `dynamo_endpoint="generate"` to exclude maintenance operations like `clear_kv_blocks`

### Important Labels
- `pod`: Specific pod name (e.g., `llama3-70b-disagg-sn-0-vllmprefillworker-hrnt5`)
- `namespace`: Kubernetes namespace (e.g., `robert`)
- `dynamo_component`: Component type (`prefill`, `backend`, `frontend`)
- `dynamo_endpoint`: Endpoint name (`generate`, `clear_kv_blocks`)
- `gpu`: GPU index (0-7 for DCGM metrics)
- `Hostname`: Node hostname (for DCGM metrics)

## Metric Collection Architecture

```text
┌─────────────────┐
│  Frontend Pod   │ ──► dynamo_frontend_* metrics (HTTP port)
└─────────────────┘

┌─────────────────┐
│ Prefill Worker  │ ──► dynamo_component_* metrics (system port 9090)
│     Pods        │     ├─ dynamo_component_request_* (request stats)
└─────────────────┘     └─ dynamo_component_*_bytes_total (data transfer)
                        └─ container_cpu_* metrics (cAdvisor)

┌─────────────────┐
│ Decode Worker   │ ──► dynamo_component_* metrics (system port 9090)
│     Pods        │     └─ dynamo_component_request_* (component stats)
└─────────────────┘     └─ container_cpu_* metrics (cAdvisor)

┌─────────────────┐
│ DCGM Exporter   │ ──► DCGM_FI_DEV_* metrics (GPU metrics port)
│   DaemonSet     │     ├─ GPU compute utilization
│ (gpu-operator)  │     ├─ GPU memory bandwidth (NIXL indicator)
└─────────────────┘     ├─ GPU memory usage
                        └─ GPU temperature

┌─────────────────┐
│ Node Exporter   │ ──► node_* metrics (node metrics port)
│   DaemonSet     │     ├─ node_cpu_seconds_total (CPU by mode)
│   (monitoring)  │     ├─ node_load1/5/15 (load average)
└─────────────────┘     └─ node_memory_* (memory stats)

┌─────────────────┐
│    cAdvisor     │ ──► container_* metrics (built into kubelet)
│   (kubelet)     │     ├─ container_cpu_usage_seconds_total
└─────────────────┘     ├─ container_cpu_cfs_throttled_periods_total
                        └─ container_memory_*

           ▼
   ┌─────────────────┐
   │  Prometheus     │ ◄─── ServiceMonitor for DCGM & Node Exporter
   │  (monitoring)   │ ◄─── PodMonitor for Dynamo workers
   └─────────────────┘ ◄─── Scrapes cAdvisor from kubelet
           ▼
   ┌─────────────────┐
   │    Grafana      │
   │  (monitoring)   │
   └─────────────────┘
```

## PodMonitor Configuration

The Dynamo operator automatically creates PodMonitors for metrics-enabled deployments:
- **Label**: `nvidia.com/metrics-enabled: "true"` on all worker pods
- **Endpoint**: System port (9090) with path `/metrics`
- **Namespace**: Pods in any namespace are discovered (via `podMonitorNamespaceSelector={}`)

To opt-out a deployment:
```yaml
apiVersion: nvidia.com/v1
kind: DynamoGraphDeployment
metadata:
  annotations:
    nvidia.com/enable-metrics: "false"
```

## DCGM ServiceMonitor Configuration

The DCGM ServiceMonitor must be manually created (see `dcgm-servicemonitor.yaml`):
- **Namespace**: `gpu-operator` (where DCGM exporter runs)
- **Label**: `release: prometheus` (required for Prometheus discovery)
- **Selector**: `app: nvidia-dcgm-exporter`
- **Endpoint**: `gpu-metrics` port with path `/metrics`

## Troubleshooting

### No metrics showing up:
1. Check PodMonitor exists: `kubectl get podmonitor -A`
2. Check pods have metrics label: `kubectl get pods -n <namespace> -l nvidia.com/metrics-enabled=true`
3. Check Prometheus targets: Visit Prometheus UI → Status → Targets

### DCGM metrics missing:
1. Check DCGM exporter running: `kubectl get daemonset -A | grep dcgm-exporter`
2. Check ServiceMonitor exists: `kubectl get servicemonitor -n gpu-operator`
3. Verify `release: prometheus` label on ServiceMonitor

### Prefill queue metrics showing zero:
- These metrics only populate when **remote prefill** requests are processed
- In local-only mode, decode workers handle prefill themselves (no queue)
- Check deployment mode and request routing configuration

### KV Cache metrics only showing decode workers:
**Important Limitation**: In disaggregated mode, prefill workers (`--disaggregation-mode prefill`) do NOT expose `dynamo_component_kvstats_*` metrics. Only decode workers expose these.

**Why this happens:**
- Prefill workers transfer KV cache to decode workers via NIXL
- They don't maintain long-term KV cache state
- Only decode workers track KV cache utilization metrics

**How to diagnose prefill worker capacity bottlenecks:**
1. **Check worker logs** for KV cache size at startup:
   ```bash
   kubectl logs -n <namespace> <prefill-worker-pod> | grep "GPU KV cache size"
   # Example output: GPU KV cache size: 254,336 tokens
   ```

2. **Calculate maximum concurrent requests**:
   ```text
   Max Concurrency = KV Cache Size ÷ Tokens Per Request
   # For ISL=8192: 254,336 ÷ 131,072 = 1.94 requests per prefill worker
   ```

3. **Monitor indirect indicators in dashboard**:
   - **Prefill Worker Processing Time**: High avg (>5s) or P99 (>10s) indicates saturation
   - **Frontend Avg TTFT**: If much higher than Prefill Processing Time, indicates queueing
   - **Gap = TTFT - Prefill Processing Time** = Queue wait time

4. **Performance signature of prefill KV cache bottleneck**:
   - Min TTFT is low (1-3s) - proves system CAN be fast
   - Avg/Max TTFT is very high (>30s) - proves requests are queueing
   - Large variance (Max ÷ Min > 20×) - signature of queueing behavior
   - This variance pattern is impossible if the issue was compute-bound - it's the mathematical signature of KV cache capacity bottleneck

### High CPU usage:
1. **Worker CPU Usage showing high values (>30 cores)**:
   - Check if workers have sufficient CPU limits configured
   - May indicate CPU-bound operations (tokenization, scheduling)
   - Compare against GPU utilization - CPU should not be the bottleneck

2. **What's normal?**:
   - vLLM workers use CPU for:
     - Request scheduling and batching
     - Tokenization (input/output processing)
     - KV cache management
     - TCP/gRPC communication (request plane)
   - Expect moderate CPU usage (5-20 cores per worker)
   - GPU compute should dominate, not CPU

## Dashboard Variables

The dashboard uses two template variables for flexibility:

### Datasource Variable
- **Variable**: `${datasource}`
- **Type**: `datasource`
- **Query**: `prometheus`
- **Auto-selects**: Default Prometheus instance
- **Purpose**: Allows the dashboard to automatically connect to your Prometheus instance without hardcoding UIDs

### Namespace Variable
- **Variable**: `${namespace}`
- **Type**: `query`
- **Query**: `label_values(dynamo_frontend_requests_total, namespace)`
- **Purpose**: Allows filtering metrics by Kubernetes namespace (e.g., "robert", "default")
- **Auto-populated**: Dynamically discovers namespaces from frontend pods

**Usage**: All dashboard queries filter by `namespace="$namespace"` to show metrics for the selected deployment. You can switch between different Dynamo deployments in different namespaces using the namespace dropdown at the top of the dashboard.
