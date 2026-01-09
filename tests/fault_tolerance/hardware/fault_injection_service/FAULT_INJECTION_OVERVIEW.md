# Fault Injection Service

Hardware fault injection framework for testing Dynamo's fault tolerance capabilities.

## Components

| Component | Purpose | Recovery Method |
|-----------|---------|-----------------|
| **Network Partitions** | Test service communication resilience | API removes NetworkPolicy → immediate |
| **GPU Faults (XID)** | Trigger NVSentinel detection & node cordoning | NVSentinel auto-eviction (~2-5 min) |
| **CUDA Shim Library** | Simulate GPU unavailability at API level | Pod reschedule to healthy node (~5-7 min) |

## Why Both CUDA + XID for GPU Fault Simulation?

To realistically simulate a GPU hardware failure, **both components are required**:

| Component | What It Does | Why Needed |
|-----------|--------------|------------|
| **CUDA Shim** | `LD_PRELOAD` intercepts CUDA calls → returns error codes | App sees GPU failure, crashes, triggers pod restart |
| **XID Injection** | Writes to `/dev/kmsg` → appears in syslog | NVSentinel detects XID, cordons node, evicts pods |

**Why both?**
- **XID alone:** NVSentinel cordons node, but app continues using "broken" GPU until pod evicted (delayed failure)
- **CUDA alone:** App fails immediately, but NVSentinel never detects anything (no syslog entry) → no node remediation
- **Together:** Realistic simulation where app fails AND infrastructure responds correctly

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GPU Fault Simulation Flow                        │
│                                                                         │
│  1. CUDA Shim Toggle ON ──▶ CUDA calls return errors ──▶ App crashes    │
│                                                                         │
│  2. XID Injection ──▶ /dev/kmsg ──▶ NVSentinel detects ──▶ Cordon node │
│                                                                         │
│  3. Pod restarts on cordoned node ──▶ Crashes again (CUDA shim)        │
│                                                                         │
│  4. Cleanup DGD spec (remove CUDA artifacts)                            │
│                                                                         │
│  5. node-drainer evicts pods ──▶ New pods on healthy nodes (clean)     │
└─────────────────────────────────────────────────────────────────────────┘
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        fault-injection-system namespace                  │
│                                                                         │
│  ┌─────────────────────────┐     ┌────────────────────────────────┐    │
│  │  fault-injection-api    │────▶│  gpu-fault-injector-kernel     │    │
│  │  (Deployment)           │     │  (DaemonSet - per GPU node)    │    │
│  │                         │     │                                │    │
│  │  - REST API :8080       │     │  - Privileged container        │    │
│  │  - NetworkPolicy mgmt   │     │  - nsenter → /dev/kmsg         │    │
│  │  - ChaosMesh mgmt       │     │  - Port :8083                  │    │
│  │  - Fault tracking       │     │  - Host PID/Network/IPC        │    │
│  └─────────────────────────┘     └────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why this design:**
- **DaemonSet:** XID injection requires kernel-level access (`/dev/kmsg`) on each GPU node
- **Privileged container:** nsenter requires root + host namespaces to write kernel messages
- **hostNetwork:** API needs direct communication with DaemonSet pods

## Container Images

```bash
# From GitHub Container Registry
docker pull ghcr.io/ai-dynamo/fault-injection-api:latest
docker pull ghcr.io/ai-dynamo/gpu-fault-injector:latest
```

## Deployment

```bash
kubectl apply -f deploy/namespace.yaml
kubectl apply -f deploy/api-service.yaml
kubectl apply -f deploy/gpu-fault-injector-kernel.yaml

# Verify
kubectl get pods -n fault-injection-system
```

## Directory Structure

```
fault_injection_service/
├── api_service/           # Central REST API
│   ├── main.py           # FastAPI endpoints
│   └── Dockerfile
├── agents/
│   └── gpu_fault_injector/  # XID injection agent
│       ├── agent.py         # FastAPI agent
│       └── gpu_xid_injector.py  # kmsg writer
├── cuda_fault_injection/  # CUDA shim library
│   ├── cuda_intercept.c   # LD_PRELOAD library
│   └── inject_into_pods.py
├── helpers/               # Python utilities
│   ├── cuda_fault_injection.py
│   ├── k8s_operations.py
│   └── inference_testing.py
└── deploy/                # Kubernetes manifests
    ├── namespace.yaml
    ├── api-service.yaml
    └── gpu-fault-injector-kernel.yaml
```

## Quick Start

### 1. Inject XID Fault (via API)

```bash
# Check available XIDs
curl http://fault-injection-api:8080/api/v1/faults/gpu/xid-types

# Inject XID 79 (GPU fell off bus)
curl -X POST http://fault-injection-api:8080/api/v1/faults/gpu/inject/xid-79 \
  -H "Content-Type: application/json" \
  -d '{"node_name": "gpu-node-001", "gpu_id": 0}'
```

### 2. Inject Network Partition

```bash
# Block worker-to-worker (keep NATS working)
curl -X POST http://fault-injection-api:8080/api/v1/faults/network/inject \
  -H "Content-Type: application/json" \
  -d '{
    "partition_type": "worker_worker",
    "source": "dynamo",
    "target": "dynamo",
    "mode": "networkpolicy",
    "parameters": {
      "namespace": "dynamo",
      "target_pod_prefix": "vllm-decode",
      "block_nats": false
    }
  }'
```

### 3. Use CUDA Shim Library

```bash
cd cuda_fault_injection
make  # Build library

# Inject into deployment
python inject_into_pods.py \
  --deployment vllm-v1-disagg-router \
  --namespace dynamo \
  --patch-deployment \
  --xid-type 79 \
  --passthrough  # Load library but disable faults initially
```

## Important Notes

### NATS Connectivity
- **Default:** `block_nats: false` (NATS traffic allowed)
- Only set `block_nats: true` for explicit NATS partition tests
- Blocking NATS breaks most Dynamo coordination

### Prerequisites
- NVSentinel with `syslog-health-monitor` and `node-drainer`
- `node-drainer` configured with test namespace in `userNamespaces`
- ChaosMesh (optional, for packet loss/delay tests)

### Supported XIDs

| XID | Name | Severity |
|-----|------|----------|
| 79 | GPU fell off bus | Critical |
| 48 | Double-bit ECC error | Critical |
| 74 | NVLink error | High |
| 94 | Contained ECC error | Medium |
| 95 | Uncontained error | Critical |
| 119 | GSP RPC Timeout | High |
| 120 | GSP Error | High |

## See Also

- `NETWORK_PARTITIONS.md` - Network partition injection details
- `cuda_fault_injection/CUDA_SHIM_LIBRARY.md` - CUDA shim library details
- `agents/GPU_XID_FAULT_INJECTION.md` - XID injection agent details
- `../../EPHEMERAL_TESTING_CI.md` - Ephemeral testing framework & CI guide
- `../../deploy/test_hw_faults.py` - Integration test example

