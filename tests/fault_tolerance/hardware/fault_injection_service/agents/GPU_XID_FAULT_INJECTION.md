# GPU Fault Injector Agent

Privileged DaemonSet that injects XID errors into the kernel message buffer to trigger NVSentinel detection.

## Why This Component

Real GPU XID errors appear in `/dev/kmsg` (kernel messages) which NVSentinel's `syslog-health-monitor` monitors. This agent:
- Writes fake XID messages to host's `/dev/kmsg` via nsenter
- Uses the exact format NVSentinel expects
- Triggers the full fault tolerance workflow

**Recovery method:** NVSentinel handles everything:
1. `syslog-health-monitor` detects XID in kernel logs
2. Node is cordoned by `fault-quarantine-module` (marked unschedulable)
3. `node-drainer` evicts pods after timeout
4. GPU driver restart for recoverable XIDs (`fault-remediation` and `janitor` modules)
5. `fault-remediation` creates a `RebootNode` CR for devastating XIDs
6. `janitor` sends reboot signal to CSP and monitors node recovery
7. `fault-quarantine` uncordons node once health checks pass

## NVSentinel Prerequisites

**Project:** [NVSentinel](https://github.com/NVIDIA/nvsentinel) - GPU health monitoring and fault recovery for Kubernetes.

**Minimum version:** v0.5.0 or later (syslog-health-monitor module required)

**Required modules:**
- `syslog-health-monitor` - Detects XID in kernel logs
- `fault-quarantine-module` - Cordons nodes with faulty GPUs
- `node-drainer` - Evicts pods from cordoned nodes
- `fault-remediation` - Creates remediation CRs (RebootNode) for fault recovery
- `janitor` - Executes node reboots via CSP API and monitors recovery

### Why Janitor is Required

Without the `janitor` module, cordoned nodes **stay cordoned forever**:

```
XID detected -> Node cordoned -> Pods evicted -> ??? (node stays cordoned)
```

The janitor module completes the recovery cycle:

```
XID detected -> Node cordoned -> Pods evicted -> RebootNode CR created
    -> Janitor sends reboot signal to CSP -> Node reboots
    -> Janitor monitors node ready state -> Health checks pass
    -> fault-quarantine uncordons node -> Node schedulable again
```

**How it works:**
1. `fault-remediation` watches for cordon events with `action: FAIL`
2. Creates a `RebootNode` CR: `maintenance-<node>-<event-id>`
3. `janitor` controller picks up the CR
4. Sends reboot signal to CSP (AWS/Azure/GCP/Nebius) via cloud API
5. Monitors node until Kubernetes reports `NodeReady` condition
6. Once ready, `fault-quarantine` sees health checks passing and uncordons

**Configuration for testing:**
```yaml
# values.yaml for NVSentinel Helm chart
node-drainer:
  enabled: true
  userNamespaces:
    - dynamo          # Your test namespace must be listed
    - default
  evictionTimeout: 120  # Reduce from default 300s for faster tests
  drainTimeout: 180

syslog-health-monitor:
  enabled: true
  xidMapping:
    79:   # GPU fell off bus
      action: FAIL
    74:   # NVLink error
      action: FAIL

fault-quarantine-module:
  enabled: true
  cordonOnFail: true

fault-remediation:
  enabled: true
  # Creates RebootNode CRs for FAIL actions

janitor:
  enabled: true
  rebootNode:
    enabled: true
    timeout: 30m  # Max time to wait for node to come back
  # CSP credentials configured via cloud provider (IRSA/Workload Identity/etc.)
```

**Important:** Your test namespace must be in `node-drainer.userNamespaces` for pod eviction to work.

### Janitor CSP Support

The janitor supports these cloud providers for node reboot:

| CSP | Method | Auth | Status |
|-----|--------|------|--------|
| AWS | EC2 RebootInstances API | IRSA | Supported |
| Azure | VM Restart API | Workload Identity | Supported |
| GCP | Compute Engine Reset | Workload Identity | Supported |
| Nebius | Compute Reset | Service Account | Future release |

**Nebius users:** Nebius support is planned for a future NVSentinel release. If you need janitor functionality on Nebius clusters now, use the custom-built package from [ai-dynamo packages](https://github.com/orgs/ai-dynamo/packages) instead of the upstream NVSentinel v0.6.0.

**Manual mode:** If CSP integration is not available, set `janitor.global.manualMode: true`. The janitor will create CRs with `ManualMode` condition, requiring an operator to manually reboot and uncordon.

## How It Works

```
+---------------------------------------------------------------------+
|                    gpu-fault-injector-kernel Pod                     |
|                    (Privileged DaemonSet)                            |
|                                                                      |
|  +-------------+     +-------------------------------------+         |
|  |  agent.py   |---->|  gpu_xid_injector.py                |         |
|  |  FastAPI    |     |                                     |         |
|  |  :8083      |     |  nsenter --target 1 --mount --uts   |         |
|  +-------------+     |          --ipc --pid --             |         |
|                      |  sh -c "echo '<3>NVRM: Xid...'      |         |
|                      |         > /dev/kmsg"                |         |
|                      +-------------------------------------+         |
|                                      |                               |
|                                      v                               |
|                      +-------------------------------------+         |
|                      |  Host /dev/kmsg                     |         |
|                      |  (via hostPID + nsenter)            |         |
|                      +-------------------------------------+         |
+---------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------+
|                    NVSentinel                                        |
|  syslog-health-monitor --> Detects XID --> Cordons node             |
|  node-drainer --> Evicts pods after timeout                         |
+---------------------------------------------------------------------+
```

## Supported XIDs

### Devastating (Always Trigger FAIL)
| XID | Description |
|-----|-------------|
| 79 | GPU fell off bus |
| 74 | NVLink uncorrectable error |
| 48 | Double-bit ECC error |
| 94 | Contained ECC error |
| 95 | Uncontained error |
| 119 | GSP RPC Timeout |
| 120 | GSP Error |
| 140 | ECC unrecovered error |

### Subsystem (May WARN/Escalate)
| XID | Subsystem | Description |
|-----|-----------|-------------|
| 31, 32, 43, 63, 64 | Memory | MMU, PBDMA, page retirement |
| 38, 39, 42 | PCIe | Bus, fabric, replay rate |
| 60, 61, 62 | Thermal | Temperature limits |
| 54, 56, 57 | Power | Power/clock state |
| 13, 45, 69 | Graphics | SM exceptions |

## API Endpoints

### Health Check
```bash
curl http://localhost:8083/health
```

### Inject XID
```bash
# Via central API (recommended)
curl -X POST http://fault-injection-api:8080/api/v1/faults/gpu/inject/xid-79 \
  -H "Content-Type: application/json" \
  -d '{"node_name": "gpu-node-001", "gpu_id": 0}'

# Direct to agent (on specific node)
curl -X POST http://<agent-pod-ip>:8083/inject-xid \
  -H "Content-Type: application/json" \
  -d '{"fault_id": "test-001", "xid_type": 79, "gpu_id": 0}'
```

### List Active Faults
```bash
curl http://localhost:8083/faults
```

## Message Format

The injected message matches NVSentinel's expected pattern:

```
NVRM: NVRM: Xid (PCI:0001:00:00.0): 79, GPU has fallen off the bus
```

**Note:** The duplicate `NVRM:` is intentional - `/dev/kmsg` splits on the first colon.

## Building the Image

**Dockerfile:** [`gpu_fault_injector/Dockerfile`](gpu_fault_injector/Dockerfile)

```bash
# Build for AMD64 (GPU nodes are x86_64)
cd tests/fault_tolerance/hardware/fault_injection_service/agents/gpu_fault_injector

# Local build
docker buildx build --platform linux/amd64 --load \
  -t ghcr.io/nvidia/dynamo/gpu-fault-injector:latest .

# Push to GHCR (requires authentication)
docker push ghcr.io/nvidia/dynamo/gpu-fault-injector:latest
```

**Pre-built images:**
```bash
# GHCR (public)
ghcr.io/nvidia/dynamo/gpu-fault-injector:latest

# Azure Container Registry (internal)
dynamoci.azurecr.io/gpu-fault-injector:latest
```

## Deployment

The agent runs as a privileged DaemonSet on GPU nodes.

**Deployment manifest:** [`../deploy/gpu-fault-injector-kernel.yaml`](../deploy/gpu-fault-injector-kernel.yaml)

```bash
# Deploy to cluster
kubectl apply -f tests/fault_tolerance/hardware/fault_injection_service/deploy/namespace.yaml
kubectl apply -f tests/fault_tolerance/hardware/fault_injection_service/deploy/gpu-fault-injector-kernel.yaml

# Verify DaemonSet is running on GPU nodes
kubectl get pods -n fault-injection-system -l app=gpu-fault-injector-kernel -o wide
```

### Security Context

Configured in the deployment YAML:

```yaml
spec:
  hostPID: true
  hostNetwork: true
  hostIPC: true
  containers:
  - name: gpu-fault-injector
    securityContext:
      privileged: true
      runAsUser: 0
      capabilities:
        add:
        - SYS_ADMIN
        - SYS_PTRACE
        - BPF
        - SYS_MODULE
```

### Required Host Mounts

Configured in [`gpu-fault-injector-kernel.yaml`](../deploy/gpu-fault-injector-kernel.yaml):

| Mount Path | Host Path | Purpose |
|------------|-----------|---------|
| `/sys/kernel/debug` | `/sys/kernel/debug` | Kernel debugging |
| `/sys/kernel/btf` | `/sys/kernel/btf` | BPF type format |
| `/host/proc` | `/proc` | GPU discovery via `/proc/driver/nvidia/gpus/` |
| `/host/dev` | `/dev` | Write to `/dev/kmsg` |
| `/sys/bus/pci` | `/sys/bus/pci` | PCI address lookup |

## GPU Discovery

The agent discovers GPUs by reading `/host/proc/driver/nvidia/gpus/`:

```
/host/proc/driver/nvidia/gpus/
├── 0001:00:00.0/information  -> Device Minor: 0 (GPU 0)
├── 0002:00:00.0/information  -> Device Minor: 1 (GPU 1)
```

This works without `nvidia-smi` and handles extended PCI addresses (Azure VMs).

## Usage Scenario

Once deployed to the cluster, an operator can issue HTTP requests to the fault-injection-api from a machine with kubectl access. The API routes requests to the appropriate agent on the target node.

**From outside the cluster (port-forward):**
```bash
# Start port-forward to fault-injection-api
kubectl port-forward -n fault-injection-system svc/fault-injection-api 8080:8080 &

# Inject XID 79 on a specific node
curl -X POST http://localhost:8080/api/v1/faults/gpu/inject/xid-79 \
  -H "Content-Type: application/json" \
  -d '{"node_name": "gpu-node-001", "gpu_id": 0}'
```

**From inside the cluster (e.g., test pod):**
```python
import httpx

async def inject_xid_79(node_name: str, gpu_id: int = 0):
    """Inject XID 79 via fault injection API."""
    api_url = "http://fault-injection-api.fault-injection-system:8080"

    response = await httpx.post(
        f"{api_url}/api/v1/faults/gpu/inject/xid-79",
        json={
            "node_name": node_name,
            "gpu_id": gpu_id
        },
        timeout=30
    )

    if response.status_code == 200:
        fault_id = response.json()["fault_id"]
        print(f"XID 79 injected: {fault_id}")
        return fault_id
    else:
        print(f"Injection failed: {response.text}")
        return None
```

## Verifying Injection

After injection, verify NVSentinel detected it.

**Architecture:**
```
+--------------+     kubectl exec     +-----------------+
|  Your        |--------------------->|  GPU Node       |
|  Workstation |                      |  (dmesg)        |
+--------------+                      +-----------------+
       |
       | kubectl logs
       v
+---------------------------------------------------------+
|  NVSentinel Namespace                                    |
|  +---------------------+  +-------------------------+   |
|  | syslog-health-      |  | fault-quarantine-       |   |
|  | monitor             |  | module                  |   |
|  | (detects XID)       |  | (cordons node)          |   |
|  +---------------------+  +-------------------------+   |
+---------------------------------------------------------+
```

**Verification commands:**

```bash
# 1. Get the target node name (where injection happened)
TARGET_NODE="gpu-node-001"

# 2. Check kernel messages on the node (exec into any pod on that node)
kubectl get pods -A -o wide | grep $TARGET_NODE | head -1
# Then exec into one of those pods:
kubectl exec -it <pod-on-target-node> -n <namespace> -- dmesg | grep -i "xid"

# 3. Check NVSentinel syslog-health-monitor logs
kubectl logs -n nvsentinel -l app.kubernetes.io/name=syslog-health-monitor --tail=100 | grep -i "xid\|fault\|cordon"

# 4. Check node is cordoned
kubectl get nodes | grep -i "schedulingdisabled"
kubectl describe node $TARGET_NODE | grep -A5 "Taints:"

# 5. Check fault-quarantine-module logs
kubectl logs -n nvsentinel -l app.kubernetes.io/name=fault-quarantine-module --tail=50

# 6. Check node-drainer logs (for eviction status)
kubectl logs -n nvsentinel -l app.kubernetes.io/name=node-drainer --tail=50
```

## Troubleshooting

### Agent not starting

```bash
# Check DaemonSet status
kubectl get daemonset -n fault-injection-system gpu-fault-injector-kernel

# Check if GPU nodes have the required label
kubectl get nodes -l nvidia.com/gpu.present=true

# Check pod events
kubectl describe pod -n fault-injection-system -l app=gpu-fault-injector-kernel

# Check pod logs
kubectl logs -n fault-injection-system -l app=gpu-fault-injector-kernel --tail=100
```

### XID not detected by NVSentinel

```bash
# 1. Verify NVSentinel pods are running
kubectl get pods -n nvsentinel

# 2. Check syslog-health-monitor is monitoring XID types
kubectl logs -n nvsentinel -l app.kubernetes.io/name=syslog-health-monitor --tail=100

# 3. Verify XID is in NVSentinel's config
kubectl get configmap -n nvsentinel nvsentinel-config -o yaml | grep -A10 "xidMapping"

# 4. Check host syslog (exec into agent pod)
kubectl exec -it -n fault-injection-system <agent-pod> -- cat /host/proc/kmsg | grep -i "xid"

# 5. Ensure test namespace is in node-drainer config
kubectl get configmap -n nvsentinel nvsentinel-config -o yaml | grep -A5 "userNamespaces"
```

### Permission denied

```bash
# Verify agent is running as privileged
kubectl get pod -n fault-injection-system -l app=gpu-fault-injector-kernel -o yaml | grep -A20 "securityContext"

# Check if hostPID is enabled
kubectl get pod -n fault-injection-system -l app=gpu-fault-injector-kernel -o yaml | grep "hostPID"
```

### Node not cordoning

```bash
# Check fault-quarantine-module logs
kubectl logs -n nvsentinel -l app.kubernetes.io/name=fault-quarantine-module --tail=100

# Check if XID action is set to FAIL
kubectl get configmap -n nvsentinel nvsentinel-config -o yaml | grep -B2 -A2 "79"

# Manually check node status
kubectl describe node <target-node> | grep -A10 "Conditions:"
```
