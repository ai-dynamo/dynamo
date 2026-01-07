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
4. GPU driver restart for recoverable XIDs

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    gpu-fault-injector-kernel Pod                 │
│                    (Privileged DaemonSet)                        │
│                                                                 │
│  ┌─────────────┐     ┌─────────────────────────────────────┐   │
│  │  agent.py   │────▶│  gpu_xid_injector.py                │   │
│  │  FastAPI    │     │                                     │   │
│  │  :8083      │     │  nsenter --target 1 --mount --uts   │   │
│  └─────────────┘     │          --ipc --pid --             │   │
│                      │  sh -c "echo '<3>NVRM: Xid...'      │   │
│                      │         > /dev/kmsg"                │   │
│                      └─────────────────────────────────────┘   │
│                                      │                         │
│                                      ▼                         │
│                      ┌─────────────────────────────────────┐   │
│                      │  Host /dev/kmsg                     │   │
│                      │  (via hostPID + nsenter)            │   │
│                      └─────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NVSentinel                                    │
│  syslog-health-monitor ──▶ Detects XID ──▶ Cordons node         │
│  node-drainer ──▶ Evicts pods after timeout                     │
└─────────────────────────────────────────────────────────────────┘
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

## Deployment

The agent runs as a privileged DaemonSet on GPU nodes:

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

**Required host mounts:**
- `/sys/kernel/debug`
- `/sys/kernel/btf`
- `/proc` (as `/host/proc`)
- `/dev`
- `/sys/bus/pci`

## GPU Discovery

The agent discovers GPUs by reading `/host/proc/driver/nvidia/gpus/`:

```
/host/proc/driver/nvidia/gpus/
├── 0001:00:00.0/information  → Device Minor: 0 (GPU 0)
├── 0002:00:00.0/information  → Device Minor: 1 (GPU 1)
```

This works without `nvidia-smi` and handles extended PCI addresses (Azure VMs).

## Usage Example

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

After injection, verify NVSentinel detected it:

```bash
# Check kernel messages (on node)
dmesg | grep -i "xid"

# Check NVSentinel logs
kubectl logs -n nvsentinel -l app.kubernetes.io/name=syslog-health-monitor --tail=50

# Check node cordoned
kubectl get nodes -o wide | grep -i schedulingdisabled
```

## Troubleshooting

### Agent not starting
- Check GPU node has `nvidia.com/gpu.present=true` label
- Verify tolerations for GPU node taints

### XID not detected by NVSentinel
- Verify NVSentinel's `syslog-health-monitor` is running
- Check `/var/log/messages` or `/var/log/syslog` on host
- Ensure XID type is in NVSentinel's monitored list

### Permission denied
- Agent must be privileged with `hostPID: true`
- Verify `runAsUser: 0` in security context

