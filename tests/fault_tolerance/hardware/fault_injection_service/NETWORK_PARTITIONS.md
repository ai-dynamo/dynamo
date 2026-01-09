# Network Partition Injection

Test Dynamo's resilience to network failures between services.

## Modes

| Mode | What It Does | Use Case |
|------|--------------|----------|
| **NetworkPolicy** | Complete traffic blocking via K8s native policies | Hard partition testing |
| **ChaosMesh** | Packet loss, delay, bandwidth limits | Degraded network testing |

## Critical: NATS Connectivity

**Default:** `block_nats: false` - NATS traffic is always allowed.

**Why:** Dynamo uses NATS for:
- Service discovery
- KV cache routing
- Worker coordination

Blocking NATS causes cluster-wide failures unrelated to what you're testing. Only set `block_nats: true` for explicit NATS partition scenarios. Every other inter-component scenario should pass since all communication goes through NATS.

## Partition Types

| Type | Description |
|------|-------------|
| `frontend_worker` | Block frontend -> worker communication |
| `worker_nats` | Block worker -> NATS (breaks coordination) |
| `worker_worker` | Block inter-worker communication |
| `custom` | Custom source/target specification |

## NetworkPolicy Mode

Creates K8s NetworkPolicy to block egress from target pods.

### Parameters

```json
{
  "partition_type": "worker_worker",
  "source": "dynamo",
  "target": "dynamo",
  "mode": "networkpolicy",
  "parameters": {
    "namespace": "dynamo",
    "target_pod_prefix": "vllm-decode",
    "block_nats": false,
    "block_all_egress": false,
    "block_ingress": false,
    "block_specific_pods": [{"app": "vllm-prefill"}],
    "allow_namespaces": ["kube-system"]
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `namespace` | string | source | Target namespace |
| `target_pod_prefix` | string | **required** | Pod name prefix to target |
| `block_nats` | bool | `false` | Block NATS traffic |
| `block_all_egress` | bool | `false` | Block all outbound traffic |
| `block_ingress` | bool | `false` | Block all inbound traffic |
| `block_specific_pods` | list | `[]` | Label selectors for pods to block |
| `allow_namespaces` | list | `[]` | Namespaces to always allow |

### Example: Block Decode -> Prefill

```bash
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
      "block_nats": false,
      "block_specific_pods": [{"nvidia.com/dynamo-component-type": "worker"}]
    }
  }'
```

## Response Format

All injection requests (both NetworkPolicy and ChaosMesh modes) return a consistent response:

```json
{
  "fault_id": "net-abc123-1234567890",
  "status": "active",
  "type": "network_partition",
  "mode": "networkpolicy",
  "target_namespace": "dynamo",
  "target_pod_prefix": "vllm-decode",
  "created_at": "2025-01-09T12:00:00Z"
}
```

| Field | Description |
|-------|-------------|
| `fault_id` | Unique identifier for recovery/tracking. Use this to recover the fault later. |
| `status` | Current state: `active`, `recovered`, or `failed` |
| `type` | Always `network_partition` for this endpoint |
| `mode` | `networkpolicy` or `chaos_mesh` depending on request |
| `target_namespace` | Namespace where fault is applied |
| `target_pod_prefix` | Pod prefix being targeted |
| `created_at` | ISO timestamp of injection |

## ChaosMesh Mode

Creates ChaosMesh NetworkChaos resource for advanced network faults.

### Prerequisites

**Project:** [ChaosMesh](https://chaos-mesh.org/) - Cloud-native chaos engineering platform.

**Minimum version:** v2.5.0 or later

**Installation:**
```bash
# Add Helm repo
helm repo add chaos-mesh https://charts.chaos-mesh.org

# Install ChaosMesh
helm install chaos-mesh chaos-mesh/chaos-mesh \
  --namespace chaos-mesh \
  --create-namespace \
  --version 2.6.3 \
  --set chaosDaemon.runtime=containerd \
  --set chaosDaemon.socketPath=/run/containerd/containerd.sock

# Verify installation
kubectl get pods -n chaos-mesh
```

**Configuration recommendations:**
```yaml
# values.yaml for ChaosMesh
chaosDaemon:
  runtime: containerd  # or docker/cri-o depending on cluster
  socketPath: /run/containerd/containerd.sock
controllerManager:
  replicaCount: 1
dashboard:
  enabled: false  # Not needed for API-only usage
```

### Parameters

```json
{
  "partition_type": "worker_worker",
  "source": "dynamo",
  "target": "dynamo",
  "mode": "chaos_mesh",
  "duration": 60,
  "parameters": {
    "namespace": "dynamo",
    "target_pod_prefix": "vllm-decode",
    "packet_loss_percent": 50,
    "delay_ms": 100,
    "delay_jitter_ms": 20,
    "bandwidth_limit": "1mbps",
    "corrupt_percent": 5,
    "duplicate_percent": 10,
    "target_nats": false
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `packet_loss_percent` | int | `0` | % of packets to drop |
| `delay_ms` | int | `0` | Latency to add (ms) |
| `delay_jitter_ms` | int | `0` | Latency variance (ms) |
| `bandwidth_limit` | string | `""` | Bandwidth cap (e.g., "1mbps") |
| `corrupt_percent` | int | `0` | % of packets to corrupt |
| `duplicate_percent` | int | `0` | % of packets to duplicate |
| `target_nats` | bool | `true` | Target NATS pods |
| `target_specific_pods` | list | `[]` | Label selectors for targets |
| `duration` | int | `null` | Auto-cleanup after N seconds |

### Example: 50% Packet Loss

```bash
curl -X POST http://fault-injection-api:8080/api/v1/faults/network/inject \
  -H "Content-Type: application/json" \
  -d '{
    "partition_type": "worker_worker",
    "source": "dynamo",
    "target": "dynamo",
    "mode": "chaos_mesh",
    "duration": 120,
    "parameters": {
      "namespace": "dynamo",
      "target_pod_prefix": "vllm-decode",
      "packet_loss_percent": 50,
      "target_nats": false
    }
  }'
```

### Example: High Latency Simulation

```bash
curl -X POST http://fault-injection-api:8080/api/v1/faults/network/inject \
  -H "Content-Type: application/json" \
  -d '{
    "partition_type": "worker_worker",
    "source": "dynamo",
    "target": "dynamo",
    "mode": "chaos_mesh",
    "parameters": {
      "namespace": "dynamo",
      "target_pod_prefix": "vllm-prefill",
      "delay_ms": 500,
      "delay_jitter_ms": 100,
      "target_nats": false
    }
  }'
```

## Recovery

### Manual Recovery

```bash
# Recover specific fault (use fault_id from injection response)
curl -X POST http://fault-injection-api:8080/api/v1/faults/{fault_id}/recover

# Cleanup all orphaned NetworkPolicies
curl -X POST "http://fault-injection-api:8080/api/v1/faults/network/cleanup?namespace=dynamo"
```

### Auto-Recovery

ChaosMesh faults with `duration` set auto-cleanup after timeout.

## Python Usage

The following helper functions can be used in test code:

```python
import httpx

async def inject_network_partition(
    namespace: str,
    target_pod_prefix: str,
    packet_loss: int = 0,
    delay_ms: int = 0
) -> str:
    """Inject network fault, return fault_id for later recovery."""

    api = "http://fault-injection-api.fault-injection-system:8080"

    params = {
        "namespace": namespace,
        "target_pod_prefix": target_pod_prefix,
        "target_nats": False,  # Keep NATS working
    }

    if packet_loss > 0:
        params["packet_loss_percent"] = packet_loss
        mode = "chaos_mesh"
    elif delay_ms > 0:
        params["delay_ms"] = delay_ms
        mode = "chaos_mesh"
    else:
        mode = "networkpolicy"

    resp = await httpx.post(f"{api}/api/v1/faults/network/inject", json={
        "partition_type": "worker_worker",
        "source": namespace,
        "target": namespace,
        "mode": mode,
        "parameters": params
    })

    return resp.json()["fault_id"]

async def recover_partition(fault_id: str):
    """Remove network partition using the fault_id from injection."""
    api = "http://fault-injection-api.fault-injection-system:8080"
    await httpx.post(f"{api}/api/v1/faults/{fault_id}/recover")
```

## Common Test Patterns

These examples use the `inject_network_partition` and `recover_partition` helper functions defined above.

### Test: Inference During Packet Loss

```python
async def test_inference_with_packet_loss():
    # Inject 30% packet loss on decode workers
    fault_id = await inject_network_partition(
        namespace="dynamo",
        target_pod_prefix="vllm-decode",
        packet_loss=30
    )

    try:
        # Inference should still work (with retries)
        result = await send_inference_request()
        assert result["success"]
    finally:
        # Always recover the partition
        await recover_partition(fault_id)
```

### Test: Worker Isolation

```python
async def test_worker_isolation():
    # Block decode workers from talking to each other (hard partition)
    fault_id = await inject_network_partition(
        namespace="dynamo",
        target_pod_prefix="vllm-decode"
    )

    try:
        # Test routing still works via frontend
        assert await send_inference_request()
    finally:
        await recover_partition(fault_id)
```

## Troubleshooting

### NetworkPolicy not taking effect

```bash
# 1. Verify CNI supports NetworkPolicy (Calico, Cilium, etc.)
kubectl get pods -n kube-system | grep -E "calico|cilium|weave"

# 2. Check policy was created
kubectl get networkpolicy -n dynamo

# 3. Describe the policy to see rules
kubectl describe networkpolicy -n dynamo -l managed-by=fault-injection-api

# 4. Check pod labels match policy selector
kubectl get pods -n dynamo --show-labels | grep vllm-decode

# 5. Test connectivity from affected pod
kubectl exec -it -n dynamo <vllm-decode-pod> -- curl -v http://<target-pod>:8080/health
```

### ChaosMesh not working

```bash
# 1. Verify ChaosMesh is installed
kubectl get pods -n chaos-mesh

# 2. Check ChaosMesh controller logs
kubectl logs -n chaos-mesh -l app.kubernetes.io/component=controller-manager --tail=100

# 3. Check NetworkChaos CR was created
kubectl get networkchaos -n dynamo

# 4. Describe the NetworkChaos resource
kubectl describe networkchaos -n dynamo

# 5. Check chaos-daemon logs on target node
TARGET_NODE=$(kubectl get pod -n dynamo <target-pod> -o jsonpath='{.spec.nodeName}')
kubectl logs -n chaos-mesh -l app.kubernetes.io/component=chaos-daemon --field-selector spec.nodeName=$TARGET_NODE --tail=50
```

### Cleanup orphaned policies

```bash
# Via API
curl -X POST "http://fault-injection-api:8080/api/v1/faults/network/cleanup?namespace=dynamo"

# Manual NetworkPolicy cleanup
kubectl delete networkpolicy -n dynamo -l managed-by=fault-injection-api

# Manual ChaosMesh cleanup
kubectl delete networkchaos -n dynamo -l managed-by=fault-injection-api

# List all fault-injection-api managed resources
kubectl get networkpolicy,networkchaos -A -l managed-by=fault-injection-api
```

### Verify partition is active

```bash
# Check active faults via API
kubectl port-forward -n fault-injection-system svc/fault-injection-api 8080:8080 &
curl http://localhost:8080/api/v1/faults | jq '.[] | select(.status == "active")'

# Test connectivity from source to target pod
kubectl exec -it -n dynamo <source-pod> -- ping -c 3 <target-pod-ip>
kubectl exec -it -n dynamo <source-pod> -- curl -v --max-time 5 http://<target-pod>:8080/health
```
