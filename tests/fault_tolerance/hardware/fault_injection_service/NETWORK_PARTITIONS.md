# Network Partition Injection

Test Dynamo's resilience to network failures between services.

## Modes

| Mode | What It Does | Use Case |
|------|--------------|----------|
| **NetworkPolicy** | Complete traffic blocking via K8s native policies | Hard partition testing |
| **ChaosMesh** | Packet loss, delay, bandwidth limits | Degraded network testing |

## Critical: NATS Connectivity

**Default:** `block_nats: false` — NATS traffic is always allowed.

**Why:** Dynamo uses NATS for:
- Service discovery
- KV cache routing
- Worker coordination

Blocking NATS causes cluster-wide failures unrelated to what you're testing. Only set `block_nats: true` for explicit NATS partition scenarios. Every other inter-component scenario should pass since all communication goes through NATS.

## Partition Types

| Type | Description |
|------|-------------|
| `frontend_worker` | Block frontend → worker communication |
| `worker_nats` | Block worker → NATS (breaks coordination) |
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

### Example: Block Decode → Prefill

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

## ChaosMesh Mode

Creates ChaosMesh NetworkChaos resource for advanced network faults.

**Prerequisite:** ChaosMesh must be installed in cluster.

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
# Recover specific fault
curl -X POST http://fault-injection-api:8080/api/v1/faults/{fault_id}/recover

# Cleanup all orphaned NetworkPolicies
curl -X POST "http://fault-injection-api:8080/api/v1/faults/network/cleanup?namespace=dynamo"
```

### Auto-Recovery

ChaosMesh faults with `duration` set auto-cleanup after timeout.

## Python Usage

```python
import httpx

async def inject_network_partition(
    namespace: str,
    target_pod_prefix: str,
    packet_loss: int = 0,
    delay_ms: int = 0
) -> str:
    """Inject network fault, return fault_id."""
    
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
    """Remove network partition."""
    api = "http://fault-injection-api.fault-injection-system:8080"
    await httpx.post(f"{api}/api/v1/faults/{fault_id}/recover")
```

## Common Test Patterns

### Test: Inference During Packet Loss

```python
async def test_inference_with_packet_loss():
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
        await recover_partition(fault_id)
```

### Test: Worker Isolation

```python
async def test_worker_isolation():
    # Block decode workers from talking to each other
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
- Verify CNI supports NetworkPolicy (Calico, Cilium, etc.)
- Check policy was created: `kubectl get networkpolicy -n dynamo`

### ChaosMesh not working
- Verify ChaosMesh is installed: `kubectl get pods -n chaos-mesh`
- Check NetworkChaos CR: `kubectl get networkchaos -n dynamo`

### Cleanup orphaned policies
```bash
# Via API
curl -X POST "http://fault-injection-api:8080/api/v1/faults/network/cleanup?namespace=dynamo"

# Manual
kubectl delete networkpolicy -n dynamo -l managed-by=fault-injection-api
```

