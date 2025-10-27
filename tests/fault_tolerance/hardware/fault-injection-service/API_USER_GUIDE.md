# Fault Injection API - User Guide

Test network fault tolerance without Kubernetes expertise. Inject network partitions, validate recovery, and ensure your system handles failures gracefully.

## What You Can Do

### NetworkPolicy Mode (Built-in)
- **Complete network isolation** - block all traffic between pods
- **No installation required** - uses Kubernetes NetworkPolicy
- Best for testing: binary failures, complete partitions, failover scenarios

### ChaosMesh Mode (Advanced)
- **Partial packet loss** - simulate 50% packet drop
- **Network delay** - add latency + jitter
- **Bandwidth limiting** - simulate slow networks
- **Requires ChaosMesh installed** - see installation below
- Best for testing: degraded networks, congestion, high-latency scenarios

### General Features
- **Automatic Recovery**: Set fault duration or manually recover
- **Easy Testing**: Run tests in-cluster (no port-forwarding needed)
- **Clean Output**: Colored test results and helper utilities included

## Prerequisites

### Required
- Fault Injection API deployed (see [README.md](README.md))
- **Recommended**: Use `run_test_incluster.py` script (no setup needed)

### Optional (for ChaosMesh tests)
- Install ChaosMesh for packet loss/delay tests:
```bash
kubectl create ns chaos-mesh
helm repo add chaos-mesh https://charts.chaos-mesh.org
helm install chaos-mesh chaos-mesh/chaos-mesh -n chaos-mesh \
  --set chaosDaemon.runtime=containerd \
  --set chaosDaemon.socketPath=/run/containerd/containerd.sock
```

### Alternative
- Port-forward API for local testing: `kubectl port-forward -n fault-injection-system svc/fault-injection-api 8080:8080`

## Quick Start

### Run Pre-Built Tests

The fastest way to test fault tolerance:

```bash
# Run Pod-to-Pod blocking test (label-based, most precise)
python scripts/run_test_incluster.py examples/test_specific_pod_to_pod_blocking.py

# Run Worker→NATS partition test
python scripts/run_test_incluster.py examples/test_partition_worker_to_nats.py

# Run Worker→Frontend partition test
python scripts/run_test_incluster.py examples/test_partition_worker_to_frontend.py

# Run Frontend→NATS partition test (critical failure scenario)
python scripts/run_test_incluster.py examples/test_partition_frontend_to_nats.py

# Run ChaosMesh tests (requires ChaosMesh installed)
python scripts/run_test_incluster.py examples/test_nats_packet_loss_50_percent.py
```

> **Note:** Update `FRONTEND_URL` in `scripts/run_test_incluster.py` (line 99) to match your deployment:
> - `vllm-agg-frontend` for aggregated deployments
> - `vllm-disagg-frontend` for disaggregated deployments
> - Verify with: `kubectl get svc -n dynamo-oviya` or your namespace

**Note:** ChaosMesh tests require ChaosMesh installed (see [Prerequisites](#prerequisites)).

**See the [Create Your Own Test](#create-your-own-test) section below for writing custom tests.**

### Test Helpers

Use built-in helpers for cleaner output and validation:

```python
from test_helpers import (
    Colors,                          # Color codes for terminal output
    get_config_from_env,            # Get API_URL, FRONTEND_URL, APP_NAMESPACE
    check_frontend_reachable,       # Verify frontend health
    send_completion_request,        # Send inference request
    validate_completion_response,   # Validate response structure
)

# Check if frontend is healthy
if check_frontend_reachable():
    print(f"{Colors.GREEN}[OK]{Colors.RESET} Frontend ready")

# Send a test request
    response = send_completion_request("Hello world", max_tokens=10)
    text = validate_completion_response(response)
print(f"Got: {text}")
```

## Common Use Cases

### NetworkPolicy Mode Examples

**1. Block NATS traffic:**
```python
fault = client.inject_network_partition(
    partition_type=NetworkPartition.CUSTOM,
    mode=NetworkMode.NETWORKPOLICY,
    namespace="dynamo-oviya",
    target_pod_prefix="vllm-agg-0-vllmdecodeworker",
    block_nats=True,
    duration=60
)
```

**2. Block specific pod (by label):**
```python
fault = client.inject_network_partition(
    partition_type=NetworkPartition.CUSTOM,
    mode=NetworkMode.NETWORKPOLICY,
    namespace="dynamo-oviya",
    target_pod_prefix="vllm-agg-0-vllmdecodeworker",
    block_specific_pods=[{"app.kubernetes.io/name": "vllm-agg-0-frontend"}],
    block_nats=False,  # Keep NATS working
    duration=60
)
```

*Find pod labels: `kubectl get pod <pod-name> -n <namespace> --show-labels`*

### ChaosMesh Mode Examples

**Requires ChaosMesh installed** (already on AKS dynamo-dev, see [Prerequisites](#prerequisites) for other clusters)

**1. Packet loss (50%):**
```python
fault = client.inject_network_partition(
    partition_type=NetworkPartition.CUSTOM,
    mode=NetworkMode.CHAOS_MESH,
    namespace="dynamo-oviya",
    target_pod_prefix="vllm-agg-0-vllmdecodeworker",
    packet_loss_percent=50,
    target_nats=True,
    duration=60
)
```

**2. Network delay:**
```python
fault = client.inject_network_partition(
    partition_type=NetworkPartition.CUSTOM,
    mode=NetworkMode.CHAOS_MESH,
    namespace="dynamo-oviya",
    target_pod_prefix="vllm-agg-0-vllmdecodeworker",
    delay_ms=100,
    delay_jitter_ms=50,
    target_nats=True,
    duration=60
)
```

## Pytest Integration

```python
@pytest.mark.fault_tolerance
def test_worker_nats_partition():
    client = FaultInjectionClient(api_url="http://localhost:8080")
    fault = client.inject_network_partition(
        mode=NetworkMode.NETWORKPOLICY,
        namespace="dynamo-oviya",
        target_pod_prefix="vllm-agg-0-vllmdecodeworker",
        block_nats=True,
        duration=60
    )
    try:
        response = send_test_request()
        assert response.status_code in [200, 503]
    finally:
        client.recover_fault(fault.fault_id)
```

## API Reference

### inject_network_partition()

```python
fault = client.inject_network_partition(
    partition_type=NetworkPartition.FRONTEND_WORKER,  # Type of partition
    source="namespace",                                # Source namespace
    target="namespace",                                # Target namespace
    mode=NetworkMode.NETWORKPOLICY,                   # NETWORKPOLICY or CHAOS_MESH
    duration=60,                                       # Auto-recover in 60s (optional)
    namespace="namespace",                             # Where to create policy
    target_pod_prefix="pod-prefix",                   # Pods to target
    
    # NetworkPolicy mode parameters:
    block_nats=True,                                   # Block NATS access (default: True)
    block_specific_pods=[],                            # Block specific pods by label (optional)
    
    # ChaosMesh mode parameters:
    packet_loss_percent=0,                             # Packet loss percentage (0-100)
    delay_ms=0,                                        # Delay in milliseconds
    delay_jitter_ms=0,                                 # Jitter for delay (ms)
    bandwidth_limit="",                                # Bandwidth limit (e.g., "1mbps")
    corrupt_percent=0,                                 # Packet corruption (0-100)
    duplicate_percent=0,                               # Packet duplication (0-100)
    target_nats=True,                                  # Target NATS traffic
    target_specific_pods=[],                           # Target specific pods
)
```

**Key Parameters:**
- `partition_type`: Use `NetworkPartition.FRONTEND_WORKER` or `NetworkPartition.CUSTOM`
- `target_pod_prefix`: Match your pod names (e.g., `"vllm-agg-0-vllmdecodeworker"`)
- `mode`: Choose `NetworkMode.NETWORKPOLICY` (complete blocking) or `NetworkMode.CHAOS_MESH` (advanced faults)
- `duration`: Auto-recover after N seconds (omit for manual recovery)

**NetworkPolicy Mode Parameters:**
- `block_nats`: Set to `True` to block NATS, `False` to keep NATS working
- `block_specific_pods`: List of label selectors to block specific pods (e.g., `[{"app.kubernetes.io/name": "frontend"}]`)

**ChaosMesh Mode Parameters (Requires ChaosMesh Installed):**
- `packet_loss_percent`: Percentage of packets to drop (0-100)
- `delay_ms`: Add delay in milliseconds
- `delay_jitter_ms`: Add jitter to delay (± ms)
- `bandwidth_limit`: Limit bandwidth (e.g., "1mbps", "100kbps")
- `corrupt_percent`: Corrupt packet data (0-100)
- `duplicate_percent`: Duplicate packets (0-100)
- `target_nats`: Target NATS traffic (default: True)
- `target_specific_pods`: Target specific pods by label selector

**Mode Comparison:**

| Mode | Use Case | Advantages | Requirements |
|------|----------|-----------|--------------|
| `NETWORKPOLICY` | Complete network isolation | Built-in Kubernetes, no extra deps | Complete blocking only |
| `CHAOS_MESH` | Degraded network conditions | Partial loss, delay, bandwidth limits | ChaosMesh must be installed |

**Blocking Strategies (NetworkPolicy Mode):**

| Strategy | Use Case | Example |
|----------|----------|---------|
| `block_nats=True` | Block NATS messaging | Broad isolation from messaging system |
| `block_specific_pods=[...]` | Block specific services | Precise pod-to-pod isolation by labels |
| Both combined | Block NATS + specific pods | Multiple isolation targets |
| Neither (both False/empty) | Custom egress rules | Use with `allow_namespaces` |

### recover_fault()

```python
client.recover_fault(fault_id)  # Removes NetworkPolicy/ChaosMesh resource
```

## Best Practices

1. **Always recover faults** - Use try/finally to ensure cleanup
2. **Verify baseline first** - Check system is healthy before injecting faults
3. **Wait after injection** - Give NetworkPolicy/ChaosMesh time to take effect (~5s)
4. **Test recovery** - Verify system returns to normal after fault clears

## Create Your Own Test

```python
# examples/my_custom_test.py
from fault_injection_client import FaultInjectionClient, NetworkMode
from test_helpers import get_config_from_env, send_completion_request

config = get_config_from_env()
client = FaultInjectionClient(api_url=config['api_url'])

fault = client.inject_network_partition(
    mode=NetworkMode.NETWORKPOLICY,
    namespace=config['app_namespace'],
    target_pod_prefix="vllm-agg-0-vllmdecodeworker",
    block_nats=True,
    duration=60
)

try:
    # Your test logic
    response = send_completion_request("Test prompt", 10)
finally:
    client.recover_fault(fault.fault_id)
```

**Run:** `python3 scripts/run_test_incluster.py examples/my_custom_test.py`

## Troubleshooting

**Fault not working:**
- Check pod names: `kubectl get pods -n <namespace>`
- Verify policy created: `kubectl get networkpolicies -n <namespace>`
- Check API logs: `kubectl logs -n fault-injection-system -l app=fault-injection-api`

**ChaosMesh not working:**
- Verify installed: `kubectl get pods -n chaos-mesh`
- Check permissions: `kubectl get clusterrole fault-injection-api -o yaml | grep chaos-mesh`

**Manual cleanup:**
- NetworkPolicy: `kubectl delete networkpolicy <name> -n <namespace>`
- ChaosMesh: `kubectl delete networkchaos <name> -n <namespace>`
