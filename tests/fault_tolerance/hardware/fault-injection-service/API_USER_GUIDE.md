# Fault Injection API - User Guide

Test network fault tolerance without Kubernetes expertise. Inject network partitions, validate recovery, and ensure your system handles failures gracefully.

## What You Can Do

- **Network Partitions**: Simulate worker↔NATS, worker↔frontend, or custom pod isolation
- **Automatic Recovery**: Set fault duration or manually recover
- **Easy Testing**: Run tests locally or in-cluster (no port-forwarding needed)
- **Clean Output**: Colored test results and helper utilities included

## Prerequisites

- Fault Injection API deployed (see [README.md](README.md))
- **Recommended**: Use `run_test_incluster.py` script (no setup needed)
- **Alternative**: Port-forward API for local testing: `kubectl port-forward -n fault-injection-system svc/fault-injection-api 8080:8080`

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
```

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

### 1. Direct Pod-to-Pod Blocking (Label-Based)

Block specific pod communication using Kubernetes label selectors. **Most precise method.**

```python
# Block worker from reaching frontend pods specifically
fault = client.inject_network_partition(
    partition_type=NetworkPartition.CUSTOM,
    source="dynamo-oviya",
    target="dynamo-oviya",
    mode=NetworkMode.NETWORKPOLICY,
    duration=60,
    namespace="dynamo-oviya",
    target_pod_prefix="vllm-agg-0-vllmdecodeworker",  # Pod that gets isolated
    block_specific_pods=[
        {"app.kubernetes.io/name": "vllm-agg-0-frontend"},  # Block by label
    ],
    block_nats=False  # Keep NATS working
)
```

**How it works:**
- NetworkPolicy applied TO: `vllm-agg-0-vllmdecodeworker-9c25m` (matched by prefix)
- Blocks traffic TO: Any pod with label `app.kubernetes.io/name=vllm-agg-0-frontend`
- Allows: NATS, DNS, and all other pods

**Find pod labels:**
```bash
kubectl get pod <pod-name> -n <namespace> --show-labels
```

**Common label patterns:**
- `{"app.kubernetes.io/name": "my-service"}`
- `{"app.kubernetes.io/component": "frontend"}`
- `{"nvidia.com/dynamo-component-type": "frontend"}`

**Block multiple services:**
```python
block_specific_pods=[
    {"app.kubernetes.io/name": "frontend"},
    {"app.kubernetes.io/name": "api-gateway"},
    {"component": "storage"},
]
```

### 2. Worker→NATS Partition

Test system resilience when a worker loses NATS connectivity. Other workers should handle requests.

```python
fault = client.inject_network_partition(
    partition_type=NetworkPartition.FRONTEND_WORKER,
    source="dynamo-oviya",
    target="dynamo-oviya",
    mode=NetworkMode.NETWORKPOLICY,
    duration=60,
    namespace="dynamo-oviya",
    target_pod_prefix="vllm-agg-0-vllmdecodeworker",
    block_nats=True
)
```


### 3. Frontend→NATS Partition

Critical test: frontend loses NATS connectivity. Tests recovery from catastrophic failure.

```python
fault = client.inject_network_partition(
    partition_type=NetworkPartition.FRONTEND_WORKER,
    source="dynamo-oviya",
    target="dynamo-oviya",
    mode=NetworkMode.NETWORKPOLICY,
    duration=60,
    namespace="dynamo-oviya",
    target_pod_prefix="vllm-agg-0-frontend",
    block_nats=True
)
```

## Pytest Integration

```python
import pytest
from fault_injection_client import FaultInjectionClient, NetworkMode, NetworkPartition

@pytest.mark.fault_tolerance
def test_worker_nats_partition():
    """Test system behavior when worker loses NATS connection"""
    client = FaultInjectionClient(api_url="http://localhost:8080")
    
    # Inject fault
    fault = client.inject_network_partition(
        partition_type=NetworkPartition.FRONTEND_WORKER,
        source="dynamo-oviya",
        target="dynamo-oviya",
        mode=NetworkMode.NETWORKPOLICY,
        namespace="dynamo-oviya",
        target_pod_prefix="vllm-agg-0-vllmdecodeworker",
        block_nats=True,
        duration=60
    )
    
    try:
        # Test application behavior during fault
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
    mode=NetworkMode.NETWORKPOLICY,                   # Always NetworkPolicy
    duration=60,                                       # Auto-recover in 60s (optional)
    namespace="namespace",                             # Where to create policy
    target_pod_prefix="pod-prefix",                   # Pods to target
    block_nats=True,                                   # Block NATS access (default: True)
    block_specific_pods=[],                            # Block specific pods by label (optional)
)
```

**Key Parameters:**
- `partition_type`: Use `NetworkPartition.FRONTEND_WORKER` or `NetworkPartition.CUSTOM`
- `target_pod_prefix`: Match your pod names (e.g., `"vllm-agg-0-vllmdecodeworker"`)
- `block_nats`: Set to `True` to block NATS, `False` to keep NATS working
- `block_specific_pods`: List of label selectors to block specific pods (e.g., `[{"app.kubernetes.io/name": "frontend"}]`)
- `duration`: Auto-recover after N seconds (omit for manual recovery)

**Blocking Strategies:**

| Strategy | Use Case | Example |
|----------|----------|---------|
| `block_nats=True` | Block NATS messaging | Broad isolation from messaging system |
| `block_specific_pods=[...]` | Block specific services | Precise pod-to-pod isolation by labels |
| Both combined | Block NATS + specific pods | Multiple isolation targets |
| Neither (both False/empty) | Custom egress rules | Use with `allow_namespaces` |

### recover_fault()

```python
client.recover_fault(fault_id)
```

Removes the NetworkPolicy and restores communication.

### Test Helpers

```python
get_config_from_env()                    # Returns: {api_url, frontend_url, app_namespace}
check_frontend_reachable(frontend_url)  # Returns: bool
send_completion_request(prompt, tokens) # Returns: Response
validate_completion_response(response)  # Returns: str (completion text)
```

## Best Practices

1. **Always recover faults** - Use try/finally to ensure cleanup
2. **Verify baseline first** - Check system is healthy before injecting faults
3. **Wait after injection** - Give NetworkPolicy time to take effect (~5s)
4. **Test recovery** - Verify system returns to normal after fault clears

## Troubleshooting

**Fault not working?**
- Check pod names match `target_pod_prefix`: `kubectl get pods -n <namespace>`
- Verify NetworkPolicy created: `kubectl get networkpolicies -n <namespace>`
- Check API logs: `kubectl logs -n fault-injection-system -l app=fault-injection-api`

**Can't recover?**
- Manually delete: `kubectl delete networkpolicy <policy-name> -n <namespace>`
- List all faults: `client.list_faults()`

## Create Your Own Test

**1. Write your test in `examples/`:**

```python
# examples/my_custom_test.py
from fault_injection_client import FaultInjectionClient, NetworkMode, NetworkPartition
from test_helpers import get_config_from_env, Colors

config = get_config_from_env()
client = FaultInjectionClient(api_url=config['api_url'])

# Option 1: Block NATS
    fault = client.inject_network_partition(
    partition_type=NetworkPartition.FRONTEND_WORKER,
    source=config['app_namespace'],
    target=config['app_namespace'],
    mode=NetworkMode.NETWORKPOLICY,
    namespace=config['app_namespace'],
    target_pod_prefix="vllm-agg-0-vllmdecodeworker",
    block_nats=True,
        duration=60
    )
    
# Option 2: Block specific pod by label
# fault = client.inject_network_partition(
#     partition_type=NetworkPartition.CUSTOM,
#     source=config['app_namespace'],
#     target=config['app_namespace'],
#     mode=NetworkMode.NETWORKPOLICY,
#     namespace=config['app_namespace'],
#     target_pod_prefix="vllm-agg-0-vllmdecodeworker",
#     block_specific_pods=[{"app.kubernetes.io/name": "vllm-agg-0-frontend"}],
#     block_nats=False,
#     duration=60
# )

print(f"{Colors.GREEN}[OK]{Colors.RESET} Fault injected: {fault.fault_id}")
    
    # Your test logic here...
    
    # Recover
    client.recover_fault(fault.fault_id)
print(f"{Colors.GREEN}[OK]{Colors.RESET} Test passed!")
```

**2. Run in-cluster (recommended):**

```bash
python scripts/run_test_incluster.py examples/my_custom_test.py
```

**3. Or run locally (requires port-forwarding):**

```bash
# Terminal 1: kubectl port-forward -n fault-injection-system svc/fault-injection-api 8080:8080
# Terminal 2: kubectl port-forward -n dynamo-oviya svc/vllm-agg-frontend 8000:8000
# Terminal 3: python examples/my_custom_test.py
```

## Need Help?

- Check API logs: `kubectl logs -n fault-injection-system -l app=fault-injection-api`
- Verify pods are running: `kubectl get pods -n fault-injection-system`
- Review active NetworkPolicies: `kubectl get networkpolicies -n <namespace>`
- See example tests in: `examples/`

