---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Fault Tolerance Testing"
---

This document describes the test infrastructure for validating Dynamo's fault tolerance mechanisms. The testing framework supports request cancellation, migration, etcd HA, and hardware fault injection scenarios.

## Overview

Dynamo's fault tolerance test suite is located in `tests/fault_tolerance/` and includes:

| Test Category | Location | Purpose |
|---------------|----------|---------|
| Cancellation | `cancellation/` | Request cancellation during in-flight operations |
| Migration | `migration/` | Request migration when workers fail |
| etcd HA | `etcd_ha/` | etcd failover and recovery |
| Hardware | `hardware/` | GPU and network fault injection |
| Deployment | `deploy/` | End-to-end deployment testing |

## Test Directory Structure

```
tests/fault_tolerance/
├── cancellation/
│   ├── test_vllm.py
│   ├── test_trtllm.py
│   ├── test_sglang.py
│   └── utils.py
├── migration/
│   ├── test_vllm.py
│   ├── test_trtllm.py
│   ├── test_sglang.py
│   └── utils.py
├── etcd_ha/
│   ├── test_vllm.py
│   ├── test_trtllm.py
│   ├── test_sglang.py
│   └── utils.py
├── hardware/
│   └── fault_injection_service/
│       ├── api_service/
│       └── agents/
├── deploy/
│   ├── test_deployment.py
│   ├── scenarios.py
│   ├── base_checker.py
│   └── ...
└── client.py
```

## Request Cancellation Tests

Test that in-flight requests can be properly canceled.

### Running Cancellation Tests

```bash
# Run all cancellation tests
pytest tests/fault_tolerance/cancellation/ -v

# Run for specific backend
pytest tests/fault_tolerance/cancellation/test_vllm.py -v
```

### Cancellation Test Utilities

The `cancellation/utils.py` module provides:

#### CancellableRequest

Thread-safe request cancellation via TCP socket manipulation:

```python
from tests.fault_tolerance.cancellation.utils import CancellableRequest

request = CancellableRequest()

# Send request in separate thread
thread = Thread(target=send_request, args=(request,))
thread.start()

# Cancel after some time
time.sleep(1)
request.cancel()  # Closes underlying socket
```

#### send_completion_request / send_chat_completion_request

Send cancellable completion requests:

```python
from tests.fault_tolerance.cancellation.utils import (
    send_completion_request,
    send_chat_completion_request
)

# Non-streaming
response = send_completion_request(
    base_url="http://localhost:8000",
    model="Qwen/Qwen3-0.6B",
    prompt="Hello, world!",
    max_tokens=100
)

# Streaming with cancellation
responses = send_chat_completion_request(
    base_url="http://localhost:8000",
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
    cancellable_request=request
)
```

#### poll_for_pattern

Wait for specific patterns in logs:

```python
from tests.fault_tolerance.cancellation.utils import poll_for_pattern

# Wait for cancellation confirmation
found = poll_for_pattern(
    log_file="/var/log/dynamo/worker.log",
    pattern="Request cancelled",
    timeout=30,
    interval=0.5
)
```

## Migration Tests

Test that requests migrate to healthy workers when failures occur.

### Running Migration Tests

```bash
# Run all migration tests
pytest tests/fault_tolerance/migration/ -v

# Run for specific backend
pytest tests/fault_tolerance/migration/test_vllm.py -v
```

### Migration Test Utilities

The `migration/utils.py` module provides:

- Frontend wrapper with configurable request planes
- Long-running request spawning for migration scenarios
- Health check disabling for controlled testing

### Example Migration Test

```python
def test_migration_on_worker_failure():
    # Start deployment with 2 workers
    deployment = start_deployment(workers=2)

    # Send long-running request
    request_thread = spawn_long_request(max_tokens=1000)

    # Kill one worker mid-generation
    kill_worker(deployment.workers[0])

    # Verify request completes on remaining worker
    response = request_thread.join()
    assert response.status_code == 200
    assert len(response.tokens) > 0
```

## etcd HA Tests

Test system behavior during etcd failures and recovery.

### Running etcd HA Tests

```bash
pytest tests/fault_tolerance/etcd_ha/ -v
```

### Test Scenarios

- **Leader failover**: etcd leader node fails, cluster elects new leader
- **Network partition**: etcd node becomes unreachable
- **Recovery**: System recovers after etcd becomes available

## Hardware Fault Injection

The fault injection service enables testing under simulated hardware failures.

### Fault Injection Service

Located at `tests/fault_tolerance/hardware/fault_injection_service/`, this FastAPI service orchestrates fault injection:

```bash
# Start the fault injection service
cd tests/fault_tolerance/hardware/fault_injection_service
python -m api_service.main
```

### Supported Fault Types

#### GPU Faults

| Fault Type | Description |
|------------|-------------|
| `XID_ERROR` | Simulate GPU XID error (various codes) |
| `THROTTLE` | GPU thermal throttling |
| `MEMORY_PRESSURE` | GPU memory exhaustion |
| `OVERHEAT` | GPU overheating condition |
| `COMPUTE_OVERLOAD` | GPU compute saturation |

#### Network Faults

| Fault Type | Description |
|------------|-------------|
| `FRONTEND_WORKER` | Partition between frontend and workers |
| `WORKER_NATS` | Partition between workers and NATS |
| `WORKER_WORKER` | Partition between workers |
| `CUSTOM` | Custom network partition |

### Fault Injection API

#### Inject GPU Fault

```bash
curl -X POST http://localhost:8080/api/v1/faults/gpu/inject \
  -H "Content-Type: application/json" \
  -d '{
    "target_pod": "vllm-worker-0",
    "fault_type": "XID_ERROR",
    "severity": "HIGH"
  }'
```

#### Inject Specific XID Error

```bash
# Inject XID 79 (GPU memory page fault)
curl -X POST http://localhost:8080/api/v1/faults/gpu/inject/xid-79 \
  -H "Content-Type: application/json" \
  -d '{"target_pod": "vllm-worker-0"}'
```

Supported XID codes: 43, 48, 74, 79, 94, 95, 119, 120

#### Inject Network Partition

```bash
curl -X POST http://localhost:8080/api/v1/faults/network/inject \
  -H "Content-Type: application/json" \
  -d '{
    "partition_type": "FRONTEND_WORKER",
    "duration_seconds": 30
  }'
```

#### Recover from Fault

```bash
curl -X POST http://localhost:8080/api/v1/faults/{fault_id}/recover
```

#### List Active Faults

```bash
curl http://localhost:8080/api/v1/faults
```

### GPU Fault Injector Agent

The GPU fault injector runs as a DaemonSet on worker nodes:

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: gpu-fault-injector
spec:
  selector:
    matchLabels:
      app: gpu-fault-injector
  template:
    spec:
      containers:
      - name: agent
        image: dynamo/gpu-fault-injector:latest
        securityContext:
          privileged: true
        volumeMounts:
        - name: dev
          mountPath: /dev
```

The agent injects fake XID messages via `/dev/kmsg` to trigger NVSentinel detection.

## Deployment Testing Framework

The `deploy/` directory contains an end-to-end testing framework.

### Test Phases

Tests run through three phases:

| Phase | Description |
|-------|-------------|
| `STANDARD` | Baseline performance under normal conditions |
| `OVERFLOW` | System behavior during fault/overload |
| `RECOVERY` | System recovery after fault resolution |

### Scenario Configuration

Define test scenarios in `scenarios.py`:

```python
from tests.fault_tolerance.deploy.scenarios import Scenario, Load, Failure

scenario = Scenario(
    name="worker_failure_migration",
    backend="vllm",
    load=Load(
        clients=10,
        requests_per_client=100,
        max_tokens=256
    ),
    failure=Failure(
        type="pod_kill",
        target="vllm-worker-0",
        trigger_after_requests=50
    )
)
```

### Running Deployment Tests

```bash
# Run all deployment tests
pytest tests/fault_tolerance/deploy/test_deployment.py -v

# Run specific scenario
pytest tests/fault_tolerance/deploy/test_deployment.py::test_worker_failure -v
```

### Validation Checkers

The framework includes pluggable validators:

```python
from tests.fault_tolerance.deploy.base_checker import BaseChecker, ValidationContext

class MigrationChecker(BaseChecker):
    def check(self, context: ValidationContext) -> bool:
        # Verify migrations occurred
        migrations = context.metrics.get("migrations_total", 0)
        return migrations > 0
```

### Results Parsing

Parse test results for analysis:

```python
from tests.fault_tolerance.deploy.parse_results import process_overflow_recovery_test

results = process_overflow_recovery_test(log_dir="/path/to/logs")
print(f"Success rate: {results['success_rate']}")
print(f"P99 latency: {results['p99_latency_ms']}ms")
```

## Client Utilities

The `client.py` module provides shared client functionality:

### Multi-Threaded Load Generation

```python
from tests.fault_tolerance.client import client

# Generate load with multiple clients
results = client(
    base_url="http://localhost:8000",
    num_clients=10,
    requests_per_client=100,
    model="Qwen/Qwen3-0.6B",
    max_tokens=256,
    log_dir="/tmp/test_logs"
)
```

### Request Options

| Parameter | Description |
|-----------|-------------|
| `base_url` | Frontend URL |
| `num_clients` | Number of concurrent clients |
| `requests_per_client` | Requests per client |
| `model` | Model name |
| `max_tokens` | Max tokens per request |
| `log_dir` | Directory for client logs |
| `endpoint` | `completions` or `chat/completions` |

## Running the Full Test Suite

### Prerequisites

1. Kubernetes cluster with GPU nodes
2. Dynamo deployment
3. etcd cluster (for HA tests)
4. Fault injection service (for hardware tests)

### Environment Setup

```bash
export KUBECONFIG=/path/to/kubeconfig
export DYNAMO_NAMESPACE=dynamo-test
export FRONTEND_URL=http://localhost:8000
```

### Run All Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all fault tolerance tests
pytest tests/fault_tolerance/ -v --tb=short

# Run with specific markers
pytest tests/fault_tolerance/ -v -m "not slow"
```

### Test Markers

| Marker | Description |
|--------|-------------|
| `slow` | Long-running tests (> 5 minutes) |
| `gpu` | Requires GPU resources |
| `k8s` | Requires Kubernetes cluster |
| `etcd_ha` | Requires multi-node etcd |

## Best Practices

### 1. Isolate Test Environments

Run fault tolerance tests in dedicated namespaces:

```bash
kubectl create namespace dynamo-fault-test
```

### 2. Clean Up After Tests

Ensure fault injection is recovered:

```bash
# List and recover all active faults
curl http://localhost:8080/api/v1/faults | jq -r '.[].id' | \
  xargs -I {} curl -X POST http://localhost:8080/api/v1/faults/{}/recover
```

### 3. Collect Logs

Preserve logs for debugging:

```bash
pytest tests/fault_tolerance/ -v \
  --log-dir=/tmp/fault_test_logs \
  --capture=no
```

### 4. Monitor During Tests

Watch system state during tests:

```bash
# Terminal 1: Watch pods
watch kubectl get pods -n dynamo-test

# Terminal 2: Watch metrics
watch 'curl -s localhost:8000/metrics | grep -E "(migration|rejection)"'
```

## Related Documentation

- [Request Migration](request-migration.md) - Migration implementation details
- [Request Cancellation](request-cancellation.md) - Cancellation implementation
- [Health Checks](../observability/health-checks.md) - Health monitoring
- [Metrics](../observability/metrics.md) - Available metrics for monitoring
