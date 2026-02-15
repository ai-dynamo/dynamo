<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Fault Tolerance Scenario Test Framework

A declarative test framework for Dynamo fault tolerance testing on Kubernetes. Tests are defined as **scenarios** composed of **events**, **checks**, and optional **reports**.

## Quick Start

### Install Dynamo Platform

Follow the [instructions](../../../docs/pages/kubernetes/installation-guide.md) to install `Dynamo` in your Kubernetes cluster.

### Mount Workspace and Kube Config

Ensure you are able to run a `Dynamo` deployment directly from your host.

Then run the development container mounting the workspace and your kube config.

```bash
./container/run.sh --mount-workspace -it -v ~/.kube:/root/.kube

# Run a specific test
pytest tests/fault_tolerance/deploy/test_deployment_scenario.py \
  -s -v \
  --namespace my-namespace \
  --image nvcr.io/nvidia/dynamo:latest \
  -k test_smoke_50_requests
```

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Test Flow](#test-flow)
- [How to Run Tests](#how-to-run-tests)
- [Command-Line Options](#command-line-options)
- [PVC Requirements](#pvc-requirements)
- [How to Add a New Test](#how-to-add-a-new-test)
- [Available Events](#available-events)
- [Available Checks](#available-checks)
- [Parameterized Tests](#parameterized-tests)
- [Extending the Framework](#extending-the-framework)
- [Output Directory Structure](#output-directory-structure)

---

## Architecture Overview

The framework uses a declarative approach where each test is a **scenario** with three components:

```
┌─────────────────────────────────────────────────────────────┐
│                      run_scenario()                          │
│                    (scenario.py)                             │
└─────────────────┬───────────────────┬───────────────────────┘
                  │                   │
      ┌───────────▼───────────┐  ┌───▼───────────────────┐
      │       Events          │  │    Checks + Reports   │
      │      (events.py)      │  │   (checks.py, etc.)   │
      └───────────┬───────────┘  └───────────────────────┘
                  │
      ┌───────────▼───────────┐
      │   ManagedDeployment   │
      │    ManagedLoad        │
      │  (tests/utils/*.py)   │
      └───────────────────────┘
```

| Component | Description |
|-----------|-------------|
| **Events** | Actions to execute (start load, wait, delete pod, etc.) |
| **Checks** | Validations after events complete (zero errors, min requests, etc.) |
| **Reports** | Optional artifacts generated after checks pass |
| **DeploymentSpec** | Configuration for what to deploy (YAML path, replicas, image) |

---

## Test Flow

Each scenario follows this execution flow:

```
1. SETUP
   └── Create ManagedDeployment (applies CR, waits for ready)
   └── Start resource monitoring (if configured)

2. EXECUTE EVENTS (in order)
   └── StartLoad → creates aiperf job
   └── Wait → sleep
   └── DeletePod / TerminateProcess → inject failure
   └── WaitForRecovery → wait for deployment ready
   └── StopLoad → terminate load and collect results

3. STOP EVENTS (reverse order)
   └── Collect results from any active loads

4. CLEANUP
   └── Collect logs from all pods
   └── Delete deployment

5. POST-CLEANUP
   └── Generate reports (if any)
   └── Run checks (assertions)
```

---

## How to Run Tests

### Prerequisites

1. **Kubernetes cluster** with Dynamo operator installed
2. **kubectl** configured and authenticated
3. **Dynamo container** with workspace mounted

### Basic Usage

```bash
# Run all scenario tests
pytest tests/fault_tolerance/deploy/test_deployment_scenario.py -s -v \
  --namespace ${NAMESPACE} \
  --image ${IMAGE}

# Run a specific test
pytest tests/fault_tolerance/deploy/test_deployment_scenario.py -s -v \
  --namespace ${NAMESPACE} \
  --image ${IMAGE} \
  -k test_smoke_50_requests

# Run parameterized tests for a specific backend
pytest tests/fault_tolerance/deploy/test_deployment_scenario.py -s -v \
  --namespace ${NAMESPACE} \
  --image ${IMAGE} \
  -k "test_engine_process_termination and trtllm"
```

### Running with Markers

```bash
# Run only fault tolerance tests
pytest tests/fault_tolerance/deploy/ -m fault_tolerance -s -v ...

# Run weekly tests
pytest tests/fault_tolerance/deploy/ -m weekly -s -v ...

# Run E2E tests
pytest tests/fault_tolerance/deploy/ -m e2e -s -v ...
```

---

## Command-Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--image` | string | None | **Required**. Docker image to use for deployment |
| `--namespace` | string | `fault-tolerance-test` | Kubernetes namespace for the test |
| `--storage-class` | string | None | Storage class for PVC (must support RWX). Uses cluster default if not specified |
| `--skip-service-restart` | flag | False | Skip restarting NATS/etcd before deployment |
| `--client-type` | choice | `aiperf` | Load generator: `aiperf` (default) or `legacy` |

### Examples

```bash
# Basic run
pytest ... --namespace my-test --image myregistry/dynamo:v1.0

# With custom storage class (required for some clusters)
pytest ... --storage-class azurefile-csi-premium

# Skip service restart (useful for iterating on same namespace)
pytest ... --skip-service-restart

# Use legacy client for load generation
pytest ... --client-type legacy
```

---

## PVC Requirements

The framework uses a **shared PersistentVolumeClaim (PVC)** for log collection and load test results.

### Requirements

| Requirement | Details |
|-------------|---------|
| **Access Mode** | `ReadWriteMany` (RWX) - multiple pods must access simultaneously |
| **Size** | Minimum 500Mi (configurable via `pvc_size` in code) |
| **Storage Class** | Must support RWX. Common options: `azurefile-csi`, `nfs`, `efs` |

### How It Works

1. **ManagedDeployment** creates the PVC when `enable_log_collection()` is called
2. Service pods write logs to `/tmp/service_logs` (mounted from PVC)
3. **ManagedLoad** writes aiperf results to the same PVC
4. After test completion, logs are extracted locally

### Specifying Storage Class

```bash
# Use a specific storage class
pytest ... --storage-class my-rwx-storage-class

# If not specified, uses cluster default (may not support RWX!)
```

### Troubleshooting PVC Issues

```bash
# Check if PVC is bound
kubectl get pvc -n ${NAMESPACE}

# Check storage class capabilities
kubectl get storageclass

# Check events for PVC issues
kubectl describe pvc -n ${NAMESPACE}
```

---

## How to Add a New Test

### Step 1: Define the Test Function

Add to `test_deployment_scenario.py`:

```python
@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.e2e
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_my_new_scenario(request):
    """
    Description of what this test validates.

    Scenario:
    1. Start load
    2. Do something
    3. Stop load
    4. Assert conditions
    """
    await run_scenario(
        request=request,
        deployment_spec=DeploymentSpec(
            "/workspace/examples/backends/trtllm/deploy/agg.yaml"
        ),
        events=[
            StartLoad(load_config=LoadConfig(duration_minutes=5, concurrency=8)),
            Wait(duration=30),
            # Add your events here
            StopLoad(),
        ],
        checks=[
            ZeroErrors(),
            MinRequests(min_count=50),
        ],
    )
```

### Step 2: Configure the Deployment

```python
# Option 1: Use a deployment YAML directly
deployment_spec = DeploymentSpec("/workspace/examples/backends/trtllm/deploy/agg.yaml")

# Option 2: Customize replicas
deployment_spec = DeploymentSpec("/workspace/examples/backends/vllm/deploy/disagg.yaml")
deployment_spec.set_service_replicas("VllmPrefillWorker", 2)
deployment_spec.set_service_replicas("VllmDecodeWorker", 2)

# Option 3: Use the helper function for parameterized tests
deployment_spec = get_deployment_spec(
    backend="trtllm",      # trtllm, vllm, or sglang
    deployment_type="agg", # agg or disagg
    worker_replicas=2
)
```

### Step 3: Configure Load

```python
# Duration-based load
load_config = LoadConfig(
    duration_minutes=5,    # Run for 5 minutes
    concurrency=8,         # 8 concurrent requests
    streaming=True,
)

# Request-count-based load
load_config = LoadConfig(
    request_count=100,     # Send exactly 100 requests
    concurrency=4,
)

# Full configuration
load_config = LoadConfig(
    model_name="Qwen/Qwen3-0.6B",
    concurrency=8,
    duration_minutes=5,
    input_tokens_mean=512,
    output_tokens_mean=64,
    streaming=True,
    request_rate=10.0,  # 10 req/s rate limit
)
```

---

## Available Events

### Load Events

| Event | Description | Parameters |
|-------|-------------|------------|
| `StartLoad` | Start an aiperf load test | `load_config`, `name="default"` |
| `StopLoad` | Stop load early, collect results | `name="default"` |
| `WaitForLoadCompletion` | Wait for load to finish naturally | `name="default"`, `timeout=None` |

### Basic Events

| Event | Description | Parameters |
|-------|-------------|------------|
| `Wait` | Sleep for duration | `duration` (seconds) |
| `WaitForRecovery` | Wait for deployment to be ready | `timeout=600`, `unready_timeout=60` |

### Failure Injection Events

| Event | Description | Parameters |
|-------|-------------|------------|
| `DeletePod` | Delete pods for services | `services=["Service1"]`, `force=True` |
| `TerminateProcess` | Kill process by name | `services`, `process_name`, `signal="SIGKILL"` |
| `RollingUpgrade` | Trigger rolling upgrade | `services`, `ready_timeout=1800` |
| `WaitForLogPattern` | Wait for pattern in logs | `service`, `pattern`, `timeout=300` |

### Example: Failure Injection

```python
events=[
    StartLoad(load_config=LoadConfig(duration_minutes=5, concurrency=8)),
    Wait(duration=30),

    # Delete a pod
    DeletePod(services=["TRTLLMDecodeWorker"]),

    # Or terminate a specific process
    TerminateProcess(
        services=["TRTLLMDecodeWorker"],
        process_name="dynamo.runtime",
        signal="SIGKILL"
    ),

    WaitForRecovery(timeout=300),
    Wait(duration=30),
    StopLoad(),
]
```

---

## Available Checks

| Check | Description | Parameters |
|-------|-------------|------------|
| `ZeroErrors` | Assert zero errors | `name="default"` |
| `MaxErrors` | Assert errors below threshold | `max_errors`, `name="default"` |
| `MinRequests` | Assert minimum successful requests | `min_count`, `name="default"` |
| `WasCancelled` | Assert cancellation status | `expected=True`, `name="default"` |
| `ServiceLogContains` | Assert pattern in service log | `service`, `pattern` |
| `ServiceLogNotContains` | Assert pattern NOT in log | `service`, `pattern` |

### Example: Multiple Checks

```python
checks=[
    ZeroErrors(),                    # No errors allowed
    MinRequests(min_count=100),      # At least 100 successful requests
    WasCancelled(expected=False),    # Load completed naturally
]

# Or for fault tolerance tests (some errors expected)
checks=[
    MaxErrors(max_errors=20),        # Allow up to 20 errors
    MinRequests(min_count=50),       # But still need 50+ successes
]
```

---

## Parameterized Tests

Use `pytest.mark.parametrize` to run tests across multiple configurations:

```python
@pytest.mark.parametrize("backend,deployment_type,replicas", [
    ("trtllm", "agg", 1),
    ("trtllm", "agg", 2),
    ("trtllm", "disagg", 1),
    ("trtllm", "disagg", 2),
    ("vllm", "agg", 1),
    ("vllm", "agg", 2),
])
async def test_my_parameterized_test(request, backend, deployment_type, replicas):
    await run_scenario(
        request=request,
        deployment_spec=get_deployment_spec(backend, deployment_type, replicas),
        events=[...],
        checks=[...],
    )
```

### Service Names by Backend

| Backend | Type | Services |
|---------|------|----------|
| trtllm | agg | Frontend, TRTLLMWorker |
| trtllm | disagg | Frontend, TRTLLMPrefillWorker, TRTLLMDecodeWorker |
| vllm | agg | Frontend, VllmWorker |
| vllm | disagg | Frontend, VllmPrefillWorker, VllmDecodeWorker |
| sglang | agg | Frontend, SglangWorker |
| sglang | disagg | Frontend, SglangPrefillWorker, SglangDecodeWorker |

---

## Extending the Framework

### Adding a New Event

1. Add to `events.py`:

```python
@dataclass
class MyNewEvent(Event):
    """Description of the event."""

    my_param: str
    name: str = ""
    results: dict[str, Any] | None = field(default=None, init=False)

    async def execute(self, ctx: "ScenarioContext") -> None:
        ctx.logger.info(f"Executing MyNewEvent with {self.my_param}")
        # Do something with ctx.deployment

    async def stop(self, ctx: "ScenarioContext") -> None:
        # Optional cleanup
        pass

    @property
    def description(self) -> str:
        return f"My new event: {self.my_param}"
```

2. Export from module and import in test file.

### Adding a New Check

1. Add to `checks.py`:

```python
@dataclass
class MyNewCheck(Check):
    """Description of the check."""

    threshold: int
    name: str = "default"

    def validate(self, ctx: "ScenarioContext") -> None:
        load = self.get_load(ctx, self.name)
        assert load and load.results, f"No results for load '{self.name}'"

        # Get value from results
        value = load.results.get("some_metric", {}).get("avg", 0)

        ctx.logger.info(f"MyNewCheck: value={value}, threshold={self.threshold}")
        assert value >= self.threshold, f"Expected >= {self.threshold}, got {value}"

    @property
    def description(self) -> str:
        return f"My check >= {self.threshold} ('{self.name}')"
```

### Adding a New Report

1. Add to `reports.py`:

```python
@dataclass
class MyNewReport(Report):
    """Description of the report."""

    output_path: str

    def generate(self, ctx: "ScenarioContext") -> None:
        # Generate report from ctx.events, ctx.resource_history, etc.
        with open(self.output_path, "w") as f:
            f.write("Report content...")

    @property
    def description(self) -> str:
        return f"Generate report: {self.output_path}"
```

---

## Output Directory Structure

After each test, logs are collected to a directory named after the test:

```
test_smoke_50_requests/
├── load/
│   ├── profile_export_aiperf.json    # AI-Perf metrics (parsed for checks)
│   ├── profile_export_aiperf.csv     # Tabular metrics
│   └── aiperf.log                    # AI-Perf execution log
├── Frontend/
│   ├── pod-name_1234567890.log       # Current container logs
│   ├── pod-name_1234567890.previous.log  # Previous container (pre-restart)
│   ├── pod-name_1234567890.metrics.log   # Prometheus metrics
│   └── pod-name_1234567890.yaml      # Pod manifest
├── TRTLLMWorker/
│   └── [same structure as Frontend]
├── services/
│   └── [logs written by services to PVC]
└── resource_history.json             # Resource monitoring data (if enabled)
```

---

## Troubleshooting

### Test hangs during deployment

```bash
# Check pod status
kubectl get pods -n ${NAMESPACE}

# Check events
kubectl get events -n ${NAMESPACE} --sort-by='.lastTimestamp'

# Check CR status
kubectl get dynamo -n ${NAMESPACE} -o yaml
```

### PVC not binding

```bash
# Check PVC status
kubectl describe pvc -n ${NAMESPACE}

# Ensure storage class supports RWX
kubectl get storageclass -o yaml | grep -A5 "name:"
```

### Load test not starting

```bash
# Check load job pod
kubectl get pods -n ${NAMESPACE} -l app=load-test

# Check load job logs
kubectl logs -n ${NAMESPACE} -l app=load-test
```

### Checks failing

1. Check the test output for specific assertion messages
2. Review `profile_export_aiperf.json` for actual metrics
3. Check service logs for errors during the test
