# Fault Tolerance Test Framework

## Quick Start

Copy an existing test from `test_deployment_scenario.py` and modify it.
All tests follow the same pattern: **deploy → events → checks**.

```python
async def test_my_scenario(namespace, image, skip_service_restart, storage_class):
    spec = DeploymentSpec.from_backend("vllm", "agg")
    spec.set_worker_replicas(2)

    await run_scenario(
        deployment_spec=spec,
        events=[
            StartLoad(load_config=LoadConfig(duration_minutes=5, concurrency=8)),
            Wait(duration=30),
            DeletePod(services=["VllmWorker"]),
            WaitForRecovery(timeout=300),
            StopLoad(),
        ],
        checks=[
            MaxErrors(max_errors=20),
            MinRequests(min_count=50),
        ],
        namespace=namespace,
        image=image,
        test_name="test_my_scenario",
        skip_service_restart=skip_service_restart,
        storage_class=storage_class,
    )
```

## Concepts

- **DeploymentSpec**: What to deploy (backend, deployment type, replicas, image)
- **Events**: Actions executed in sequence (start load, inject fault, wait for recovery)
- **Checks**: Assertions validated after all events complete
- **LoadConfig**: Parameters for the aiperf load generator

## DeploymentSpec

```python
# From backend name (recommended)
spec = DeploymentSpec.from_backend("vllm", "disagg")

# From explicit YAML path
spec = DeploymentSpec("/workspace/examples/backends/vllm/deploy/agg.yaml")

# Useful properties and methods
spec.backend                        # "vllm" (auto-detected)
spec.worker_services()              # ["VllmPrefillWorker", "VllmDecodeWorker"]
spec.set_worker_replicas(2)         # Set all workers to 2 replicas
spec.set_service_replicas("X", 3)   # Set specific service replicas
spec.set_image("my-image:tag")      # Override image for all services
spec["VllmWorker"].replicas = 2     # Direct service access
```

## Available Events

| Event | Purpose | Key Params |
|-------|---------|------------|
| `StartLoad` | Begin load generation | `load_config`, `name` |
| `StopLoad` | Terminate running load early | `name` |
| `WaitForLoadCompletion` | Wait for load to finish naturally | `name`, `timeout` |
| `Wait` | Pause for N seconds | `duration` |
| `DeletePod` | Kill pod(s) for a service | `services`, `force` |
| `WaitForRecovery` | Wait for deployment to heal | `timeout` |
| `RollingUpgrade` | Trigger rolling restart | `services` |
| `TerminateProcess` | Kill a process inside a pod | `services`, `process_name`, `signal` |
| `WaitForLogPattern` | Wait for regex in service logs | `service`, `pattern`, `timeout` |
| `RunCommand` | Execute arbitrary command in pod | `services`, `command` |

## Available Checks

| Check | Purpose | Key Params |
|-------|---------|------------|
| `ZeroErrors` | Assert no errors in load results | `name` |
| `MaxErrors` | Assert errors below threshold | `max_errors`, `name` |
| `MinRequests` | Assert minimum successful requests | `min_count`, `name` |
| `WasCancelled` | Assert cancellation status | `expected`, `name` |
| `ServiceLogContains` | Assert pattern present in logs | `service`, `pattern` |
| `ServiceLogNotContains` | Assert pattern NOT in logs | `service`, `pattern` |

## Common Patterns

### Smoke test (no faults)
```python
events=[
    StartLoad(load_config=LoadConfig(request_count=50, concurrency=4)),
    WaitForLoadCompletion(),
]
checks=[ZeroErrors(), MinRequests(min_count=50)]
```

### Pod kill + recovery
```python
events=[
    StartLoad(load_config=LoadConfig(duration_minutes=5, concurrency=8)),
    Wait(duration=30),
    DeletePod(services=["VllmWorker"]),
    WaitForRecovery(timeout=300),
    Wait(duration=30),
    StopLoad(),
]
checks=[MaxErrors(max_errors=20), MinRequests(min_count=50)]
```

### Rolling upgrade
```python
events=[
    StartLoad(load_config=LoadConfig(duration_minutes=15, concurrency=8)),
    Wait(duration=30),
    RollingUpgrade(services=["TRTLLMDecodeWorker"]),
    Wait(duration=30),
    StopLoad(),
]
checks=[ZeroErrors(), MinRequests(min_count=100)]
```

### Custom fault injection
```python
events=[
    StartLoad(load_config=LoadConfig(duration_minutes=5, concurrency=8)),
    Wait(duration=30),
    RunCommand(services=["VllmWorker"], command="stress --vm 1 --vm-bytes 2G --timeout 30s"),
    Wait(duration=60),
    StopLoad(),
]
checks=[MaxErrors(max_errors=50)]
```

## Writing Custom Events

Subclass `Event` and implement `execute()` and `description`:

```python
@dataclass
class WaitForMetric(Event):
    """Wait for a metric condition on a service."""
    service: str
    metric_name: str
    condition: str  # e.g., "> 0.9"
    timeout: int = 300
    name: str = ""
    results: dict[str, Any] | None = field(default=None, init=False)

    async def execute(self, ctx):
        # Access metrics via ctx.deployment.port_forward()
        ...

    @property
    def description(self):
        return f"Wait for {self.metric_name} {self.condition}"
```

## Multi-backend Testing

Use `@pytest.mark.parametrize` for testing across backends:

```python
@pytest.mark.parametrize("backend,deployment_type,replicas", [
    ("vllm", "agg", 1),
    ("trtllm", "agg", 1),
    ("trtllm", "disagg", 2),
])
async def test_pod_kill(namespace, image, ..., backend, deployment_type, replicas):
    spec = DeploymentSpec.from_backend(backend, deployment_type)
    spec.set_worker_replicas(replicas)
    await run_scenario(deployment_spec=spec, ...)
```

## Running Tests

```bash
# In the dev container with kubectl access:
pytest tests/fault_tolerance/deploy/test_deployment_scenario.py \
    -s -v \
    --namespace my-namespace \
    --image nvcr.io/nvidian/dynamo-dev/my-image:tag \
    -k test_smoke_50_requests
```
