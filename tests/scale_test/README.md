# Scale Testing Tool

A scale testing tool that deploys configurable numbers of Dynamo mocker instances as DynamoGraphDeployment (DGD) resources on Kubernetes, with load generation to verify functionality.

## Overview

This tool allows you to:
- Deploy multiple Dynamo mocker DGD deployments on Kubernetes
- Each DGD contains a Frontend and MockerWorker service
- The Dynamo operator manages NATS/etcd infrastructure
- Generate load across all frontends to verify functionality
- Collect and report latency/throughput metrics

## Prerequisites

- **Kubernetes cluster** with kubectl access
- **Dynamo operator** installed on the cluster
- **Python 3.10+** with required dependencies
- **kubectl** configured to access your cluster

### Verify Prerequisites

```bash
# Check kubectl is configured
kubectl cluster-info

# Check Dynamo operator is installed
kubectl get crd dynamographdeployments.nvidia.com

# Check operator is running
kubectl get pods -n dynamo-operator-system
```

## Architecture

```
+---------------------------------------------------------------+
|                      Scale Test Tool                           |
+---------------------------------------------------------------+
                               |
           +-------------------+-------------------+
           v                                       v
    +-------------+                         +-------------+
    | Kubernetes  |                         |    Load     |
    |    API      |                         |  Generator  |
    +-------------+                         +-------------+
           |                                       |
           v                                       |
    +-------------+                                |
    |   Dynamo    |                                |
    |  Operator   |                                |
    +-------------+                                |
           |                                       |
           +---manages--->  NATS/etcd Services     |
           |                                       |
   +-------+-------+-------+                       |
   v               v       v                       |
+--------+   +--------+   +--------+              |
| DGD    |   | DGD    |   | DGD    |              |
| scale- |   | scale- |   | scale- |              |
| test-1 |   | test-2 |   | test-N |              |
+---+----+   +---+----+   +---+----+              |
    |            |            |                    |
    v            v            v                    |
+--------+   +--------+   +--------+              |
|Frontend|   |Frontend|   |Frontend| <------------+
| :8000  |   | :8000  |   | :8000  |
+---+----+   +---+----+   +---+----+
    |            |            |
    v            v            v
+--------+   +--------+   +--------+
| Mocker |   | Mocker |   | Mocker |
| Worker |   | Worker |   | Worker |
+--------+   +--------+   +--------+
```

## Quick Start

### Run a Full Test

Deploy 10 DGD deployments, generate load for 60 seconds at 2 QPS, then cleanup:

```bash
python -m tests.scale_test run --count 10 --duration 60 --qps 2 --namespace my-namespace
```

### Start Deployments for Manual Testing

Deploy 5 DGDs and keep them running for manual testing:

```bash
python -m tests.scale_test start --count 5 --namespace my-namespace
```

Press `Ctrl+C` to cleanup and exit.

### Cleanup Leftover DGDs

If DGDs were left running (e.g., after a crash or using `--no-cleanup`):

```bash
python -m tests.scale_test cleanup --namespace my-namespace
```

## Commands

### `start`

Deploy N DGDs and wait for manual testing.

```bash
python -m tests.scale_test start [options]
```

Options:
- `--count N`: Number of DGD deployments (default: 5)
- `--namespace NS`: Kubernetes namespace (default: default)
- `--image IMAGE`: Container image (default: nvcr.io/nvidia/ai-dynamo/dynamo-base:latest)
- `--model-path PATH`: Model path for tokenizer (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- `--speedup-ratio RATIO`: Mocker speedup multiplier (default: 10.0)
- `--timeout SECS`: Timeout for DGDs to become ready (default: 600)
- `--name-prefix PREFIX`: Prefix for DGD names (default: scale-test)
- `--no-cleanup`: Do not delete DGDs on exit

### `run`

Run a full test: deploy + load generation + cleanup.

```bash
python -m tests.scale_test run [options]
```

All options from `start`, plus:
- `--duration SECS`: Load test duration in seconds (default: 60)
- `--qps RATE`: Queries per second to send (default: 1.0)

### `cleanup`

Cleanup any leftover scale test DGDs.

```bash
python -m tests.scale_test cleanup [options]
```

Options:
- `--namespace NS`: Kubernetes namespace (default: default)
- `--name-prefix PREFIX`: DGD name prefix to match (default: scale-test)

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--count` | 5 | Number of DGD deployments to create |
| `--namespace` | default | Kubernetes namespace for deployments |
| `--image` | nvcr.io/nvidia/ai-dynamo/dynamo-base:latest | Container image |
| `--model-path` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | Model path for tokenizer |
| `--speedup-ratio` | 10.0 | Mocker speedup (higher = faster simulation) |
| `--timeout` | 600 | Timeout for DGDs to become ready (seconds) |
| `--name-prefix` | scale-test | Prefix for DGD names |
| `--duration` | 60 | Load test duration (seconds) |
| `--qps` | 1.0 | Queries per second to generate |
| `-v, --verbose` | false | Enable debug logging |
| `--no-cleanup` | false | Keep DGDs after exit |

## Example Output

```
$ python -m tests.scale_test run --count 10 --duration 120 --qps 2 --namespace test

Deploying 10 DGD resources to Kubernetes...
Namespace: test
Image: nvcr.io/nvidia/ai-dynamo/dynamo-base:latest

Creating DGD 1/10: scale-test-1
DGD scale-test-1 created successfully
Creating DGD 2/10: scale-test-2
DGD scale-test-2 created successfully
...

Waiting for all DGDs to be ready...
DGD scale-test-1 is ready
DGD scale-test-2 is ready
...

============================================================
All DGDs ready!
============================================================

Generating load for 120 seconds at 2 QPS...
Targeting 10 frontends...
Load generation complete.

======================================================================
LOAD GENERATION RESULTS
======================================================================

Frontend http://scale-test-1-frontend.test.svc.cluster.local:8000:
  Requests: 24
  Successful: 24
  Errors: 0 (0.0%)
  Avg latency: 45.2ms
  P50 latency: 42.1ms
  P99 latency: 78.3ms

...

----------------------------------------------------------------------
Total requests: 240
Total errors: 0
Overall error rate: 0.0%
======================================================================

Cleaning up DGDs...
All DGDs deleted.
```

## DGD Structure

Each DGD deployment creates:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: scale-test-N
  namespace: <namespace>
spec:
  services:
    Frontend:
      dynamoNamespace: scale-test-N
      componentType: frontend
      replicas: 1
      # HTTP server on port 8000
    MockerWorker:
      dynamoNamespace: scale-test-N
      componentType: worker
      replicas: 1
      # Simulates LLM inference
```

## Namespace Isolation

Each DGD uses a unique Dynamo namespace to ensure isolation:

- DGD name: `scale-test-1`, `scale-test-2`, ..., `scale-test-N`
- Dynamo namespace: matches DGD name for service discovery
- Kubernetes namespace: shared across all DGDs (configurable)

## Troubleshooting

### DGDs not becoming ready

Check DGD status:
```bash
kubectl get dgd -n <namespace>
kubectl describe dgd scale-test-1 -n <namespace>
```

Check pod status:
```bash
kubectl get pods -n <namespace> -l nvidia.com/dynamo-graph-deployment-name=scale-test-1
kubectl logs <pod-name> -n <namespace>
```

### Operator not processing DGDs

Check operator logs:
```bash
kubectl logs -n dynamo-operator-system -l control-plane=controller-manager
```

### Image pull errors

Ensure the image is accessible from your cluster:
```bash
kubectl run test --image=<your-image> --rm -it -- echo "Image works"
```

### Cleanup not working

Manually delete DGDs:
```bash
kubectl delete dgd -n <namespace> -l app.kubernetes.io/managed-by=scale-test
# Or delete by name prefix
kubectl get dgd -n <namespace> | grep scale-test | awk '{print $1}' | xargs kubectl delete dgd -n <namespace>
```

## Programmatic Usage

You can also use the components directly in Python:

```python
import asyncio
from tests.scale_test import ScaleManager, LoadGenerator

async def run_scale_test():
    # Using async context manager for automatic cleanup
    async with ScaleManager(
        num_deployments=5,
        kubernetes_namespace="test",
        model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ) as manager:
        # Get frontend URLs
        urls = await manager.get_frontend_urls()
        print(f"Frontends: {urls}")

        # Run load generation
        load_generator = LoadGenerator(
            frontend_urls=urls,
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        )
        await load_generator.generate_load(
            duration_sec=30,
            qps=1.0,
        )
        load_generator.print_summary()

asyncio.run(run_scale_test())
```

## Module Structure

```
tests/scale_test/
  __init__.py          # Package exports
  __main__.py          # Entry point for python -m
  cli.py               # CLI commands and argument parsing
  config.py            # Configuration dataclasses
  dgd_builder.py       # DGD spec builder utility
  load_generator.py    # Load generation logic
  scale_manager.py     # Kubernetes DGD management
  utils.py             # Utility functions
  templates/
    mocker_deployment.yaml  # Base DGD template
```

## Dependencies

- Python 3.10+
- kubernetes-asyncio (async Kubernetes client)
- kr8s (Kubernetes utilities)
- OpenAI Python client (for load generation)
- requests
