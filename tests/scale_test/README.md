# Scale Testing Tool

Deploys configurable numbers of Dynamo mocker instances as DynamoGraphDeployment (DGD) resources on Kubernetes with load generation.

## Prerequisites

- Kubernetes cluster with kubectl access
- Dynamo operator installed
- Python 3.10+ with dependencies: `pip install kubernetes-asyncio kr8s openai requests`

```bash
# Verify setup
kubectl get crd dynamographdeployments.nvidia.com
kubectl get pods -n dynamo-operator-system
```

### Private Container Registries

For private registries like NGC:

```bash
kubectl create secret docker-registry nvcr-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password=<YOUR_NGC_API_KEY> \
  --namespace=<YOUR_NAMESPACE>
```

Then use `--image-pull-secrets nvcr-secret` in your commands.

## Quick Start

All commands run from the workspace root using `python -m tests.scale_test`.

```bash
# Full test: deploy 10 DGDs, load test at 2 QPS for 60s, cleanup
python -m tests.scale_test run --count 10 --duration 60 --qps 2 --namespace my-ns --in-cluster

# Deploy and keep running for manual testing
python -m tests.scale_test start --count 5 --namespace my-ns

# Load test existing DGDs
python -m tests.scale_test load --namespace my-ns --duration 60 --qps 2 --in-cluster

# Cleanup leftover DGDs
python -m tests.scale_test cleanup --namespace my-ns
```

Use `--in-cluster` when frontends are ClusterIP services (the default).

## Commands

### `start` - Deploy and wait

```bash
python -m tests.scale_test start --count 5 --namespace my-ns
```

### `run` - Deploy, load test, cleanup

```bash
python -m tests.scale_test run --count 10 --duration 60 --qps 2 --in-cluster
```

### `load` - Load test existing DGDs

```bash
python -m tests.scale_test load --namespace my-ns --qps 5 --in-cluster
```

### `cleanup` - Delete DGDs

```bash
python -m tests.scale_test cleanup --namespace my-ns
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--count` | 5 | Number of DGD deployments |
| `--namespace` | default | Kubernetes namespace |
| `--image` | nvcr.io/.../vllm-runtime:latest | Container image |
| `--image-pull-secrets` | - | Secret names for image pulls |
| `--model-path` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | Model path |
| `--speedup-ratio` | 10.0 | Mocker speedup |
| `--timeout` | 600 | Ready timeout (seconds) |
| `--duration` | 60 | Load test duration (seconds) |
| `--qps` | 1.0 | Queries per second |
| `--in-cluster` | false | Run load generator as K8s Job |
| `--load-gen-pods` | 1 | Parallel load generator pods |
| `--load-gen-processes` | 1 | Processes per pod |
| `--no-cleanup` | false | Keep DGDs after exit |
| `-v` | false | Debug logging |

## High QPS Scaling

For high QPS (1000+), use multiple pods and/or processes:

```bash
# 1000 QPS with 4 processes
python -m tests.scale_test run --count 10 --qps 1000 --in-cluster --load-gen-processes 4

# 5000 QPS with 5 pods x 2 processes = 10 workers
python -m tests.scale_test run --count 20 --qps 5000 --in-cluster \
  --load-gen-pods 5 --load-gen-processes 2
```

| Target QPS | Configuration |
|------------|---------------|
| < 500 | Default |
| 500-2000 | `--load-gen-processes 4` |
| 2000-5000 | `--load-gen-pods 4 --load-gen-processes 2` |
| 5000+ | `--load-gen-pods 10 --load-gen-processes 4` |

## Troubleshooting

```bash
# Check DGD status
kubectl get dgd -n <namespace>
kubectl describe dgd scale-test-1 -n <namespace>

# Check pods
kubectl get pods -n <namespace> -l nvidia.com/dynamo-graph-deployment-name=scale-test-1
kubectl logs <pod-name> -n <namespace>

# Check operator
kubectl logs -n dynamo-operator-system -l control-plane=controller-manager

# Manual cleanup
kubectl get dgd -n <namespace> | grep scale-test | awk '{print $1}' | xargs kubectl delete dgd -n <namespace>
```

## Programmatic Usage

```python
import asyncio
from tests.scale_test import ScaleManager

async def run_test():
    manager = ScaleManager(
        num_deployments=5,
        kubernetes_namespace="test",
        model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )
    
    try:
        await manager._init_kubernetes()
        await manager.deploy_dgds()
        await manager.wait_for_dgds_ready()
        
        # Run load generator as Kubernetes Job
        await manager.run_load_generator_job(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            duration_sec=30,
            qps=1.0,
        )
    finally:
        await manager.cleanup()

asyncio.run(run_test())
```

Or use the async context manager:

```python
async with ScaleManager(num_deployments=5, kubernetes_namespace="test") as manager:
    urls = await manager.get_frontend_urls()
    # ... run tests
```
