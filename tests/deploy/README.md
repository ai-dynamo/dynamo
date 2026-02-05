# Deployment Tests

CI deployment tests for validating Dynamo deployments and inference functionality.

## Overview

These tests replicate the functionality of the shell-based deploy tests in `.github/workflows/pr.yaml` (the `deploy-test-vllm`, `deploy-test-sglang`, and `deploy-test-trtllm` jobs).

Each test:
1. Applies a DynamoGraphDeployment manifest to Kubernetes
2. Waits for all pods to reach ready state
3. Tests model availability via `/v1/models` API endpoint
4. Tests inference via `/v1/chat/completions` API endpoint
5. Validates response format and content
6. Automatically cleans up the deployment

## Test Configurations

The tests cover 10 deployment scenarios across 3 frameworks:

### vLLM (4 profiles)
- `vllm-agg`: Aggregated (single worker)
- `vllm-agg_router`: Aggregated with KV router
- `vllm-disagg`: Disaggregated (separate prefill/decode)
- `vllm-disagg_router`: Disaggregated with KV router

### SGLang (2 profiles)
- `sglang-agg`: Aggregated
- `sglang-agg_router`: Aggregated with router

### TensorRT-LLM (4 profiles)
- `trtllm-agg`: Aggregated
- `trtllm-agg_router`: Aggregated with router
- `trtllm-disagg`: Disaggregated
- `trtllm-disagg_router`: Disaggregated with router

## Running Tests

### Prerequisites

1. **Kubernetes cluster access**: Tests require a valid kubeconfig
2. **Namespace**: Either existing or permissions to create one
3. **Python dependencies**:
   ```bash
   pip install pytest pytest-asyncio pyyaml kubernetes kr8s
   ```

### Running All Deploy Tests

```bash
# Run all deploy tests
pytest -m "deploy and pre_merge" tests/deploy/

# With verbose output
pytest -v -m "deploy and pre_merge" tests/deploy/
```

### Running Tests for Specific Framework

```bash
# vLLM only
pytest -m "deploy and pre_merge and vllm" tests/deploy/

# SGLang only
pytest -m "deploy and pre_merge and sglang" tests/deploy/

# TensorRT-LLM only
pytest -m "deploy and pre_merge and trtllm" tests/deploy/
```

### Running Specific Test Configuration

```bash
# Run specific test by name
pytest -k "vllm-agg" tests/deploy/

# Run multiple specific profiles
pytest -k "agg_router" tests/deploy/
```

### Configuration via Environment Variables

```bash
# Set custom namespace
export DYNAMO_DEPLOY_NAMESPACE=my-test-namespace

# Set custom runtime image
export DYNAMO_RUNTIME_IMAGE=my-registry.com/dynamo:latest-vllm-amd64

# Set custom model
export MODEL_NAME=meta-llama/Llama-2-7b-hf

# Use base64-encoded kubeconfig (CI pattern)
export KUBECONFIG_B64=$(cat ~/.kube/config | base64 -w0)

# Or use standard KUBECONFIG
export KUBECONFIG=~/.kube/config

# Run tests
pytest -m "deploy" tests/deploy/
```

### Configuration via Command-Line Options

```bash
# Custom namespace
pytest --deploy-namespace=my-namespace tests/deploy/

# Custom runtime image
pytest --runtime-image=my-registry.com/dynamo:tag tests/deploy/
```

## CI Integration

In CI, these tests run in the `deploy-test-pytest` job alongside the legacy shell-based tests:

```yaml
deploy-test-pytest:
  needs: [deploy-operator, vllm, sglang, trtllm]
  strategy:
    matrix:
      framework: [vllm, sglang, trtllm]
  env:
    DYNAMO_DEPLOY_NAMESPACE: ${{ needs.deploy-operator.outputs.NAMESPACE }}
    DYNAMO_RUNTIME_IMAGE: ...
```

The pytest tests run in parallel with the legacy tests to allow comparison and validation during the transition period.

## Test Structure

```
tests/deploy/
├── __init__.py
├── README.md                    # This file
├── conftest.py                  # Pytest fixtures and configuration
└── test_ci_deployments.py       # Main test file with all configurations
```

### conftest.py

Provides fixtures for:
- `deploy_namespace`: Kubernetes namespace (from env or CLI)
- `runtime_image`: Container image for deployment (from env or CLI)
- `kubeconfig_path`: Path to kubeconfig file
- `model_name`: Model name for inference testing
- `ingress_suffix`: Ingress suffix for deployments

### test_ci_deployments.py

Contains:
- `DeployConfig`: Dataclass for deployment configuration
- `DEPLOY_CONFIGS`: Dictionary of all test configurations
- `wait_for_model_available()`: Helper to poll `/v1/models` endpoint
- `validate_chat_completion()`: Helper to test and validate inference
- `test_deploy_inference()`: Main parametrized test function

## Test Markers

Tests use the following pytest markers:

- `@pytest.mark.deploy`: All deployment tests
- `@pytest.mark.k8s`: Requires Kubernetes
- `@pytest.mark.e2e`: End-to-end tests
- `@pytest.mark.pre_merge`: Run before merge
- `@pytest.mark.vllm` / `sglang` / `trtllm`: Framework-specific
- `@pytest.mark.gpu_1` / `gpu_2` / `gpu_4`: GPU requirements

## Debugging

### View Test Logs

```bash
# Run with detailed logging
pytest -v --log-cli-level=DEBUG -m "deploy and vllm-agg" tests/deploy/

# Keep temporary files for inspection
pytest --basetemp=/tmp/pytest-deploy tests/deploy/
```

### Check Kubernetes Resources

During test execution, you can inspect resources in another terminal:

```bash
# Watch pods
kubectl get pods -n <namespace> --watch

# Check deployments
kubectl get dynamographdeployments -n <namespace>

# View pod logs
kubectl logs -n <namespace> <pod-name>
```

### Common Issues

**Issue: `No kubeconfig found`**
```bash
# Solution: Set KUBECONFIG environment variable
export KUBECONFIG=~/.kube/config
```

**Issue: `Namespace already exists`**
```bash
# Solution: Use a different namespace or clean up existing one
kubectl delete namespace <namespace>
# Or set a different namespace
export DYNAMO_DEPLOY_NAMESPACE=my-unique-namespace
```

**Issue: `Image pull errors`**
```bash
# Solution: Ensure runtime image is accessible
# Check image exists
docker pull $DYNAMO_RUNTIME_IMAGE

# Or use locally available image
export DYNAMO_RUNTIME_IMAGE=local-image:tag
```

**Issue: `Timeout waiting for pods`**
```bash
# Solution: Check pod status and events
kubectl describe pods -n <namespace>
kubectl get events -n <namespace> --sort-by='.lastTimestamp'
```

## Comparison with Legacy Tests

| Aspect | Legacy (Shell) | Pytest |
|--------|---------------|--------|
| **Location** | `.github/workflows/pr.yaml` | `tests/deploy/` |
| **Language** | Bash | Python |
| **Test Results** | Text logs | JUnit XML |
| **Granularity** | 3 jobs (pass/fail) | 10 test cases |
| **Local Execution** | Requires replicating CI env | `pytest -m deploy` |
| **Debugging** | Parse logs | Pytest tracebacks |
| **Reusability** | Copy-paste shell scripts | Import Python functions |

## Future Work

- [ ] Add performance metrics collection
- [ ] Add support for multimodal inference tests
- [ ] Add tests for different model sizes
- [ ] Add tests for error scenarios (OOM, timeout, etc.)
- [ ] Parallelize tests where possible (separate namespaces)
- [ ] Add integration with test reporting dashboard

## Related Documentation

- [ManagedDeployment API](../utils/managed_deployment.py)
- [Fault Tolerance Deploy Tests](../fault_tolerance/deploy/)
- [CI Workflow](.github/workflows/pr.yaml)
