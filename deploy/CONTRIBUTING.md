# Contributing to Dynamo Deploy

Welcome to the Dynamo Deploy project! This guide will help you get started with contributing to the deployment infrastructure and tooling for the Dynamo distributed inference platform.

## Getting Started

### Prerequisites


### Quick Setup

### Project Structure

The deploy directory contains several key components:

```
deploy/
├── cloud/                    # Cloud deployment platform
│   ├── helm/                # Cloud platform Helm charts
│   └── operator/            # Kubernetes operator (Go)
├── helm/                    # Manual deployment Helm charts
├── metrics/                 # Monitoring and observability
├── sdk/                     # Python scripts
└── inference-gateway/       # Gateway components
```

## Development Environment

### Setting Up Your Environment


### IDE Configuration

**VS Code:**

- Install Go extension
- Install Python extension
- Configure settings for Go formatting and linting
- Add workspace settings for consistent formatting

### Contribution Workflow Caveats

- We do signed commits

```bash
commit -S
```

- Every time you modify `deploy/helm/charts/crds/templates/*.yaml`, please bump up the version of the CRD helm chart in
    1. deploy/helm/charts/platform/components/operator/Chart.yaml
    2. deploy/helm/charts/platform/Chart.yaml
then

```bash
deploy/helm/charts/platform
helm dependency update
```

#### Commit Message Guidelines

Follow conventional commit format:

- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `test:` adding or updating tests
- `refactor:` code refactoring
- `perf:` performance improvements
- `ci:` CI/CD changes

Examples:

```
feat(operator): add support for custom resource limits
fix(sdk): resolve service discovery timeout issue
docs(helm): update deployment guide with new examples
test(e2e): add integration tests for disaggregated serving
```

## Style Guide

### Go Code Style (Operator)

Follow standard Go conventions.


### Python Code Style (SDK)

Follow PEP 8 and use modern Python practices:


### YAML/Helm Templates

```yaml
# Use consistent indentation (2 spaces)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "dynamo.fullname" . }}
  labels:
    {{- include "dynamo.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "dynamo.selectorLabels" . | nindent 6 }}
```

## Testing

Once you have an MR up and standard checks pass trigger the integration tests by adding the comment “/ok to test <COMMIT-ID> “


### Unit Tests

**Go Tests (Operator):**

```bash
cd deploy/operator
go test ./... -v
go test -race ./...
```

### Integration Tests

**End-to-End Deployment Tests:**

```bash
# Run full deployment test suite
pytest tests/serve/test_dynamo_serve.py -v

# Test specific deployment scenarios
pytest tests/serve/test_dynamo_serve.py::test_serve_deployment[agg] -v
```

**Operator Integration Tests:**

```bash
cd deploy/operator
make test-e2e
```

### Writing Tests

**Example Unit Test:**

**Example Integration Test:**


### Examples Testing

Ensure documentation examples work.


Thank you for contributing to Dynamo Deploy! 🚀
