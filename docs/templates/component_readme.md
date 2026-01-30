---
orphan: true
---

# <Component>

<!-- 2-3 sentence overview of what this component does and its role in Dynamo -->

## Feature Matrix

| Feature | Status |
|---------|--------|
| Feature 1 | ✅ Supported |
| Feature 2 | 🚧 Experimental |
| Feature 3 | ❌ Not Supported |

## Quick Start

### Prerequisites

- Dynamo installed (`pip install nvidia-dynamo`)
- <!-- Other prerequisites -->

### Python

```bash
python -m dynamo.<component> --model <model-path>
```

### Kubernetes

```yaml
apiVersion: dynamo.nvidia.com/v1alpha1
kind: DynamoGraphDeploymentRequest
metadata:
  name: <component>-example
spec:
  # Minimal configuration
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--param1` | `value` | Description |
| `--param2` | `value` | Description |

## Next Steps

- \[<Component> Guide\](<component>_guide.md) - Deployment and configuration
- \[<Component> Examples\](<component>_examples.md) - Usage examples
- \[<Component> Design\](/docs/design_docs/<component>_design.md) - Architecture
