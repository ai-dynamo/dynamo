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

- <!-- List prerequisites -->

### Usage

```bash
# Add minimal usage example from existing docs
# Example pattern (from Router):
# python -m dynamo.frontend --router-mode kv --http-port 8000
```

### Kubernetes

```yaml
# Add DGDR example - use apiVersion: nvidia.com/v1alpha1
# Example pattern (from Router):
# apiVersion: nvidia.com/v1alpha1
# kind: DynamoGraphDeployment
# metadata:
#   name: <component>-deployment
# spec:
#   services:
#     ...
```

<!-- EXAMPLE: Filled-in Quick Start for Router would look like:

### Prerequisites

- Dynamo platform installed
- At least one backend worker running

### Usage

```bash
python -m dynamo.frontend --router-mode kv --http-port 8000
```

### Kubernetes

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: router-example
spec:
  graphs:
    - name: frontend
      replicas: 1
```
-->

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| <!-- param --> | <!-- default --> | <!-- description --> |

## Next Steps

| Document | Path | Description |
|----------|------|-------------|
| `<Component> Guide` | `<component>_guide.md` | Deployment and configuration |
| `<Component> Examples` | `<component>_examples.md` | Usage examples |
| `<Component> Design` | `/docs/design_docs/<component>_design.md` | Architecture |

<!-- Convert table rows to markdown links -->
