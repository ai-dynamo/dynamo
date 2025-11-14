# Deploy YAML Validation

Validates `deploy.yaml` files against the DynamoGraphDeployment CRD schema without requiring kubectl or a Kubernetes cluster.

## Features

- **Dynamic validation** from the CRD OpenAPI v3 schema - adapts automatically when the schema changes
- **Schema validation**: Types, enums, required fields, constraints
- **Structural linting**: Detects indentation errors and misplaced properties
- **Line numbers**: Pinpoints exact locations of schema violations

## Quick Start

```bash
# Install dependencies
pip install pyyaml jsonschema

# Validate all deploy files
python3 deploy/utils/validate_deployments.py

# Validate specific files
python3 deploy/utils/validate_deployments.py recipes/*/deploy.yaml

# Verbose output
python3 deploy/utils/validate_deployments.py --verbose
```

## Integration

### Pre-commit Hook

Automatically validates on commit:

```bash
pre-commit install
pre-commit run validate-deploy-yaml --all-files
```

### CI Workflow

Runs on every PR via `.github/workflows/validate-deployments.yml`.

## Example Output

**Valid files:**
```
✅ recipes/qwen3-32b-fp8/trtllm/agg/deploy.yaml: Valid
✅ All 2 file(s) are valid!
```

**Validation errors:**
```
❌ recipes/example/deploy.yaml: 3 error(s)
   - Line 8: spec.backendFramework: value 'invalid' must be one of: 'sglang', 'vllm', 'trtllm'
   - Line 26: spec.services.Frontend.replicas: expected type 'integer', got 'str'
   - spec.services.Worker.unknownField: unexpected property 'unknownField' (not defined in schema - check indentation)
```

## How It Works

1. Extracts OpenAPI v3 schema from the CRD:
   ```
   deploy/cloud/operator/config/crd/bases/nvidia.com_dynamographdeployments.yaml
   ```
2. Validates using JSON Schema (Draft 7)
3. Performs additional structural checks for indentation errors

## Schema Updates

When the CRD schema changes (via kubebuilder annotations), validation automatically adapts:

```bash
cd deploy/cloud/operator
make manifests  # Regenerate CRD
```

No code changes needed in the validator!

## Troubleshooting

**Missing dependencies:**
```bash
pip install pyyaml jsonschema
```

**CRD not found:**
```bash
python3 deploy/utils/validate_deployments.py --crd path/to/crd.yaml
```

**Pre-commit not running:**
```bash
pip install pre-commit
pre-commit install
```
