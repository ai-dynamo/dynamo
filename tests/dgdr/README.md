# DGDR v1beta1 Tests

This folder contains end-to-end tests for the DGDR v1beta1 API.

## What We Are Testing So Far

1. Validation and conversion baseline
- Webhook validation for accepted/rejected DGDR specs
- v1alpha1 -> v1beta1 conversion behavior
- Basic CRD UX checks (e.g., shortname/columns)

2. Lifecycle behavior
- Rapid + `autoApply=false` reaches `Ready`
- Rapid + `autoApply=true` reaches `Deployed` in non-mocker mode

3. Profiling behavior
- Profiling output ConfigMap exists and `final_config.yaml` is parseable
- Planner feature appears in generated DGD when enabled
- Known rapid total GPU budget regression is tracked as `xfail`

## Test Files

- `tests/dgdr/test_dgdr_validation.py`
- `tests/dgdr/test_dgdr_lifecycle.py`
- `tests/dgdr/test_dgdr_profiling.py`

## Prerequisites

1. Kubernetes cluster with Dynamo operator and DGDR CRD/webhook installed
2. `kubectl` configured for that cluster
3. Python environment with `pytest`
4. Required args for all DGDR runs:
- `--dgdr-namespace`
- `--dgdr-image`

## How To Run

### Mocker Mode (default, no real GPU deployment)

```bash
PYTHONPATH=components/src python3 -m pytest tests/dgdr/ -v \
  --dgdr-namespace=dynamo-system \
  --dgdr-image=nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.0.2
```

### GPU Cluster Mode (real deployment path)

```bash
PYTHONPATH=components/src python3 -m pytest tests/dgdr/ -v \
  --dgdr-namespace=dynamo-system \
  --dgdr-image=nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.0.2 \
  --dgdr-no-mocker
```

### Optional: Validation/Conversion Only (fastest subset)

```bash
PYTHONPATH=components/src python3 -m pytest tests/dgdr/test_dgdr_validation.py -v \
  --dgdr-namespace=dynamo-system \
  --dgdr-image=nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.0.2
```

## Scope Note

This suite currently focuses on validation, conversion, lifecycle, and profiling behavior.
Broader DGDR operator semantics (hardware overrides, immutability, cleanup, and deeper
status assertions) are planned as follow-up test expansion.
