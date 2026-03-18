# DGDR v1beta1 Tests (PR2 Scope)

This directory contains the DGDR v1beta1 tests currently in scope for PR2.

## Test files in this branch

- `tests/dgdr/test_dgdr_validation.py`
  - Webhook validation and version conversion checks (no profiling lifecycle assertions)
- `tests/dgdr/test_dgdr_lifecycle.py`
  - Focused lifecycle checks for rapid mode
  - `autoApply=false` reaches `Ready`
  - `autoApply=true` reaches `Deployed` in non-mocker mode
- `tests/dgdr/test_dgdr_profiling.py`
  - Profiling-focused checks for rapid mode
  - Output ConfigMap generation and `final_config.yaml` parseability
  - Planner feature emitted in generated DGD
  - Known budget guard as `xfail` (tracked issue)

## Prerequisites

1. Kubernetes cluster with Dynamo operator + DGDR CRD/webhook installed
2. `kubectl` configured for that cluster
3. Python + pytest dependencies available
4. Required test args:
   - `--dgdr-namespace`
   - `--dgdr-image`

## How to run

### 1) Validation/conversion only (fastest)

```bash
PYTHONPATH=components/src python3 -m pytest tests/dgdr/test_dgdr_validation.py -v \
  --dgdr-namespace=dynamo-system \
  --dgdr-image=nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.0.2
```

### 2) PR2 lifecycle + profiling tests (mocker/default)

```bash
PYTHONPATH=components/src python3 -m pytest \
  tests/dgdr/test_dgdr_lifecycle.py \
  tests/dgdr/test_dgdr_profiling.py -v \
  --dgdr-namespace=dynamo-system \
  --dgdr-image=nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.0.2
```

### 3) Real-GPU lifecycle deployment path (non-mocker)

```bash
PYTHONPATH=components/src python3 -m pytest tests/dgdr/test_dgdr_lifecycle.py -v \
  --dgdr-namespace=dynamo-system \
  --dgdr-image=nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.0.2 \
  --dgdr-no-mocker
```

## Notes

- PR2 intentionally focuses on lifecycle + profiling coverage.
- Broader operator semantics (hardware/overrides/status/immutability/cleanup) are planned for PR3.
- If required args are omitted, DGDR tests are skipped by fixture logic.
