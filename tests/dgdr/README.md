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

## Test Matrix

| # | File | Class | Test | Verifies |
|---|---|---|---|---|
| 1 | `test_dgdr_validation.py` | `TestDGDRValidation` | `test_missing_model_rejected` | Webhook rejects DGDR without `spec.model` |
| 2 | | | `test_thorough_with_auto_backend_rejected` | `thorough` + `backend: auto` invalid |
| 3 | | | `test_invalid_backend_rejected` | Unknown backend rejected |
| 4 | | | `test_invalid_search_strategy_rejected` | Unknown searchStrategy rejected |
| 5 | | | `test_invalid_optimization_type_rejected` | Invalid optimizationType rejected |
| 6 | | | `test_valid_minimal_dgdr_accepted` | Model + image only passes |
| 7 | | | `test_valid_full_spec_accepted` | Full v1beta1 spec passes |
| 8 | | | `test_v1beta1_is_storage_version` | CRD storage version is v1beta1 |
| 9 | | | `test_kubectl_shortname_dgdr_works` | `kubectl get dgdr` shortName works |
| 10 | | | `test_kubectl_get_columns_schema` | Columns: NAME, MODEL, BACKEND, PHASE |
| 11 | | `TestDGDRVersionConversion` | `test_v1alpha1_dgdr_can_be_applied` | v1alpha1 stored as v1beta1 |
| 12 | | | `test_v1beta1_get_on_v1alpha1_object` | Conversion webhook serves v1alpha1 |
| 13 | `test_dgdr_lifecycle.py` | `TestDGDRLifecycle` | `test_rapid_autoapply_false_reaches_ready` | Rapid profiling → Ready (3h timeout) |
| 14 | | | `test_rapid_autoapply_true_reaches_deployed_without_mocker` | GPU deploy → Deployed (skipped in mocker) |
| 15 | `test_dgdr_profiling.py` | `TestDGDRProfilingRapid` | `test_rapid_autoapply_false_emits_output_configmap` | Output ConfigMap has `final_config.yaml` |
| 16 | | | `test_rapid_planner_feature_emits_planner_service` | Generated DGD includes Planner |
| 17 | | | `test_rapid_generated_dgd_respects_total_gpus_budget` | GPUs ≤ totalGpus (**xfail** #8583) |

## Scope Note

This suite currently focuses on validation, conversion, lifecycle, and profiling behavior.
Broader DGDR operator semantics (hardware overrides, immutability, cleanup, and deeper
status assertions) are planned as follow-up test expansion.
