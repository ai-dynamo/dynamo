# DGDR and Sweeper Result: qwen3-32b-fp8-trtllm-agg

## Result

| Field | Value |
| --- | --- |
| Status | `passed` |
| Scope | `cluster-validation` |
| Tested by | ashnamehrotra |
| Revision | `2c3ccc681e2e385708f7107bf94fa439731c3421` |
| Model | `Qwen3-32B FP8` |
| Backend | `trtllm` |
| Deployment mode | `agg` |
| Recipe GPUs | 2 |
| Compatible GPU SKUs | h100, h200, a100 |
| Hardware profile | `azure-h100-nd96isr` |
| Recipe | `recipes/qwen3-32b-fp8/trtllm/agg/deploy.yaml` |

## Phase Status

| Phase | Status |
| --- | --- |
| `existing-profiler-generation` | `passed` |
| `sweeper-aic-materialization` | `passed` |
| `sweeper-direct-materialization` | `passed` |
| `sweeper-search` | `passed` |

## Cluster Validation

| Variant | Status | Duration (seconds) |
| --- | --- | ---: |
| `dgdr-v1beta1` | `passed` | 191.77 |
| `reference` | `passed` | 127.63 |
| `sweeper-aic` | `passed` | 211.59 |

```yaml
gpuProducts:
  Standard_ND96isr_H100_v5:
    gpus: 32
    nodes: 4
```
