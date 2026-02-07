# Prometheus Metrics Comparison: vLLM vs SGLang vs TensorRT-LLM

This doc has been moved to [Dynamo Metrics Comparison (Google Doc)](https://docs.google.com/document/d/1Righfrz_2n_MkXXEyZ5Vc_pblymvunlhIX6p1dUwfd8/edit?tab=t.c7n9kldymgcb).

## Overview

| Framework | Metric Prefix | Total Unique Metrics |
|-----------|---------------|---------------------|
| vLLM | `vllm:` | ~30 |
| SGLang | `sglang:` | ~40+ |
| TensorRT-LLM | `trtllm_` | ~5 |

All frameworks share the common `dynamo_component_*` metrics from the Dynamo runtime.
