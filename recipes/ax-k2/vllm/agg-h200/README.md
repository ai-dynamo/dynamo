<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A.X-K2 H200 Aggregated Serving

Use four TP8 workers (32 H200 GPUs) with FP8 weights, expert parallelism,
KV-aware routing, EAGLE3 k=3, and a 262,144-token context limit. The pinned
image uses `--moe-backend flashinfer_cutlass`, `--kv-cache-dtype bfloat16`,
and `--attention-backend FLASH_ATTN_MLA_SPARSE`. If model initialization runs
out of memory, lower `--max-model-len` to 32768 on every worker and remeasure.

Edit `kustomize/base/deploy.yaml`, then regenerate from the repository root:

```bash
python3 scripts/kustomize-matrix.py unfold recipes/ax-k2/vllm/agg-h200/.kustomize-matrix.yaml
python3 scripts/kustomize-matrix.py render recipes/ax-k2/vllm/agg-h200/.kustomize-matrix.yaml
```

Apply `deploy-generic.yaml` with your cluster bindings. The measured aggregated
[W1 workload](../../perf/h200/README.md#h200-w1-measurements) uses fresh
workers and distinct input seeds; the existing Job has different sweep settings.
