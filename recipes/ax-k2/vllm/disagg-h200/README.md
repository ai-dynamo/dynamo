<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A.X-K2 H200 Disaggregated Serving

Use TP8 workers with expert parallelism, KV-aware routing, EAGLE3 k=3,
and a 262,144-token context limit. If model initialization runs out of memory,
lower `--max-model-len` to 32768 on every worker.

Edit `kustomize/base/deploy.yaml`, then regenerate from the repository root:

```bash
python3 scripts/kustomize-matrix.py unfold recipes/ax-k2/vllm/disagg-h200/.kustomize-matrix.yaml
python3 scripts/kustomize-matrix.py render recipes/ax-k2/vllm/disagg-h200/.kustomize-matrix.yaml
```

Apply `deploy-generic.yaml` with your cluster bindings and run the
[one-Job AIPerf sweep](../../perf/h200/README.md).
