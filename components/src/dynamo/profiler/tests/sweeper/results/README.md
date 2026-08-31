<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contributed Sweeper Results

This directory indexes sanitized execution reports contributed from compatible clusters. Final
DGD goldens remain under `generated/`; reports record what ran and whether each phase passed.

## Result Index

| GPU SKU | Case | Scope | Tested by | Revision | Report |
| --- | --- | --- | --- | --- | --- |
| H100 SXM | `qwen3-32b-fp8-trtllm-agg` | Cluster validation | ashnamehrotra | `2c3ccc681e` | [2026-08-31](h100/2026-08-31-2c3ccc681e-ashnamehrotra/qwen3-32b-fp8-trtllm-agg/report.md) |

The initial H100 result predates the suite/hardware composition merged in #14046. It preserves the
original schema v2 JSON and Markdown output as historical evidence. New runs emit schema v3
`report.json` files; they do not generate Markdown.

## Contribute a Result

Run offline generation with `--report`, then run `tests/deploy/test_sweeper_cases.py` against a
compatible Kubernetes cluster. Review `report.json` for sensitive information before copying it
under:

```text
<gpu-sku>/<run-id>/<case>/
```

Use a run ID such as `<date>-<short-git-sha>-<github-handle>`. Do not commit generated DGDs,
candidates, caches, model data, logs containing secrets, or Kubernetes credentials.
