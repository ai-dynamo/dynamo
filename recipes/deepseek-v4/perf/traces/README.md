<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Trace Staging

The three agentic Mooncake-format traces are vendored in this directory via Git
LFS (see `.gitattributes`):

```text
64k_400_90kv_agent_new_noschedule.jsonl               # full
64k_400_90kv_agent_new_noschedule_short_30perc.jsonl  # 30% subset
64k_400_90kv_agent_new_noschedule_short_15perc.jsonl  # 15% subset
```

A fresh clone contains only LFS pointer files. Fetch the real content before
staging:

```bash
git lfs pull --include="recipes/deepseek-v4/perf/traces/*.jsonl"
```

Then copy them onto the model-cache PVC at `/model-cache/traces/` — see
[`../README.md`](../README.md) ("Stage Traces") for the `kubectl cp` step.
