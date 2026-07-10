<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Trace Staging

Trace JSONL files are intentionally not copied into this work-tracker staging
tree yet.

For local DGD benchmarking, stage Mooncake-format traces onto the model-cache
PVC at:

```text
/model-cache/traces/
```

Expected agentic filenames:

```text
64k_400_90kv_agent_new_noschedule.jsonl
64k_400_90kv_agent_new_noschedule_short_30perc.jsonl
64k_400_90kv_agent_new_noschedule_short_15perc.jsonl
```

The final Dynamo PR can vendor trace files using Git LFS, matching the policy in
`.gitattributes`, once the exact release trace set is selected.
