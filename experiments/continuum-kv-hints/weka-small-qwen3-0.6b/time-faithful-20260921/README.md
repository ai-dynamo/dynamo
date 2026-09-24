<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Source-clock oracle-retention WEKA matrix (2026-09-21)

This protocol replays four complete, lifecycle-rich WEKA traces whose source-clock spans fit within ten minutes. Token counts remain scaled by eight for Qwen3-0.6B, but request timestamps and latency annotations are not scaled.

The selector chooses the four highest-request-count eligible traces with source spans no longer than 600 seconds. With the pinned dataset, these are rows 13, 292, 116, and 98: 275 requests total. Every trajectory starts at the beginning. `--num-conversations 4` allows each complete root and subagent tree to drain while preventing a freed lane from recycling into a fifth root.

For every parent turn that spawns a subagent and later resumes, AIPerf attaches the recorded pause until the next parent turn as `X-Dynamo-Retention-TTL-Ms`. Dynamo converts that request-specific value to `ttl_seconds`; there is no fixed TTL fallback. SessionPrefixIndexer still resolves the selected worker's parent-session lineages into the retain action, and the action log records the resolved lineage count.

The scenario's global idle-gap cap is set above the benchmark duration so it cannot compress the replay. Results are written to `../artifacts/oracle-retention-20260921-<UTC timestamp>/` with the generated dataset manifest, exact dataset and AIPerf commands, source revisions and working-tree patches, per-request records, metrics, and Dynamo logs.

```bash
export CONTINUUM_IMAGE=<local-image-with-the-current-Dynamo-policy>
./experiments/continuum-kv-hints/weka-small-qwen3-0.6b/time-faithful-20260921/run_matrix.sh
```
