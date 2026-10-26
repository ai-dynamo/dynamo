---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Conditional Disaggregation
subtitle: Route selected requests to local prefill and decode on decode workers
---

> [!WARNING]
> **Experimental.** Validate/tune conditional disaggregation against your workload before using it in production.

Conditional disaggregation enables a hybrid of aggregated and disaggregated request routing. The router may serve a request `prefill worker -> decode worker`, or it may send the request directly to a decode worker and the backend runs local prefill plus decode there.

For workloads with a high degree of KV reusage on long-ISL requests, e.g. multi-turn / agentic conversation scenarios, conditional disaggregation can help your deployment maintain predictable SLA. Compared to unconditional disaggregation, it reduces memory pressure / TTFT on prefill workers by optimizing reuse of *decode-worker* KV cache; compared to unconditionally aggregated deployments, it avoids the heavy ITL penalty incurred by co-scheduling heavy prefill workload onto decode workers.

Enable conditional disaggregation with `--router-conditional-disagg` on the frontend:

```bash
python -m dynamo.frontend \
    --router-mode kv \
    --router-conditional-disagg
```

Tune the policy with `--router-conditional-disagg-config`. For example:

```bash
python -m dynamo.frontend \
    --router-mode kv \
    --router-conditional-disagg \
    --router-conditional-disagg-config '{"policy":"isl_or_load","eff_isl_threshold":4096,"eff_isl_ratio_threshold":0.6}'
```

## Backend Requirements

Conditional disaggregation requires decode-worker KV visibility. The router uses decode-side KV events to estimate effective ISL and decide whether local prefill+decode is cheaper than remote prefill.

Configure workers as follows:

| Backend | Requirement |
| --- | --- |
| vLLM | Pass `--kv-events-config` on prefill and decode workers. |
| TensorRT-LLM | Pass `--publish-kv-events` on prefill AND decode workers to opt them into KV-aware routing. |
| SGLang | Not supported yet. |

If decode workers do not publish KV events, the router cannot accurately assess bypass conditions.

Append these additional flags to tune the conditional disaggregation policy:

> [!NOTE]
> We recommend tuning these values against your workload and deployment configuration.

For ISL-based policies, `effective ISL` is the request prompt length after subtracting the selected decode worker's cached prefix overlap. The absolute threshold limits the number of prompt tokens the decode worker may need to compute locally. The ratio threshold limits that local work as a fraction of the raw prompt length. These thresholds measure both "how much compute does this request require" (absolute) and "how compute/memory-bound is this request, due to the ratio of its computed / cached KV cache" (ratio), respectively.

As a tuning starting point, we recommend choosing thresholds based on your workload's expected effective-ISL and the `effective ISL : raw ISL` ratio distribution. For example, with `isl_bounding`, setting the absolute threshold to p25 of the workload's effective ISL would make the absolute-threshold check pass for roughly 25% of requests.

| Flag | Default | Use |
| --- | --- | --- |
| `--router-conditional-disagg` | Disabled | Enables conditional disaggregation. Requires `--router-mode kv`, `--router-kv-events`, and separate prefill/decode worker pools. |
| `--router-conditional-disagg-config` | Unset | JSON object for policy settings. Supported fields: `policy`, `eff_isl_threshold`, `eff_isl_ratio_threshold`, `prefill_busy_threshold`, and `decode_busy_threshold`. |

The config fields map to these policy settings:

| Config Field | Default | Use |
| --- | --- | --- |
| `policy` | `isl_bounding` | Selects the policy: `isl_bounding`, `prefill_load`, or `isl_or_load`. |
| `eff_isl_threshold` | `2048` | For `isl_bounding` and the ISL condition within `isl_or_load`, require effective ISL to be below this many tokens. |
| `eff_isl_ratio_threshold` | `0.7` | For `isl_bounding` and the ISL condition within `isl_or_load`, require the `effective ISL : raw ISL` ratio to be below this value. Must be in `[0.0, 1.0]`. |
| `prefill_busy_threshold` | Unset | Sets the prefill busy threshold for `prefill_load` and `isl_or_load`. When unset, those policies inherit `--router-queue-threshold` if it is set. |
| `decode_busy_threshold` | Unset | Decode-busy guard. When unset, the guard is disabled. When set, conditional disaggregation uses the normal remote-prefill path if the selected decode worker's projected active decode KV blocks exceed this fraction of KV capacity. If the selected decode worker does not report KV capacity, the router falls back to the normal prefill-decode disaggregation path instead. This uses router-side active decode block accounting. |

The available policies are:

| Policy | Local Prefill+Decode Condition |
| --- | --- |
| `isl_bounding` | Effective ISL is below `eff_isl_threshold` AND the effective/raw ISL ratio is below `eff_isl_ratio_threshold`. |
| `prefill_load` | The selected prefill worker is above the configured prefill busy threshold. |
| `isl_or_load` | Either the `isl_bounding` condition OR the `prefill_load` condition is true. |
