<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A.X-K2 vLLM patches

These four Python patches add A.X-K2 support to the **vLLM 0.26.0** package in
the Dynamo 1.4.1 CUDA vLLM runtime. The model patches originate from
[SKT-AI/vllm](https://github.com/SKT-AI/vllm), whose
[axk2-v0.23.0 branch](https://github.com/SKT-AI/vllm/tree/axk2-v0.23.0)
contains the provider's vLLM 0.23.0 implementation.

The [runtime Dockerfile template](../../../templates/vllm_runtime.Dockerfile)
applies all four patches to the installed vLLM package in numeric filename order:
`0001`, `0002`, `0003`, then `0004`. Patch `0002` depends on the model added by
`0001`.

| Patch | Source | Purpose |
| --- | --- | --- |
| [0001](0001-Model-Add-SKT-A.X-K2-axk2.patch) | [SKT model support](https://github.com/SKT-AI/vllm/commit/ab93b327a43717e6712cc8e76fa6a3bb6985bdbc), adapted for vLLM 0.26.0 | Add the A.X-K2 model, configuration, and registration. |
| [0002](0002-Model-axk2-reject-non-fused-attention-gate-checkpoin.patch) | [SKT checkpoint validation](https://github.com/SKT-AI/vllm/commit/57c7f3255895f3964d1f996edbcd5f988e0a787d) | Reject checkpoints whose attention-output gate has not been fused offline. |
| [0003](0003-Spec-Decode-honor-DSpark-anchor-sampling-layout.patch) | Local DSpark configuration fix | Derive the anchor layout from the checkpoint's `sample_from_anchor` setting. |
| [0004](0004-fix-kv-cache-support-sparse-MLA-targets-with-SWA-dra.patch) | Backport of [vLLM commit e18f0037](https://github.com/vllm-project/vllm/commit/e18f0037a5d54dc2ead5896af896305f2bf57496) | Allow sparse-MLA targets and sliding-window drafts to share a compatible KV-cache allocation. |

The SKT links identify the corresponding commits in the published fork; the
patch headers retain their original commit IDs.

The runtime retains the upstream vLLM 0.26.0 DSpark implementation. At build time,
[validate_axk2_port.py](../../../deps/vllm/validate_axk2_port.py) checks A.X-K2
and DSpark model registration, DSpark configuration conversion, and the
sparse-MLA plus sliding-window-draft cache-grouping fallback.

Deployment and benchmark instructions are provided separately in the
[A.X-K2 recipe PR](https://github.com/ai-dynamo/dynamo/pull/14635).
