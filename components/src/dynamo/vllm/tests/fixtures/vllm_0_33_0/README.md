<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Pinned vLLM CPU references

The `.py.txt` files are byte-for-byte upstream sources from vLLM `v0.33.0`, commit `4ba787ec9959e566d3b587a796dff07673324da5`. Their original Apache-2.0 headers are preserved. The fixture reader checks each complete file against the Git blob ID in `decision_native_loader.PINNED_SOURCE_BLOBS` before parsing or executing it. Do not edit these upstream copies.

These files make native strategy, reducer, cancellation, budget, and overload CPU tests mandatory without an installed native decisions API or a configured source checkout. An explicit `DYNAMO_VLLM_SOURCE_REPO` override still reads that repository and fails on missing or changed source; it never falls back to these files. `DYNAMO_VLLM_REQUIRE_PINNED_SOURCE=1` remains compatible because bundled references are already required and verified.

`api-contract.json` is an AST projection of the exact source-owned signatures, selected sampling defaults, and label limit used by the lightweight CPU stand-ins. Each projection records its owner's pinned Git blob ID. Reproduce or verify it against a local source repository with:

```bash
python3 -m dynamo.vllm.tests.decision_stub_contract --source-repo /path/to/vllm --check
python3 -m dynamo.vllm.tests.decision_stub_contract --source-repo /path/to/vllm
```

The second command prints the regenerated JSON. The fidelity tests always check the CPU stand-ins against this contract and also regenerate the projection when an explicit repository override is present. The stand-ins cover only the exercised API subset, not tokenizer quality, full renderer behavior, scheduler execution, physical GPU abort settlement, or model quality. Tests that explicitly probe an installed runtime remain separate optional smoke checks.
