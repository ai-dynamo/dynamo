<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Test Model Size Guardrails

CI runs on shared L4 GPUs (24 GB VRAM). Loading multi-gigabyte models for tests
that only exercise plumbing wastes minutes per run and slows every other PR in
the queue.

## Scope

Applies only to tests that **load or reference a real model** — typically e2e
and integration tests under `tests/serve/`, `tests/router/`, `tests/mm_router/`,
`tests/fault_tolerance/`, `tests/kvbm_integration/`, `tests/basic/`, and
similar. Skip this rule when a test mocks the model or never references a
Hugging Face model id, `--model` arg, or framework-specific model path.

## Flag and suggest a smaller model when

A test loads or references a model that is:

- **≥7B parameters** (matches `*-7B*`, `*-8B*`, `*-13B*`, `*-32B*`, `*-70B*`,
  etc. in the model id), **or**
- **≥10 GB on disk** in any common dtype.

If the test is exercising plumbing (routing, scheduling, frontend, fault
tolerance), suggest one of the small models already used in this repo:

- Text: `Qwen/Qwen3-0.6B`
- Multimodal: `Qwen/Qwen3-VL-2B-Instruct` (or `Qwen/Qwen2-VL-2B-Instruct`)
- Small text fallback: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

## Hard NO GO: peak VRAM > 24 GB

If `@pytest.mark.profiled_vram_gib(N)` (or equivalent) has `N > 24`, the test
**cannot run** on pre/post/nightly CI runners. Flag it as blocking, not as a
suggestion: either shrink the model / batch / KV cap below 24 GB, or move the
test to a marker that runs on a larger-VRAM tier (and confirm such a tier
exists in the CI matrix).

See `tests/README.md` for the marker reference and how to derive the number
via `tests/utils/profile_pytest.py`.

## Missing markers: ask the author to profile

If a test loads a model but does **not** carry a
`@pytest.mark.profiled_vram_gib(N)` marker (plus the framework-specific KV
marker — `requested_vllm_kv_cache_bytes`, `requested_sglang_kv_tokens`, or
`requested_trtllm_kv_tokens`), the parallel scheduler can't size it and
`--max-vram-gib` filtering deselects it as unsafe. Suggest the author profile
the test with `tests/utils/profile_pytest.py` and add the resulting markers;
see `tests/README.md` for the exact invocation per framework.

## Carve-out for tests that genuinely need the larger model

Leave the model alone if the test:

1. Has a comment explaining what only the larger model exposes.
2. Is gated outside `pre_merge` (e.g. `@pytest.mark.post_merge`).
3. Carries a `@pytest.mark.profiled_vram_gib(...)` marker so the parallel
   scheduler can size it.
