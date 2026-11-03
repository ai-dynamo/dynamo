# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Contract test for the SGLang sidecar's adaptive speculative decoding reserve.

SGLang's server info reports only the starting step count under
``--speculative-adaptive``, so the sidecar mirrors the largest step in SGLang's
default adaptive config (``SGLANG_DEFAULT_ADAPTIVE_MAX_STEPS`` in
``lib/sidecar/sglang/src/engine.rs``) to size requests that omit ``max_tokens``.
If this test fails, update that constant: a value below SGLang's makes SGLang
reject every such request as too long.
"""

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]

SIDECAR_DEFAULT_ADAPTIVE_MAX_STEPS = 7


def test_default_adaptive_config_largest_step_matches_sidecar():
    from sglang.srt.speculative.adaptive_spec_params import (
        resolve_candidate_steps_from_config,
    )

    candidate_steps = resolve_candidate_steps_from_config(cfg_path=None)

    assert max(candidate_steps) == SIDECAR_DEFAULT_ADAPTIVE_MAX_STEPS
