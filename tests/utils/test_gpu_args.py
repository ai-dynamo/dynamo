# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for the TensorRT-LLM override args built by gpu_utils.sh.

This shells out to the real ``build_trtllm_override_args_with_mem``, so it
fails if the helper stops emitting what the GPU-parallel scheduler relies on.
Nothing in CI runs ``bash examples/common/gpu_utils.sh --self-test``, so this is
the only automated guard on what that helper emits.
"""

from __future__ import annotations

import json
import os

import pytest

from tests.utils import gpu_args

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _env_without_profile_overrides() -> dict[str, str]:
    """A copy of the environment with the scheduler's override vars removed.

    The helper reads them from the environment, so a stray value inherited from
    an enclosing profiled run would otherwise decide the result.
    """
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("_PROFILE_OVERRIDE_")
    }


def test_trtllm_override_caps_tokens_without_imposing_a_memory_fraction() -> None:
    """The profiler sizes the token cap; the engine config sizes the memory.

    Thirteen launch scripts share this helper, each with its own engine config
    declaring its own ``free_gpu_memory_fraction`` -- 0.85 for qwen3, 0.30 for
    the multimodal configs. Those values reach the worker and survive the
    override merge, so naming a fraction here would impose one launcher's memory
    policy on all of them.
    """
    env = _env_without_profile_overrides()
    env["_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS"] = "2592"

    args = gpu_args.build_trtllm_override_args(env)

    assert args[0] == "--override-engine-args"
    kv_cache_config = json.loads(args[1])["kv_cache_config"]
    assert kv_cache_config["max_tokens"] == 2592
    assert "free_gpu_memory_fraction" not in kv_cache_config

    # No token cap: an unprofiled launch keeps the engine config's own sizing.
    assert gpu_args.build_trtllm_override_args(_env_without_profile_overrides()) == []
