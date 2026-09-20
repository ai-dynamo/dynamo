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


def test_trtllm_override_states_kv_memory_fraction_only_with_token_cap() -> None:
    """The fraction belongs to the token cap, not to every launch.

    The override replaces the engine config's whole ``kv_cache_config``, so a
    token cap on its own silently leaves the memory fraction at the engine
    default instead of a chosen value. Stating it is only correct while the
    profiler is sizing the launch: emitting it unconditionally would impose one
    fraction on every launch script that sources the helper, whatever its own
    engine config declares.
    """
    env = _env_without_profile_overrides()
    env["_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS"] = "2592"

    args = gpu_args.build_trtllm_override_args(env)

    assert args[0] == "--override-engine-args"
    kv_cache_config = json.loads(args[1])["kv_cache_config"]
    assert kv_cache_config["max_tokens"] == 2592
    assert kv_cache_config["free_gpu_memory_fraction"] == 0.85

    # No token cap: an unprofiled launch keeps the engine config's own sizing.
    assert gpu_args.build_trtllm_override_args(_env_without_profile_overrides()) == []
