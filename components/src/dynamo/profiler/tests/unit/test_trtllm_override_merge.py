# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for TRT-LLM --override-engine-args / --trtllm.* conflict resolution (GitHub #8659)."""

import json

import pytest

from dynamo.profiler.utils.config_modifiers.trtllm import _merge_overrides_into_args

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


def test_merge_overrides_into_existing_override_engine_args():
    """When --override-engine-args is already present, overrides merge into the
    JSON blob instead of appending mutually-exclusive --trtllm.* flags."""
    existing_json = json.dumps(
        {
            "cache_transceiver_config": {"backend": "DEFAULT"},
            "disable_overlap_scheduler": True,
            "kv_cache_config": {"tokens_per_block": 32},
        }
    )
    args = [
        "--model-path",
        "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
        "--override-engine-args",
        existing_json,
    ]

    result = _merge_overrides_into_args(
        args,
        {
            "kv_cache_config.enable_block_reuse": False,
            "disable_overlap_scheduler": False,
            "cache_transceiver_config": None,
        },
    )

    assert not any(a.startswith("--trtllm.") for a in result), (
        "Both --override-engine-args and --trtllm.* flags present; "
        "TRT-LLM will reject this combination"
    )

    idx = result.index("--override-engine-args")
    merged = json.loads(result[idx + 1])

    assert merged["disable_overlap_scheduler"] is False
    assert merged["cache_transceiver_config"] is None
    assert merged["kv_cache_config"]["enable_block_reuse"] is False
    assert merged["kv_cache_config"]["tokens_per_block"] == 32
