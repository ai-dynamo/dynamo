# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for profiler model-info helpers."""

import json

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
]

try:
    from dynamo.profiler.utils.model_info import get_mamba_cache_align_block_size
except ImportError as e:
    pytest.skip(f"Skip (missing dependency): {e}", allow_module_level=True)


def test_mamba_cache_align_block_size_from_local_config(tmp_path) -> None:
    """Mamba align floor follows vLLM's Mamba/attention page size."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "mamba_num_heads": 128,
                "mamba_head_dim": 64,
                "ssm_state_size": 128,
            }
        )
    )

    assert get_mamba_cache_align_block_size(tmp_path) == 8320
