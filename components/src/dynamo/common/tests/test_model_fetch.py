# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for identifying model sources that engines cannot resolve themselves."""

import pytest

from dynamo.common.model_fetch import needs_local_model_path

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("ngc://nvstaging/nim/nemotron-3-ultra:hf-rl-052726-nvfp4-e9744cc", True),
        ("Qwen/Qwen3-0.6B", False),
        ("/data/llms/Qwen3-0.6B", False),
    ],
)
def test_needs_local_model_path(model: str, expected: bool) -> None:
    assert needs_local_model_path(model) is expected
