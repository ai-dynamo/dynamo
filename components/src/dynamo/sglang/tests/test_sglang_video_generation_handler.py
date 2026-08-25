# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang video request validation."""

import pytest

try:
    from dynamo.sglang.request_handlers.video_generation.video_generation_handler import (
        validate_video_extra_params,
    )
except ImportError:
    pytest.skip(
        "SGLang video-generation dependencies not available", allow_module_level=True
    )

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


def test_sglang_video_rejects_nonempty_extra_params():
    with pytest.raises(ValueError, match="only supported by the vLLM-Omni"):
        validate_video_extra_params({"extra_params": {"task": "t2va"}})


@pytest.mark.parametrize("extra_params", [None, {}])
def test_sglang_video_allows_omitted_or_empty_extra_params(extra_params):
    validate_video_extra_params({"extra_params": extra_params})
