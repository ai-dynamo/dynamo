# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for resolving user-provided custom encoders."""

from types import SimpleNamespace

import pytest

from dynamo.vllm.multimodal_utils.custom_encoder import prepare_custom_encoder

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


def test_preflight_rejects_non_backend_class():
    with pytest.raises(TypeError, match="VisionEncoderBackend subclass"):
        prepare_custom_encoder(
            "json.JSONDecoder",
            SimpleNamespace(),
            SimpleNamespace(),
        )
