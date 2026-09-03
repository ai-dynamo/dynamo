# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from dynamo.sglang.protocol import CreateVideoRequest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_rejects_typed_input_references() -> None:
    with pytest.raises(ValidationError, match="not supported by the SGLang"):
        CreateVideoRequest.model_validate(
            {
                "prompt": "a cat",
                "model": "video-model",
                "input_references": [
                    {"type": "image", "source": "https://example.com/cat.png"}
                ],
            }
        )
