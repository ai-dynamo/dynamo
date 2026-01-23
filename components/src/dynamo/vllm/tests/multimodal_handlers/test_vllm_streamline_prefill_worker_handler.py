# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MultimodalStreamlinePrefillWorkerHandler."""

import pytest

from dynamo.vllm.multimodal_handlers import MultimodalStreamlinePrefillWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


class TestMultimodalStreamlinePrefillWorkerHandler:
    """Test suite for MultimodalStreamlinePrefillWorkerHandler."""

    pass
