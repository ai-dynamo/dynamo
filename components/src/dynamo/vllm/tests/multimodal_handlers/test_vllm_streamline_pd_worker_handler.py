# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MultimodalStreamlinePdWorkerHandler."""

import pytest

from dynamo.vllm.multimodal_handlers import MultimodalStreamlinePdWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


class TestMultimodalStreamlinePdWorkerHandler:
    """Test suite for MultimodalStreamlinePdWorkerHandler."""

    pass
