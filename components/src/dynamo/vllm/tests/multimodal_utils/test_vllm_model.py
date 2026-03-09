# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DynamoMultimodalEmbeddingCacheConnector."""


import pytest
import torch

from dynamo.vllm.multimodal_utils.model import construct_qwen_decode_mm_data

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class TestMultiModalUtils:
    def test_construct_qwen_decode_mm_data(self):
        max_rounds = int(torch.finfo(torch.float16).max) + 2
        for i in range(max_rounds):
            # Should not raise any exception
            try:
                construct_qwen_decode_mm_data(
                    image_grid_thw=[16, 16],
                    embeddings_shape=[2, 1024],
                    request_id=str(i),
                )
            except Exception as e:
                pytest.fail(
                    f"construct_qwen_decode_mm_data raised {type(e).__name__} on round {i}: {e}"
                )
