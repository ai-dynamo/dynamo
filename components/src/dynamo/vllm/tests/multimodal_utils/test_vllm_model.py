# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DynamoMultimodalEmbeddingCacheConnector."""


import pytest
import torch

from dynamo.vllm.multimodal_utils.model import (
    construct_grid_thw_decode_mm_data,
    construct_mm_data,
    is_grid_thw_image_data,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class TestMultiModalUtils:
    def test_construct_grid_thw_decode_mm_data(self):
        max_rounds = int(torch.finfo(torch.float16).max) + 2
        expected_image_grid_thw_tensor = torch.tensor([16, 16])
        for i in range(max_rounds):
            # Should not raise any exception
            try:
                mm_data = construct_grid_thw_decode_mm_data(
                    image_grid_thw=[16, 16],
                    embeddings_shape=[2, 1024],
                    request_id=str(i),
                )
            except Exception as e:
                pytest.fail(
                    f"construct_grid_thw_decode_mm_data raised {type(e).__name__} on round {i}: {e}"
                )
            assert "image" in mm_data
            assert "image_grid_thw" in mm_data["image"]
            assert "image_embeds" in mm_data["image"]
            assert torch.allclose(
                mm_data["image"]["image_grid_thw"], expected_image_grid_thw_tensor
            )
            # Embedding values are randomly genearted as placehodler, we only check the shape
            assert mm_data["image"]["image_embeds"].shape == (2, 1024)

    def test_construct_mm_data_uses_grid_metadata_when_present(self):
        image_embeds = torch.randn(1, 4, 8)

        mm_data = construct_mm_data(
            model="test-model",
            embeddings_dtype=torch.float16,
            image_embeds=image_embeds,
            image_grid_thw=[[1, 2, 2]],
        )

        assert is_grid_thw_image_data(mm_data["image"])
        assert mm_data["image"]["image_embeds"].shape == (4, 8)
        assert mm_data["image"]["image_grid_thw"].tolist() == [[1, 2, 2]]

    def test_construct_mm_data_uses_plain_image_tensor_without_grid_metadata(self):
        image_embeds = torch.randn(1, 4, 8)

        mm_data = construct_mm_data(
            model="test-model",
            embeddings_dtype=torch.float16,
            image_embeds=image_embeds,
        )

        assert isinstance(mm_data["image"], torch.Tensor)
        assert not is_grid_thw_image_data(mm_data["image"])
