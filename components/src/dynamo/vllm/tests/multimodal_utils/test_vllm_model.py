# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DynamoMultimodalEmbeddingCacheConnector."""


import pytest
import torch
from PIL import Image

from dynamo.vllm.multimodal_utils.model import (
    construct_kimi_vision_chunk_data,
    construct_mm_data,
    construct_qwen_decode_mm_data,
    is_kimi_k25_model,
    is_qwen_vl_model,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class TestMultiModalUtils:
    def test_is_qwen_vl_model_detects_qwen35_local_config(self, tmp_path):
        model_dir = tmp_path / "qwen35"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            '{"architectures":["Qwen3_5MoeForConditionalGeneration"],'
            '"model_type":"qwen3_5_moe","vision_config":{"image_size":448}}'
        )

        assert is_qwen_vl_model(str(model_dir))

    def test_is_qwen_vl_model_rejects_qwen35_text_only_config(self, tmp_path):
        model_dir = tmp_path / "qwen35-text"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            '{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3_5_moe_text"}'
        )

        assert not is_qwen_vl_model(str(model_dir))

    def test_is_kimi_k25_model_detects_local_config(self, tmp_path):
        model_dir = tmp_path / "checkpoints"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            '{"architectures":["KimiK25ForConditionalGeneration"],"model_type":"kimi_k25"}'
        )

        assert is_kimi_k25_model(str(model_dir))

    def test_construct_kimi_vision_chunk_data(self):
        image = Image.new("RGB", (2, 2), color="red")

        mm_data = construct_kimi_vision_chunk_data([image])

        assert "vision_chunk" in mm_data
        assert len(mm_data["vision_chunk"]) == 1
        assert mm_data["vision_chunk"][0]["type"] == "image"
        assert mm_data["vision_chunk"][0]["image"] is image
        assert mm_data["vision_chunk"][0]["uuid"] is None

    def test_construct_mm_data_rejects_kimi_embeddings(self):
        with pytest.raises(ValueError, match="raw vision_chunk inputs"):
            construct_mm_data(
                model="moonshotai/Kimi-K2.5",
                embeddings_dtype=torch.float16,
                image_embeds=torch.randn(1, 4),
            )

    def test_construct_qwen_decode_mm_data(self):
        max_rounds = int(torch.finfo(torch.float16).max) + 2
        expected_image_grid_thw_tensor = torch.tensor([16, 16])
        for i in range(max_rounds):
            # Should not raise any exception
            try:
                mm_data = construct_qwen_decode_mm_data(
                    image_grid_thw=[16, 16],
                    embeddings_shape=[2, 1024],
                    request_id=str(i),
                )
            except Exception as e:
                pytest.fail(
                    f"construct_qwen_decode_mm_data raised {type(e).__name__} on round {i}: {e}"
                )
            assert "image" in mm_data
            assert "image_grid_thw" in mm_data["image"]
            assert "image_embeds" in mm_data["image"]
            assert torch.allclose(
                mm_data["image"]["image_grid_thw"], expected_image_grid_thw_tensor
            )
            # Embedding values are randomly genearted as placehodler, we only check the shape
            assert mm_data["image"]["image_embeds"].shape == (2, 1024)
