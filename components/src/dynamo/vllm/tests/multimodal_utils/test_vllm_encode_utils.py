# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from dynamo.vllm.multimodal_utils.encode_utils import (
    encode_image_embeddings,
    get_encoder_components,
    split_image_embeddings,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class DummyVisionTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.probe = torch.nn.Parameter(torch.zeros(1))

    def forward(self, pixel_values):
        return SimpleNamespace(last_hidden_state=pixel_values + 1)


class DummyProjector(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 2


class DummyVisionTowerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = DummyVisionTower()
        self.multi_modal_projector = DummyProjector()


class DummyGridAwareEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.probe = torch.nn.Parameter(torch.zeros(1))
        self.spatial_merge_size = 2

    def forward(self, pixel_values, grid_thw):
        del grid_thw
        return pixel_values.squeeze(0)


class DummyGridAwareWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(visual=DummyGridAwareEncoder())


class DummyImageFeaturesModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.probe = torch.nn.Parameter(torch.zeros(1))
        self.last_grid_thw = None

    def get_image_features(self, pixel_values, image_grid_thw=None):
        self.last_grid_thw = image_grid_thw
        return SimpleNamespace(pooler_output=pixel_values.squeeze(0) + 3)


class TestEncodeUtils:
    def test_get_encoder_components_prefers_vision_tower_and_projector(self):
        vision_encoder, projector = get_encoder_components(
            "test-model", DummyVisionTowerModel()
        )

        assert isinstance(vision_encoder, DummyVisionTower)
        assert isinstance(projector, DummyProjector)

    def test_get_encoder_components_discovers_nested_visual_encoder(self):
        vision_encoder, projector = get_encoder_components(
            "test-model", DummyGridAwareWrapper()
        )

        assert isinstance(vision_encoder, DummyGridAwareEncoder)
        assert projector is None

    def test_encode_image_embeddings_uses_image_feature_extractor_metadata(self):
        model = DummyImageFeaturesModel()
        image_embeds = {
            "pixel_values": torch.ones(1, 4, 6),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }

        embeddings = encode_image_embeddings(
            "test-model",
            image_embeds=image_embeds,
            vision_encoder=model,
        )

        assert model.last_grid_thw is not None
        assert model.last_grid_thw.tolist() == [[1, 2, 2]]
        assert embeddings.shape == (1, 4, 6)
        assert torch.allclose(embeddings, torch.full((1, 4, 6), 4.0))

    def test_split_image_embeddings_uses_spatial_merge_metadata(self):
        embeddings = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)
        image_embeds = {
            "image_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 2]]),
        }

        split_embeddings, image_grid_thw = split_image_embeddings(
            embeddings=embeddings,
            image_embeds=image_embeds,
            vision_encoder=DummyGridAwareEncoder(),
        )

        assert image_grid_thw == [[1, 4, 4], [1, 2, 2]]
        assert len(split_embeddings) == 2
        assert split_embeddings[0].shape == (4, 3)
        assert split_embeddings[1].shape == (1, 3)
