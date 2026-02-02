# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang multimodal multi-image support (DIS-1266)."""

import pytest

from dynamo.sglang.protocol import (
    MultiModalInput,
    MultiModalInputGroup,
    MultiModalRequest,
    PreprocessedRequest,
    SamplingOptions,
    SglangMultimodalRequest,
    StopConditions,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.pre_merge,
]


class TestMultiModalInputGroup:
    """Tests for MultiModalInputGroup class."""

    def test_create_empty_group(self):
        """Test creating an empty MultiModalInputGroup."""
        group = MultiModalInputGroup()
        assert group.multimodal_input is not None
        assert group.multimodal_input.image_url is None
        assert group.multimodal_input.video_url is None
        assert group.image_grid_thw is None
        assert group.embeddings_shape is None
        assert group.serialized_request is None

    def test_create_group_with_image_url(self):
        """Test creating a group with an image URL."""
        image_url = "http://example.com/image.jpg"
        group = MultiModalInputGroup(
            multimodal_input=MultiModalInput(image_url=image_url)
        )
        assert group.multimodal_input.image_url == image_url
        assert group.multimodal_input.video_url is None

    def test_create_group_with_video_url(self):
        """Test creating a group with a video URL."""
        video_url = "http://example.com/video.mp4"
        group = MultiModalInputGroup(
            multimodal_input=MultiModalInput(video_url=video_url)
        )
        assert group.multimodal_input.video_url == video_url
        assert group.multimodal_input.image_url is None

    def test_group_with_embeddings_metadata(self):
        """Test group with embeddings shape and grid info."""
        group = MultiModalInputGroup(
            multimodal_input=MultiModalInput(image_url="http://example.com/image.jpg"),
            image_grid_thw=[[1, 14, 14]],
            embeddings_shape=(1, 196, 1536),
        )
        assert group.embeddings_shape == (1, 196, 1536)
        assert group.image_grid_thw == [[1, 14, 14]]

    def test_group_serialization(self):
        """Test that MultiModalInputGroup can be serialized to JSON."""
        group = MultiModalInputGroup(
            multimodal_input=MultiModalInput(image_url="http://example.com/image.jpg"),
            image_grid_thw=[[1, 14, 14]],
            embeddings_shape=(1, 196, 1536),
        )
        json_str = group.model_dump_json()
        assert "http://example.com/image.jpg" in json_str
        assert "196" in json_str

    def test_group_deserialization(self):
        """Test that MultiModalInputGroup can be deserialized from JSON."""
        json_str = '{"multimodal_input": {"image_url": "http://example.com/image.jpg"}, "embeddings_shape": [1, 196, 1536]}'
        group = MultiModalInputGroup.model_validate_json(json_str)
        assert group.multimodal_input.image_url == "http://example.com/image.jpg"
        assert group.embeddings_shape == (1, 196, 1536)


class TestSglangMultimodalRequest:
    """Tests for SglangMultimodalRequest with multiple images."""

    def _create_preprocessed_request(self, token_ids=None):
        """Helper to create a PreprocessedRequest."""
        return PreprocessedRequest(
            token_ids=token_ids or [1, 2, 3, 4, 5],
            stop_conditions=StopConditions(max_tokens=100),
            sampling_options=SamplingOptions(temperature=0.7),
        )

    def test_create_request_with_empty_inputs(self):
        """Test creating request with empty multimodal inputs list."""
        request = SglangMultimodalRequest(
            request=self._create_preprocessed_request(),
            multimodal_inputs=[],
        )
        assert len(request.multimodal_inputs) == 0

    def test_create_request_with_single_image(self):
        """Test creating request with a single image (backwards compatibility)."""
        group = MultiModalInputGroup(
            multimodal_input=MultiModalInput(image_url="http://example.com/image1.jpg")
        )
        request = SglangMultimodalRequest(
            request=self._create_preprocessed_request(),
            multimodal_inputs=[group],
        )
        assert len(request.multimodal_inputs) == 1
        assert (
            request.multimodal_inputs[0].multimodal_input.image_url
            == "http://example.com/image1.jpg"
        )

    def test_create_request_with_multiple_images(self):
        """Test creating request with multiple images."""
        groups = [
            MultiModalInputGroup(
                multimodal_input=MultiModalInput(
                    image_url=f"http://example.com/image{i}.jpg"
                )
            )
            for i in range(3)
        ]
        request = SglangMultimodalRequest(
            request=self._create_preprocessed_request(),
            multimodal_inputs=groups,
        )
        assert len(request.multimodal_inputs) == 3
        for i, group in enumerate(request.multimodal_inputs):
            assert (
                group.multimodal_input.image_url == f"http://example.com/image{i}.jpg"
            )

    def test_request_serialization_with_multiple_images(self):
        """Test serialization of request with multiple images."""
        groups = [
            MultiModalInputGroup(
                multimodal_input=MultiModalInput(
                    image_url=f"http://example.com/image{i}.jpg"
                ),
                embeddings_shape=(1, 100 + i * 50, 1536),
            )
            for i in range(2)
        ]
        request = SglangMultimodalRequest(
            request=self._create_preprocessed_request(),
            multimodal_inputs=groups,
        )
        json_str = request.model_dump_json()

        # Verify all images are in the serialized output
        assert "image0.jpg" in json_str
        assert "image1.jpg" in json_str
        assert "100" in json_str  # First embeddings shape
        assert "150" in json_str  # Second embeddings shape

    def test_request_deserialization_with_multiple_images(self):
        """Test deserialization of request with multiple images."""
        json_str = """{
            "request": {
                "token_ids": [1, 2, 3],
                "stop_conditions": {"max_tokens": 50},
                "sampling_options": {"temperature": 0.5}
            },
            "multimodal_inputs": [
                {"multimodal_input": {"image_url": "http://example.com/a.jpg"}, "embeddings_shape": [1, 100, 1536]},
                {"multimodal_input": {"image_url": "http://example.com/b.jpg"}, "embeddings_shape": [1, 200, 1536]}
            ]
        }"""
        request = SglangMultimodalRequest.model_validate_json(json_str)

        assert len(request.multimodal_inputs) == 2
        assert (
            request.multimodal_inputs[0].multimodal_input.image_url
            == "http://example.com/a.jpg"
        )
        assert request.multimodal_inputs[0].embeddings_shape == (1, 100, 1536)
        assert (
            request.multimodal_inputs[1].multimodal_input.image_url
            == "http://example.com/b.jpg"
        )
        assert request.multimodal_inputs[1].embeddings_shape == (1, 200, 1536)


class TestMultiModalRequestParsing:
    """Tests for parsing MultiModalRequest with multiple images."""

    def test_parse_request_with_single_image(self):
        """Test parsing a request with a single image."""
        request = MultiModalRequest(
            model="test-model",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "http://example.com/image.jpg"},
                        },
                    ],
                }
            ],
        )

        # Extract images from messages
        images = []
        for msg in request.messages:
            for item in msg.content:
                if item.type == "image_url":
                    images.append(item.image_url.url)

        assert len(images) == 1
        assert images[0] == "http://example.com/image.jpg"

    def test_parse_request_with_multiple_images_in_one_message(self):
        """Test parsing a request with multiple images in a single message."""
        request = MultiModalRequest(
            model="test-model",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "http://example.com/image1.jpg"},
                        },
                        {"type": "text", "text": "Compare these two images"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "http://example.com/image2.jpg"},
                        },
                    ],
                }
            ],
        )

        images = []
        for msg in request.messages:
            for item in msg.content:
                if item.type == "image_url":
                    images.append(item.image_url.url)

        assert len(images) == 2
        assert images[0] == "http://example.com/image1.jpg"
        assert images[1] == "http://example.com/image2.jpg"

    def test_parse_request_with_images_across_messages(self):
        """Test parsing a request with images across multiple messages."""
        request = MultiModalRequest(
            model="test-model",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First image:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "http://example.com/image1.jpg"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Second image:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "http://example.com/image2.jpg"},
                        },
                    ],
                },
            ],
        )

        images = []
        for msg in request.messages:
            for item in msg.content:
                if item.type == "image_url":
                    images.append(item.image_url.url)

        assert len(images) == 2
        assert images[0] == "http://example.com/image1.jpg"
        assert images[1] == "http://example.com/image2.jpg"


class TestTokenExpansion:
    """Tests for token expansion logic with multiple images."""

    def test_find_image_token_positions(self):
        """Test finding positions of image tokens in token sequence."""
        image_token_id = 151655  # Example Qwen image token ID
        token_ids = [1, 2, image_token_id, 3, 4, image_token_id, 5]

        positions = [i for i, t in enumerate(token_ids) if t == image_token_id]

        assert positions == [2, 5]

    def test_expand_single_token(self):
        """Test expanding a single image token."""
        image_token_id = 151655
        token_ids = [1, 2, image_token_id, 3]
        num_patches = 196

        pos = token_ids.index(image_token_id)
        expanded = (
            token_ids[:pos] + [image_token_id] * num_patches + token_ids[pos + 1 :]
        )

        assert len(expanded) == 3 + num_patches  # 1, 2, [196 tokens], 3
        assert expanded[:2] == [1, 2]
        assert expanded[-1] == 3
        assert expanded[2 : 2 + num_patches] == [image_token_id] * num_patches

    def test_expand_multiple_tokens_reverse_order(self):
        """Test expanding multiple image tokens in reverse order to preserve indices."""
        image_token_id = 151655
        token_ids = [1, image_token_id, 2, image_token_id, 3]
        num_patches_list = [100, 200]  # Different patch counts for each image

        positions = [i for i, t in enumerate(token_ids) if t == image_token_id]
        assert positions == [1, 3]

        # Expand in reverse order
        for idx in range(len(positions) - 1, -1, -1):
            pos = positions[idx]
            num_patches = num_patches_list[idx]
            token_ids = (
                token_ids[:pos] + [image_token_id] * num_patches + token_ids[pos + 1 :]
            )

        # Verify final sequence
        # Original: [1, IMG, 2, IMG, 3]
        # After expanding second IMG (200 patches): [1, IMG, 2, IMG*200, 3]
        # After expanding first IMG (100 patches): [1, IMG*100, 2, IMG*200, 3]
        expected_length = 1 + 100 + 1 + 200 + 1  # 303
        assert len(token_ids) == expected_length

        # Check structure
        assert token_ids[0] == 1
        assert token_ids[1:101] == [image_token_id] * 100
        assert token_ids[101] == 2
        assert token_ids[102:302] == [image_token_id] * 200
        assert token_ids[302] == 3
