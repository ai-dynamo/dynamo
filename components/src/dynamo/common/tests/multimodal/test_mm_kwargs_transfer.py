# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MM kwargs NIXL transfer (sender/receiver)."""

from unittest.mock import MagicMock

import pytest

from dynamo.common.multimodal.mm_kwargs_transfer import (
    MmKwargsSender,
    MmKwargsTransferMetadata,
    TensorTransferSpec,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class TestMmKwargsTransferMetadata:
    """Tests for the Pydantic metadata model."""

    def test_roundtrip_serialization(self):
        """Metadata serializes and deserializes correctly."""
        spec = TensorTransferSpec(
            field_name="pixel_values",
            shape=[100, 1176],
            dtype_str="float32",
            serialized_request="base64metadata==",
        )
        meta = MmKwargsTransferMetadata(
            modality="image",
            tensor_specs=[spec],
            mm_hashes=["abcd1234" * 8],
        )

        dumped = meta.model_dump()
        restored = MmKwargsTransferMetadata.model_validate(dumped)

        assert restored.modality == "image"
        assert len(restored.tensor_specs) == 1
        assert restored.tensor_specs[0].field_name == "pixel_values"
        assert restored.tensor_specs[0].shape == [100, 1176]
        assert restored.tensor_specs[0].dtype_str == "float32"
        assert restored.mm_hashes == ["abcd1234" * 8]

    def test_multiple_tensor_specs(self):
        """Multiple tensors (e.g., pixel_values + image_grid_thw)."""
        specs = [
            TensorTransferSpec(
                field_name="pixel_values",
                shape=[100, 1176],
                dtype_str="float32",
                serialized_request="meta1",
            ),
            TensorTransferSpec(
                field_name="image_grid_thw",
                shape=[1, 3],
                dtype_str="int64",
                serialized_request="meta2",
            ),
        ]
        meta = MmKwargsTransferMetadata(
            modality="image",
            tensor_specs=specs,
            mm_hashes=["hash1", "hash2"],
        )
        assert len(meta.tensor_specs) == 2
        assert meta.tensor_specs[0].field_name == "pixel_values"
        assert meta.tensor_specs[1].field_name == "image_grid_thw"


class TestMmKwargsSender:
    """Tests for the sender side (prepare method)."""

    @pytest.mark.asyncio
    async def test_prepare_with_no_features_returns_none(self):
        """Empty features list returns None."""
        sender = MmKwargsSender()
        meta, futures = await sender.prepare([], modality="image")
        assert meta is None
        assert futures == []

    @pytest.mark.asyncio
    async def test_prepare_with_no_data_returns_none(self):
        """Features with data=None are skipped."""
        from dynamo.common.multimodal.mm_kwargs_transfer import MmKwargsSender

        feat = MagicMock()
        feat.data = None
        feat.modality = "image"

        sender = MmKwargsSender()
        meta, futures = await sender.prepare([feat], modality="image")
        assert meta is None
        assert futures == []
