# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from dynamo.common.multimodal.processed_media import ProcessedField, ProcessedMedia
from dynamo.vllm.multimodal_utils.processed_media_adapter import _combine_fields

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def _media(t: int) -> ProcessedMedia:
    patches = t * 4
    return ProcessedMedia(
        modality="video",
        fields={
            "pixel_values_videos": ProcessedField(
                np.zeros((patches, 6), dtype=np.float32),
                {"kind": "flat", "sizes_key": "patches_per_video"},
                False,
                True,
            ),
            "patches_per_video": ProcessedField(
                np.asarray([patches]), {"kind": "batched"}, True, False
            ),
            "video_grid_thw": ProcessedField(
                np.asarray([[t, 2, 2]]), {"kind": "batched"}, True, True
            ),
            "timestamps": ProcessedField(
                np.arange(t, dtype=np.float64)[None, :],
                {"kind": "batched"},
                True,
                True,
            ),
        },
        feature_token_counts=[t],
        original_sizes=[(2, 2)],
        content_hashes=[str(t)],
    )


def test_combines_layouts_without_model_specific_adapter_logic():
    inputs, configs = _combine_fields([_media(1), _media(2)])

    assert inputs["pixel_values_videos"].shape == (12, 6)
    assert inputs["video_grid_thw"].shape == (2, 3)
    assert [value.shape for value in inputs["timestamps"]] == [(1,), (2,)]
    assert "patches_per_video" not in inputs
    assert set(configs) == set(inputs)
    assert torch.equal(inputs["video_grid_thw"][1], torch.tensor([2, 2, 2]))


def test_single_item_reuses_numpy_storage_without_cat_or_stack():
    media = _media(2)

    inputs, _ = _combine_fields([media])

    pixel_values = media.fields["pixel_values_videos"].value
    grid = media.fields["video_grid_thw"].value
    timestamps = media.fields["timestamps"].value
    assert (
        inputs["pixel_values_videos"].data_ptr()
        == torch.from_numpy(pixel_values).data_ptr()
    )
    assert inputs["video_grid_thw"].data_ptr() == torch.from_numpy(grid).data_ptr()
    assert inputs["timestamps"].data_ptr() == torch.from_numpy(timestamps).data_ptr()
