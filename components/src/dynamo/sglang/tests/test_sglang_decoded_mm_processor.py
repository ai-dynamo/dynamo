# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from dynamo.sglang.request_handlers.llm.decode_handler import (
    _decoded_items,
    _require_formatted_prompt,
)
from dynamo.sglang.request_handlers.llm.decoded_mm_processor import (
    DecodedMmProcessor,
    _normalize_video_metadata,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def test_normalize_video_metadata_unwraps_variant():
    raw = {
        "Video": {
            "source_fps": 30.0,
            "source_duration": 30.0,
            "source_total_frames": 900,
            "frames_indices": [0, 28, 56],
            "sampled_timestamps": [0.0, 0.93, 1.86],
        }
    }
    meta = _normalize_video_metadata(raw)
    assert meta["fps"] == 30.0
    assert meta["duration"] == 30.0
    # total_num_frames is the SOURCE count, not the sampled count.
    assert meta["total_num_frames"] == 900
    assert isinstance(meta["frames_indices"], np.ndarray)
    assert meta["frames_indices"].dtype == np.int64
    assert meta["frames_indices"].tolist() == [0, 28, 56]
    assert meta["video_backend"] == "dynamo"


def test_normalize_video_metadata_accepts_flat_dict():
    # Defensive: a dict already unwrapped (no "Video" key) is used as-is.
    flat = {
        "source_fps": 24.0,
        "source_duration": 2.0,
        "source_total_frames": 48,
        "frames_indices": [0, 24],
    }
    meta = _normalize_video_metadata(flat)
    assert meta["fps"] == 24.0
    assert meta["total_num_frames"] == 48


def test_normalize_video_metadata_requires_metadata():
    with pytest.raises(ValueError, match="missing metadata"):
        _normalize_video_metadata(None)


def test_assemble_dict_video_only_filters_keys():
    processor_output = {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
        "pixel_values_videos": "pvv",
        "video_grid_thw": "vgt",
        # second_per_grid_ts intentionally absent (Qwen3-VL omits it).
    }
    combined = DecodedMmProcessor._assemble_dict(
        processor_output, has_image=False, has_video=True
    )
    assert combined == {
        "format": "processor_output",
        "pixel_values_videos": "pvv",
        "video_grid_thw": "vgt",
    }
    # Non-feature keys and image keys are dropped.
    assert "input_ids" not in combined
    assert "pixel_values" not in combined


def test_assemble_dict_image_and_video():
    processor_output = {
        "pixel_values": "pv",
        "image_grid_thw": "igt",
        "pixel_values_videos": "pvv",
        "video_grid_thw": "vgt",
    }
    combined = DecodedMmProcessor._assemble_dict(
        processor_output, has_image=True, has_video=True
    )
    assert combined["format"] == "processor_output"
    assert combined["pixel_values"] == "pv"
    assert combined["image_grid_thw"] == "igt"
    assert combined["pixel_values_videos"] == "pvv"
    assert combined["video_grid_thw"] == "vgt"


def test_decoded_items_filters_url_variants():
    mm_data = {
        "video_url": [
            {"Decoded": {"shape": [8, 4, 4, 3]}},
            {"Url": "https://example.com/v.mp4"},
        ]
    }
    items = _decoded_items(mm_data, "video_url")
    assert items == [{"shape": [8, 4, 4, 3]}]
    assert _decoded_items(mm_data, "image_url") == []


def test_require_formatted_prompt():
    assert _require_formatted_prompt({"extra_args": {"formatted_prompt": "hi"}}) == "hi"
    with pytest.raises(ValueError, match="formatted_prompt"):
        _require_formatted_prompt({"extra_args": {}})
    with pytest.raises(ValueError, match="formatted_prompt"):
        _require_formatted_prompt({})
