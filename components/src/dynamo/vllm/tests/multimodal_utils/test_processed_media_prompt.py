# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

from dynamo.common.multimodal.model_prompt import expand_processed_media_prompt
from dynamo.common.multimodal.processed_media import ProcessedField, ProcessedMedia

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [90]


def media() -> ProcessedMedia:
    return ProcessedMedia(
        modality="video",
        fields={
            "video_grid_thw": ProcessedField(
                np.asarray([[1, 2, 2]]), {"kind": "batched"}, True, True
            ),
            "timestamps": ProcessedField(
                np.asarray([[0.25]]), {"kind": "batched"}, True, True
            ),
        },
        feature_token_counts=[1],
        original_sizes=[(2, 2)],
        content_hashes=["abc"],
    )


def config():
    return SimpleNamespace(
        model_type="qwen3_vl",
        vision_config=SimpleNamespace(spatial_merge_size=2),
        vision_start_token_id=1,
        video_token_id=2,
        vision_end_token_id=3,
    )


def test_qwen_prompt_expansion_uses_common_processed_media_contract():
    result = expand_processed_media_prompt(
        [9, 1, 2, 3, 8], [media()], Tokenizer(), config()
    )
    assert result.token_ids == [9, 90, 1, 2, 3, 8]
    assert (result.ranges[0].offset, result.ranges[0].length) == (1, 4)
    assert result.ranges[0].mask == [False, False, True, False]


def test_rejects_prompt_video_count_mismatch():
    with pytest.raises(ValueError, match="mismatch"):
        expand_processed_media_prompt([9], [media()], Tokenizer(), config())
