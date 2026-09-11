# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the media passthrough sanitizer.

These run without any engine or model dependency: the point of the
sanitizer is to refuse a dangerous request field before anything is loaded,
so the test proves the rejection with no network and no model activity.
"""

import pytest

from dynamo.common.protocols import (
    MEDIA_PASSTHROUGH_KEY,
    MediaPassthroughRejected,
    sanitize_media_passthrough,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _wrap(knobs):
    return {MEDIA_PASSTHROUGH_KEY: knobs}


def test_none_and_empty_are_empty():
    assert sanitize_media_passthrough(None) == {}
    assert sanitize_media_passthrough({}) == {}
    assert sanitize_media_passthrough(_wrap({})) == {}


def test_benign_knobs_pass_through():
    knobs = {
        "backend_custom_knob": 0.5,
        "denoise_strength": 0.8,
        "generate_sound": True,
    }
    assert sanitize_media_passthrough(_wrap(knobs)) == knobs


@pytest.mark.parametrize(
    "key",
    [
        "frame_interpolation_model_path",  # the reported RCE field
        "guardrails",  # the reported guardrail bypass
        "extra_args",  # the container itself
        "vae_checkpoint",
        "unet_ckpt",
        "lora_repo",
        "hf_hub_id",
        "weights_url",
        "source_uri",
        "custom_local_dir",
        "safety_checker",
        "nsfw_filter",
        "content_moderation",
        "_private_attr",
    ],
)
def test_reserved_and_shaped_keys_are_rejected(key):
    with pytest.raises(MediaPassthroughRejected):
        sanitize_media_passthrough(_wrap({key: "x", "harmless": 1}))


def test_rejection_names_the_offending_key():
    with pytest.raises(
        MediaPassthroughRejected, match="frame_interpolation_model_path"
    ):
        sanitize_media_passthrough(
            _wrap({"frame_interpolation_model_path": "attacker/repo"})
        )


def test_rejected_is_a_value_error():
    # Request building catches ValueError and returns a client error, so the
    # sanitizer's exception must be a ValueError subclass.
    assert issubclass(MediaPassthroughRejected, ValueError)


def test_original_bucket_is_not_mutated():
    bucket = _wrap({"backend_custom_knob": 1})
    sanitize_media_passthrough(bucket)["backend_custom_knob"] = 999
    assert bucket[MEDIA_PASSTHROUGH_KEY]["backend_custom_knob"] == 1
