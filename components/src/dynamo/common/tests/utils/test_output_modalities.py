# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for output_modalities utility functions."""

import pytest

from dynamo.common.utils.output_modalities import OutputModality, get_output_modalities
from dynamo.llm import ModelType

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
]


def model_type_flags(result) -> set:
    """Extract the set of flag names from a ModelType flags."""
    return set(str(result).split(","))


class TestOutputModality:
    """Tests for the OutputModality enum."""

    def test_from_name_text(self):
        """Test lookup of text modality."""
        modality = OutputModality.from_name("text")
        assert modality is OutputModality.TEXT
        assert modality.value == (ModelType.Chat, ModelType.Completions)

    def test_from_name_image(self):
        """Test lookup of image modality."""
        modality = OutputModality.from_name("image")
        assert modality is OutputModality.IMAGE
        assert modality.value == (ModelType.Images, ModelType.Chat)

    def test_from_name_video(self):
        """Test lookup of video modality."""
        modality = OutputModality.from_name("video")
        assert modality is OutputModality.VIDEO
        assert modality.value == (ModelType.Videos,)

    def test_from_name_audio(self):
        """Test lookup of audio modality."""
        modality = OutputModality.from_name("audio")
        assert modality is OutputModality.AUDIO
        assert modality.value == (ModelType.Audios,)

    def test_from_name_case_insensitive(self):
        """Test that from_name is case-insensitive."""
        assert OutputModality.from_name("TEXT") is OutputModality.TEXT
        assert OutputModality.from_name("Text") is OutputModality.TEXT
        assert OutputModality.from_name("IMAGE") is OutputModality.IMAGE
        assert OutputModality.from_name("Image") is OutputModality.IMAGE

    def test_from_name_invalid_raises(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown output modality"):
            OutputModality.from_name("invalid")

    def test_valid_names(self):
        """Test that valid_names returns all expected lowercase names."""
        names = OutputModality.valid_names()
        assert names == {"text", "image", "video", "audio"}


class TestGetOutputModalities:
    """Tests for the get_output_modalities function."""

    def test_single_text(self):
        """Test text modality produces chat and completions flags."""
        result = get_output_modalities(["text"], "model-repo")
        assert result is not None
        assert model_type_flags(result) == {"chat", "completions"}

    def test_single_image(self):
        """Test image modality produces images and chat flags."""
        result = get_output_modalities(["image"], "model-repo")
        assert result is not None
        assert model_type_flags(result) == {"images", "chat"}

    def test_single_video(self):
        """Test video modality produces videos flag."""
        result = get_output_modalities(["video"], "model-repo")
        assert result is not None
        assert model_type_flags(result) == {"videos"}

    def test_single_audio(self):
        """Test audio modality produces audios flag."""
        result = get_output_modalities(["audio"], "model-repo")
        assert result is not None
        assert model_type_flags(result) == {"audios"}

    def test_combined_text_image(self):
        """Test combining text and image modalities."""
        result = get_output_modalities(["text", "image"], "model-repo")
        assert result is not None
        assert model_type_flags(result) == {"chat", "completions", "images"}

    def test_combined_all(self):
        """Test combining all modalities."""
        result = get_output_modalities(
            ["text", "image", "audio", "video"], "model-repo"
        )
        assert result is not None
        assert model_type_flags(result) == {
            "chat",
            "completions",
            "images",
            "audios",
            "videos",
        }

    def test_empty_input_returns_none(self):
        """Test empty input list returns None."""
        result = get_output_modalities([], "model-repo")
        assert result is None

    def test_invalid_modality_raises(self):
        """Test invalid modality name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown output modality"):
            get_output_modalities(["invalid"], "model-repo")
