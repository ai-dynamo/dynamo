# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for runtime utilities."""

import pytest

from dynamo.common.utils.runtime import slugify_model_name

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class TestSlugifyModelName:
    """Test model name slugification with hash suffix."""

    def test_basic_models(self):
        """Test common HuggingFace model names."""
        result = slugify_model_name("Qwen/Qwen2.5-7B-Instruct")
        assert result.startswith("generate_qwen_qwen2_5-7b-instruct_")
        assert len(result.split("_")[-1]) == 8  # hash is 8 chars

        result = slugify_model_name("Qwen/Qwen3-0.6B")
        assert result.startswith("generate_qwen_qwen3-0_6b_")

        result = slugify_model_name("meta-llama/Llama-3.2-1B")
        assert result.startswith("generate_meta-llama_llama-3_2-1b_")

    def test_different_models_different_endpoints(self):
        """Ensure different models produce different endpoints."""
        ep1 = slugify_model_name("Qwen/Qwen2.5-7B-Instruct")
        ep2 = slugify_model_name("Qwen/Qwen3-0.6B")
        assert ep1 != ep2

    def test_same_model_same_endpoint(self):
        """Ensure same model produces consistent endpoint."""
        ep1 = slugify_model_name("Qwen/Qwen2.5-7B-Instruct")
        ep2 = slugify_model_name("Qwen/Qwen2.5-7B-Instruct")
        assert ep1 == ep2

    def test_special_characters_handled(self):
        """Test models with special characters."""
        result = slugify_model_name("model@name#test")
        assert result.startswith("generate_model_name_test_")
        assert "_" in result  # should have underscores replacing special chars

    def test_all_special_chars_fallback(self):
        """Test model name with all special characters."""
        result = slugify_model_name("$%@#!")
        assert result.startswith("generate_model_")
        assert len(result.split("_")[-1]) == 8  # hash suffix present

    def test_long_model_name_truncation(self):
        """Test very long model names are truncated."""
        long_name = "a" * 100
        result = slugify_model_name(long_name)
        # slug portion should be max 40 chars + "generate_" (9) + "_" (1) + hash (8) = max 58
        assert len(result) <= 58

    def test_case_insensitive(self):
        """Test that different cases of same name produce same slug (but different hash)."""
        ep1 = slugify_model_name("MODEL/Name")
        ep2 = slugify_model_name("model/name")
        # Slugs are same (both lowercase), but hashes differ
        assert ep1.split("_")[-1] != ep2.split("_")[-1]  # Different hashes

    def test_collision_resistance(self):
        """Test that similar model names don't collide."""
        # These would collide with pure slugification
        ep1 = slugify_model_name("model-name")
        ep2 = slugify_model_name("model_name")
        # Hash suffix prevents collision
        assert ep1 != ep2
