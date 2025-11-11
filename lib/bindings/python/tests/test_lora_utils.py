# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test suite for LoRA utility functions exposed via Python bindings."""


class TestLoraNameToHashId:
    """Tests for the lora_name_to_hash_id function."""

    def test_import_function(self):
        """Test that lora_name_to_hash_id can be imported from dynamo.llm."""
        from dynamo.llm import lora_name_to_hash_id

        assert callable(lora_name_to_hash_id)

    def test_returns_positive_integer(self):
        """Test that the function returns a positive integer."""
        from dynamo.llm import lora_name_to_hash_id

        result = lora_name_to_hash_id("test_lora")
        assert isinstance(result, int)
        assert result > 0

    def test_returns_signed_int32_range(self):
        """Test that returned values are in signed int32 range (1 to 2,147,483,647)."""
        from dynamo.llm import lora_name_to_hash_id

        max_int32 = 0x7FFFFFFF  # 2,147,483,647
        result = lora_name_to_hash_id("test_lora")
        assert 1 <= result <= max_int32

    def test_deterministic_hash(self):
        """Test that the same input always produces the same output."""
        from dynamo.llm import lora_name_to_hash_id

        name = "my_lora_adapter"
        result1 = lora_name_to_hash_id(name)
        result2 = lora_name_to_hash_id(name)
        assert result1 == result2

    def test_different_names_produce_different_ids(self):
        """Test that different names produce different IDs."""
        from dynamo.llm import lora_name_to_hash_id

        id1 = lora_name_to_hash_id("lora_adapter_1")
        id2 = lora_name_to_hash_id("lora_adapter_2")
        assert id1 != id2

    def test_empty_string(self):
        """Test that empty string produces a valid ID."""
        from dynamo.llm import lora_name_to_hash_id

        result = lora_name_to_hash_id("")
        assert isinstance(result, int)
        assert result > 0

    def test_special_characters(self):
        """Test that names with special characters work correctly."""
        from dynamo.llm import lora_name_to_hash_id

        names = [
            "lora-with-dashes",
            "lora_with_underscores",
            "lora.with.dots",
            "lora/with/slashes",
            "lora@special#chars$123",
        ]
        results = [lora_name_to_hash_id(name) for name in names]

        # All should be valid
        for result in results:
            assert isinstance(result, int)
            assert result > 0
            assert result <= 0x7FFFFFFF

        # All should be unique
        assert len(results) == len(set(results))

    def test_long_names(self):
        """Test that very long names work correctly."""
        from dynamo.llm import lora_name_to_hash_id

        long_name = "a" * 10000
        result = lora_name_to_hash_id(long_name)
        assert isinstance(result, int)
        assert result > 0

    def test_consistency_across_multiple_calls(self):
        """Test consistency with multiple different names."""
        from dynamo.llm import lora_name_to_hash_id

        test_names = [f"lora_{i}" for i in range(100)]
        # Call twice for each name
        results_first = [lora_name_to_hash_id(name) for name in test_names]
        results_second = [lora_name_to_hash_id(name) for name in test_names]

        # Should be identical
        assert results_first == results_second

        # All should be in valid range
        for result in results_first:
            assert 1 <= result <= 0x7FFFFFFF

    def test_matches_repeated_calls(self):
        """Test that repeated calls produce the same result."""
        from dynamo.llm import lora_name_to_hash_id

        name = "my_lora_adapter"
        results = set()
        for i in range(100):
            results.add(lora_name_to_hash_id(name))
        assert len(results) == 1
