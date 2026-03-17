# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dynamic --trtllm.* flag parsing."""

import pytest

from dynamo.trtllm.dynamic_flags import infer_type, parse_dynamic_flags, set_nested


class TestInferType:
    def test_int(self):
        assert infer_type("42") == 42
        assert isinstance(infer_type("42"), int)

    def test_negative_int(self):
        assert infer_type("-1") == -1
        assert isinstance(infer_type("-1"), int)

    def test_zero(self):
        assert infer_type("0") == 0
        assert isinstance(infer_type("0"), int)

    def test_float(self):
        assert infer_type("0.9") == 0.9
        assert isinstance(infer_type("0.9"), float)

    def test_negative_float(self):
        assert infer_type("-0.5") == -0.5
        assert isinstance(infer_type("-0.5"), float)

    def test_bool_true(self):
        assert infer_type("true") is True
        assert infer_type("True") is True
        assert infer_type("TRUE") is True

    def test_bool_false(self):
        assert infer_type("false") is False
        assert infer_type("False") is False
        assert infer_type("FALSE") is False

    def test_string(self):
        assert infer_type("hello") == "hello"
        assert infer_type("/path/to/model") == "/path/to/model"

    def test_string_not_bool(self):
        # "yes", "no" should remain strings
        assert infer_type("yes") == "yes"
        assert infer_type("no") == "no"

    def test_scientific_notation(self):
        assert infer_type("1e3") == 1000.0
        assert isinstance(infer_type("1e3"), float)


class TestSetNested:
    def test_single_key(self):
        d: dict[str, object] = {}
        set_nested(d, ["key"], "value")
        assert d == {"key": "value"}

    def test_two_levels(self):
        d: dict[str, object] = {}
        set_nested(d, ["a", "b"], 42)
        assert d == {"a": {"b": 42}}

    def test_three_levels(self):
        d: dict[str, object] = {}
        set_nested(d, ["a", "b", "c"], True)
        assert d == {"a": {"b": {"c": True}}}

    def test_preserves_existing(self):
        d = {"a": {"x": 1}}
        set_nested(d, ["a", "y"], 2)
        assert d == {"a": {"x": 1, "y": 2}}

    def test_overwrites_leaf(self):
        d = {"a": {"b": "old"}}
        set_nested(d, ["a", "b"], "new")
        assert d == {"a": {"b": "new"}}


class TestParseDynamicFlags:
    def test_empty(self):
        assert parse_dynamic_flags([]) == {}

    def test_single_flat(self):
        result = parse_dynamic_flags(["--trtllm.max_batch_size", "32"])
        assert result == {"max_batch_size": 32}

    def test_nested(self):
        result = parse_dynamic_flags(
            ["--trtllm.kv_cache_config.free_gpu_memory_fraction", "0.7"]
        )
        assert result == {"kv_cache_config": {"free_gpu_memory_fraction": 0.7}}

    def test_deeply_nested(self):
        result = parse_dynamic_flags(["--trtllm.a.b.c.d", "hello"])
        assert result == {"a": {"b": {"c": {"d": "hello"}}}}

    def test_multiple_flags(self):
        result = parse_dynamic_flags(
            [
                "--trtllm.kv_cache_config.free_gpu_memory_fraction",
                "0.7",
                "--trtllm.kv_cache_config.enable_block_reuse",
                "false",
                "--trtllm.tensor_parallel_size",
                "4",
            ]
        )
        assert result == {
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.7,
                "enable_block_reuse": False,
            },
            "tensor_parallel_size": 4,
        }

    def test_bool_values(self):
        result = parse_dynamic_flags(
            ["--trtllm.some_flag", "true", "--trtllm.other_flag", "false"]
        )
        assert result == {"some_flag": True, "other_flag": False}

    def test_string_values(self):
        result = parse_dynamic_flags(["--trtllm.model", "/path/to/model"])
        assert result == {"model": "/path/to/model"}

    def test_unrecognized_arg_exits(self):
        with pytest.raises(SystemExit):
            parse_dynamic_flags(["--unknown-flag", "value"])

    def test_missing_value_exits(self):
        with pytest.raises(SystemExit):
            parse_dynamic_flags(["--trtllm.some_key"])

    def test_missing_value_next_is_flag_exits(self):
        with pytest.raises(SystemExit):
            parse_dynamic_flags(["--trtllm.key1", "--trtllm.key2", "val"])

    def test_empty_key_segment_exits(self):
        with pytest.raises(SystemExit):
            parse_dynamic_flags(["--trtllm..bad", "value"])

    def test_keys_preserved_as_is(self):
        """Keys are not transformed — hyphens, underscores, mixed case all pass through."""
        result = parse_dynamic_flags(["--trtllm.My-Key_name.SubKey", "42"])
        assert result == {"My-Key_name": {"SubKey": 42}}

    def test_conflicting_path_exits(self):
        """Scalar then nested path on same key should fail gracefully."""
        with pytest.raises(SystemExit):
            parse_dynamic_flags(["--trtllm.a", "1", "--trtllm.a.b", "2"])
