# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.llm import lora_name_to_id

max_int32 = 0x7FFFFFFF


class TestLoraNameToId:
    @pytest.mark.timeout(5)
    def test_import_function(self):
        assert callable(lora_name_to_id)

    @pytest.mark.timeout(5)
    def test_returns_positive_integer_for_different_names(self):
        for i in range(100):
            result = lora_name_to_id(f"test_lora_{i}")
            assert isinstance(result, int)
            assert 1 <= result <= max_int32

    @pytest.mark.timeout(5)
    def test_different_names_produce_different_ids(self):
        id1 = lora_name_to_id("lora_adapter_1")
        id2 = lora_name_to_id("lora_adapter_2")
        assert id1 != id2

    @pytest.mark.timeout(5)
    def test_consistency_across_multiple_calls(self):
        test_names = [f"lora_{i}" for i in range(100)]
        results_first = [lora_name_to_id(name) for name in test_names]
        results_second = [lora_name_to_id(name) for name in test_names]
        assert results_first == results_second
