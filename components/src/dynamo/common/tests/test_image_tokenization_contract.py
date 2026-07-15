# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.common.image_tokenization import ImageTokenizationSpec, type_identity

pytestmark = [
    pytest.mark.unit,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_image_tokenization_specs_have_stable_wire_values():
    assert [spec.value for spec in ImageTokenizationSpec] == [
        "qwen2_vl_v1",
        "qwen3_vl_v1",
        "moonvit_v1",
    ]


def test_type_identity_uses_exact_concrete_type():
    class Parent:
        pass

    class Child(Parent):
        pass

    assert type_identity(Child()).endswith(".Child")
    assert not type_identity(Child()).endswith(".Parent")
