# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo._core import get_structural_tag_parser_names

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.timeout(30),
]


def test_structural_tag_parser_names_binding():
    names = get_structural_tag_parser_names()
    assert "deepseek_v41" in names
    assert "gemma4" not in names
