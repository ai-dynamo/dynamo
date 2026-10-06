# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import pytest

from dynamo.profiler.utils.ais_system import resolve_ais_system

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
]


class SampleGPUSKUType(str, Enum):
    GB200 = "gb200"
    GB200SXM = "gb200_sxm"
    H200SXM = "h200_sxm"


@pytest.mark.parametrize(
    ("gpu_sku", "expected"),
    [
        (SampleGPUSKUType.GB200, "gb200"),
        (SampleGPUSKUType.GB200SXM, "gb200"),
        ("GB200_SXM", "gb200"),
        (SampleGPUSKUType.H200SXM, "h200_sxm"),
    ],
)
def test_resolve_ais_system(gpu_sku: str | SampleGPUSKUType, expected: str) -> None:
    assert resolve_ais_system(gpu_sku) == expected
