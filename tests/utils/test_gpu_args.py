# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.utils.gpu_args import map_cuda_visible_devices

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    ("logical_indices", "inherited", "expected"),
    [
        ([0], None, "0"),
        ([1, 2], None, "1,2"),
        ([0], "1", "1"),
        ([0, 1], "4,5", "4,5"),
        ([1, 0], " GPU-a , GPU-b ", "GPU-b,GPU-a"),
        ([0, 1], "MIG-a,MIG-b", "MIG-a,MIG-b"),
    ],
)
def test_map_cuda_visible_devices(logical_indices, inherited, expected):
    assert map_cuda_visible_devices(logical_indices, inherited) == expected


@pytest.mark.parametrize(
    ("logical_indices", "inherited", "message"),
    [
        ([-1], None, "non-negative"),
        ([0], "", "does not expose any devices"),
        ([1], "7", "exposes only"),
    ],
)
def test_map_cuda_visible_devices_rejects_invalid_assignment(
    logical_indices, inherited, message
):
    with pytest.raises(ValueError, match=message):
        map_cuda_visible_devices(logical_indices, inherited)
