# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.fault_tolerance.cancellation.sglang_devices import (
    resolve_sglang_disaggregated_devices,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.fault_tolerance,
]


@pytest.mark.parametrize(
    "visible_devices,expected",
    [
        pytest.param(None, ("0", "1"), id="unset-default"),
        pytest.param("3,5", ("3", "5"), id="numeric-scheduler-assignment"),
        pytest.param(
            "GPU-decode,GPU-prefill,GPU-spare",
            ("GPU-decode", "GPU-prefill"),
            id="uuid-scheduler-assignment",
        ),
        pytest.param(" MIG-decode , MIG-prefill ", ("MIG-decode", "MIG-prefill")),
    ],
)
def test_resolve_sglang_disaggregated_devices(visible_devices, expected):
    assert resolve_sglang_disaggregated_devices(visible_devices) == expected


@pytest.mark.parametrize("visible_devices", ["", "7", " , "])
def test_resolve_sglang_disaggregated_devices_requires_two_entries(visible_devices):
    with pytest.raises(ValueError, match="requires at least two entries"):
        resolve_sglang_disaggregated_devices(visible_devices)
