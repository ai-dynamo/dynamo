# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("torch", reason="torch is required")

from gpu_memory_service.integrations.common.utils import (  # noqa: E402
    DEFAULT_GMS_RO_CONNECT_TIMEOUT_MS,
    get_gms_ro_connect_timeout_ms,
)


@pytest.mark.parametrize(
    ("extra_config", "expected"),
    [
        ({}, DEFAULT_GMS_RO_CONNECT_TIMEOUT_MS),
        ({"gms_ro_connect_timeout_ms": 120_000}, 120_000),
        ({"gms_ro_connect_timeout_ms": "120000"}, 120_000),
        ({"gms_ro_connect_timeout_ms": None}, None),
        ({"gms_ro_connect_timeout_ms": "null"}, None),
    ],
)
def test_get_gms_ro_connect_timeout_ms(extra_config, expected):
    assert get_gms_ro_connect_timeout_ms(extra_config) == expected


@pytest.mark.parametrize("value", [True, -1, "slow", object()])
def test_get_gms_ro_connect_timeout_ms_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="gms_ro_connect_timeout_ms"):
        get_gms_ro_connect_timeout_ms({"gms_ro_connect_timeout_ms": value})
