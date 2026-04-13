# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.integrations.common.utils import get_gms_lock_mode
from gpu_memory_service.integrations.vllm.utils import configure_gms_lock_mode

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def test_get_gms_lock_mode_defaults_to_auto():
    assert get_gms_lock_mode({}) == RequestedLockType.AUTO


@pytest.mark.parametrize(
    ("raw_mode", "expected"),
    [
        ("rw", RequestedLockType.RW),
        ("RO", RequestedLockType.RO),
        (RequestedLockType.AUTO, RequestedLockType.AUTO),
    ],
)
def test_get_gms_lock_mode_parses_supported_values(raw_mode, expected):
    assert get_gms_lock_mode({"gms_lock_mode": raw_mode}) == expected


def test_get_gms_lock_mode_rejects_removed_gms_read_only_flag():
    with pytest.raises(ValueError, match="gms_read_only was removed"):
        get_gms_lock_mode({"gms_read_only": True})


def test_get_gms_lock_mode_rejects_invalid_value():
    with pytest.raises(ValueError, match="rw, ro, auto"):
        get_gms_lock_mode({"gms_lock_mode": "reader"})


@pytest.mark.parametrize(
    ("extra_config", "expected_mode"),
    [
        (None, "auto"),
        ({}, "auto"),
        ({"gms_lock_mode": "auto"}, "auto"),
        ({"gms_lock_mode": "rw"}, "rw"),
        ({"gms_lock_mode": "ro"}, "ro"),
        ('{"gms_lock_mode": "rw"}', "rw"),
        ('{"gms_lock_mode": "ro"}', "ro"),
    ],
)
def test_configure_gms_lock_mode_preserves_explicit_mode_and_defaults_to_auto(
    extra_config, expected_mode
):
    engine_args = SimpleNamespace(model_loader_extra_config=extra_config)

    configure_gms_lock_mode(engine_args)

    assert engine_args.model_loader_extra_config["gms_lock_mode"] == expected_mode


def test_configure_gms_lock_mode_rejects_removed_gms_read_only_flag():
    engine_args = SimpleNamespace(model_loader_extra_config={"gms_read_only": True})

    with pytest.raises(ValueError, match="gms_read_only was removed"):
        configure_gms_lock_mode(engine_args)
