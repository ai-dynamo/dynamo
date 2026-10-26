# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GMS V1 loader options."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from gpu_memory_service.v1.snapshot import loader

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


# A pod's GPUs in PCI bus order, and in the order CUDA numbers them.
_BUS_ORDER = ["GPU-a", "GPU-b", "GPU-c", "GPU-d"]
_CUDA_ORDER = ["GPU-b", "GPU-a", "GPU-c", "GPU-d"]


def _load(argv: list[str]):
    """The loader's load_weights call on that pod."""
    identity = loader.device_identity
    with patch.object(loader, "init_vmm"), patch.object(
        loader, "load_weights"
    ) as load_weights, patch.object(
        identity, "gpu_uuid", _BUS_ORDER.__getitem__
    ), patch.object(
        identity, "cuda_ordinal", _CUDA_ORDER.index
    ):
        loader.main(["--checkpoint-dir", "/ckpt", *argv])
    return load_weights.call_args


def _backend_config(argv: list[str]) -> dict:
    return _load(argv).kwargs["backend_config"]


def test_device_n_loads_its_artifact_into_the_nth_gpu_in_bus_order(monkeypatch):
    monkeypatch.setenv("GMS_SOCKET_DIR", "/run/gms")

    assert _load(["--device", "1"]).args == (
        "/ckpt/device-1",
        "/run/gms/gms_GPU-b_weights.sock",
        0,
    )


def test_backend_config_carries_storage_backend_settings():
    assert _backend_config(
        [
            "--sharded-ssd-roots",
            "/ssd0,/ssd1",
            "--sharded-ssd-queues-per-root",
            "4",
            "--posix-backend-param",
            "ios_pool_size=64",
        ]
    ) == {
        "sharded_ssd_roots": ["/ssd0", "/ssd1"],
        "sharded_ssd_queues_per_root": 4,
        "posix_backend_params": {"ios_pool_size": "64"},
    }
