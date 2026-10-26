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


def _backend_config(argv: list[str]) -> dict:
    with patch.object(loader, "init_vmm"), patch.object(
        loader, "load_weights"
    ) as load_weights:
        loader.main(["--checkpoint-dir", "/ckpt", *argv])
    return load_weights.call_args.kwargs["backend_config"]


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
