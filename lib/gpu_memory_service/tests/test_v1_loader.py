# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GMS V1 loader options and the ModelExpress source identity."""

from __future__ import annotations

import importlib.util
from unittest.mock import patch

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from gpu_memory_service.snapshot.backends import mx
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


def test_mx_device_selects_peer_rank_and_listen_port():
    config = _backend_config(
        ["--transfer-backend", "mx", "--mx-source-key", "ckpt-1", "--device", "3"]
    )

    assert config["mx_rank"] == 3
    assert config["mx_source_key"] == "ckpt-1"
    assert config["mx_server_url"] is None
    assert config["mx_listen_port"] == 25558


def test_mx_needs_a_source_key():
    with pytest.raises(SystemExit):
        _backend_config(["--transfer-backend", "mx"])


@pytest.mark.skipif(
    importlib.util.find_spec("modelexpress") is None, reason="modelexpress required"
)
def test_source_key_is_the_whole_identity():
    identity = mx.source_identity("ckpt-1")

    assert identity.model_name == "ckpt-1"
    assert identity.tensor_parallel_size == 0
    assert dict(identity.extra_parameters) == {"memory": "gms-v1"}
    assert mx.source_identity("ckpt-1") == identity
    assert mx.source_identity("ckpt-2") != identity
