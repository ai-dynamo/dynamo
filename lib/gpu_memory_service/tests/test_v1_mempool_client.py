# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch", reason="torch is required")

import gpu_memory_service.v1.client.mempool as mempool_module  # noqa: E402
from gpu_memory_service.v1.client.mempool import TorchMempoolMemoryClient  # noqa: E402

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.none,
]


def test_weight_and_kv_lifecycle(monkeypatch):
    client = object.__new__(TorchMempoolMemoryClient)
    client._device = 0
    client._state, client._weights_state = "RUNNING", "OPEN"
    client._weights_pool, client._kv_cache_pool = object(), object()
    client._active_domain = Mock()
    client._weights = Mock(mappings=("mapping",))
    client._kv_cache = Mock()
    monkeypatch.setattr(client, "_destroy_weights_pool", Mock())
    monkeypatch.setattr(client, "_raise_if_allocator_failed", Mock())
    regions = []

    @contextmanager
    def use_pool(domain, pool):
        client._active_domain.get.return_value = domain
        regions.append((domain, pool))
        yield

    monkeypatch.setattr(client, "_use_pool", use_pool)
    client._weights.create_mapping.return_value = 11
    client._kv_cache.create_mapping.return_value = 22
    copy_tensors = Mock()
    monkeypatch.setattr(
        mempool_module,
        "copy_non_parameter_tensors_to_default_allocator",
        copy_tensors,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    monkeypatch.setattr(mempool_module.gc, "collect", Mock())

    with client.weight_region():
        assert client._malloc(10, 0, 0) == 11
    models = (object(), object(), object())
    client.publish_weights(models)
    client.suspend()
    assert client._state == "SUSPENDED"
    client.resume()
    with client.kv_cache_region():
        assert client._malloc(20, 0, 0) == 22

    assert regions == [
        (mempool_module._WEIGHTS, client._weights_pool),
        (mempool_module._KV_CACHE, client._kv_cache_pool),
    ]
    client._weights.create_mapping.assert_called_once_with(10)
    client._kv_cache.create_mapping.assert_called_once_with(20)
    copy_tensors.assert_called_once_with(models, ("mapping",))
    client._weights.commit.assert_called_once_with()
    assert client._state == "RUNNING"
    assert client._weights_state == "PUBLISHED"
