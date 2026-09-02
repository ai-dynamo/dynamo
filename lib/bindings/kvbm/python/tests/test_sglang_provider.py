# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the owner-backed host-memory provider seam."""

from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.kvbm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_registered_provider_is_idempotent_and_exclusive(load_source):
    provider_module = load_source("test_kvbm_sglang_provider", "provider.py")
    first = SimpleNamespace(name="first")
    second = SimpleNamespace(name="second")

    provider_module.register_host_memory_provider(first)
    provider_module.register_host_memory_provider(first)

    assert provider_module.get_host_memory_provider() is first
    with pytest.raises(RuntimeError, match="different KVBM host-memory provider"):
        provider_module.register_host_memory_provider(second)


def test_provider_is_loaded_once_from_environment(
    monkeypatch, install_module, load_source
):
    provider_module = load_source("test_kvbm_sglang_provider_env", "provider.py")
    provider = SimpleNamespace(name="owner")
    calls = []

    def create_provider():
        calls.append("create")
        return provider

    install_module("test_owner_provider", create_provider=create_provider)
    monkeypatch.setenv(
        "DYN_KVBM_HOST_MEMORY_PROVIDER", "test_owner_provider:create_provider"
    )

    assert provider_module.get_host_memory_provider() is provider
    assert provider_module.get_host_memory_provider() is provider
    assert calls == ["create"]


@pytest.mark.parametrize("factory_path", [None, "module_without_separator"])
def test_provider_configuration_is_required(monkeypatch, load_source, factory_path):
    provider_module = load_source("test_kvbm_sglang_provider_missing", "provider.py")
    if factory_path is None:
        monkeypatch.delenv("DYN_KVBM_HOST_MEMORY_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("DYN_KVBM_HOST_MEMORY_PROVIDER", factory_path)

    with pytest.raises(RuntimeError, match="owner-backed host-memory provider"):
        provider_module.get_host_memory_provider()
