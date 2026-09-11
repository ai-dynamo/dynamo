# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from gpu_memory_service.integrations.sglang import install_vmm_ipc_kv as installer

pytest.importorskip("sglang")

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
]


@pytest.mark.parametrize(
    "args,kwargs",
    [((8, True), {}), ((8,), {"post_capture_active": True})],
)
def test_post_capture_pool_is_rejected_before_allocator_or_native_init(
    monkeypatch, args, kwargs
):
    native_called = False

    def native_init(_self, size, post_capture_active=False):
        nonlocal native_called
        native_called = True

    monkeypatch.setattr(
        installer,
        "_resolve_kv_pool_device",
        lambda *_args, **_kwargs: pytest.fail("allocator setup must not begin"),
    )

    with pytest.raises(RuntimeError, match="post-capture VMM pools"):
        installer._persistent_init(
            native_init, "MHATokenToKVPool", object(), args, kwargs
        )

    assert native_called is False


@pytest.mark.parametrize(
    "name",
    [
        "MHATokenToKVPoolFP4",
        "MHATokenToKVPoolMXFP8",
        "MLATokenToKVPoolFP4",
        "DSATokenToKVPool",
    ],
)
def test_install_rebinds_unsupported_pool_constructors(monkeypatch, name):
    from sglang.srt.mem_cache import kv_cache_configurator

    original = getattr(kv_cache_configurator, name)
    monkeypatch.setattr(kv_cache_configurator, name, original)
    monkeypatch.setattr(installer, "_INSTALLED", False)

    assert installer.install()
    wrapped = getattr(kv_cache_configurator, name)

    assert issubclass(wrapped, original)
    with pytest.raises(RuntimeError, match=name):
        wrapped()
