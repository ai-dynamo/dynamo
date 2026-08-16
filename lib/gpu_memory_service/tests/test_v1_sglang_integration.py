# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from unittest.mock import Mock

import pytest

pytest.importorskip("sglang", reason="SGLang is required")

import gpu_memory_service.v1.integrations.sglang.plugin as plugin  # noqa: E402
from sglang.srt.plugins.hook_registry import HookRegistry, HookType  # noqa: E402

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.sglang,
    pytest.mark.core,
]


def test_sglang_hooks_capture_models_and_delegate_memory_control(monkeypatch):
    hooks = {}

    def capture_hook(_registry, target, hook, hook_type=HookType.AFTER, **_kwargs):
        hooks[target] = hook

    monkeypatch.setattr(HookRegistry, "register", classmethod(capture_hook))
    client = Mock()
    client.weight_region.return_value = nullcontext()
    client.kv_cache_region.return_value = nullcontext()
    adapter = object.__new__(plugin.GMSV1MemorySaverAdapter)
    adapter._client = client
    adapter._models = []
    monkeypatch.setattr(plugin, "_adapter", lambda: adapter)
    monkeypatch.setenv("DYN_SGL_ENABLE_GMS_V1", "true")
    barrier = Mock()
    monkeypatch.setattr(plugin.torch.distributed, "barrier", barrier)

    plugin.register_gms_v1_plugin()

    assert hooks.keys() == {
        plugin._INITIAL_MODEL_LOAD_TARGET,
        plugin._FACTORY_TARGET,
        plugin._RELEASE_MEMORY_OCCUPATION_TARGET,
    }
    target, draft = object(), object()
    observe_model = hooks[plugin._INITIAL_MODEL_LOAD_TARGET]
    observe_model(Mock(model=target))
    observe_model(Mock(model=draft), is_draft_worker=True)

    original_factory = Mock()
    assert hooks[plugin._FACTORY_TARGET](original_factory, enable=True) is adapter
    release = hooks[plugin._RELEASE_MEMORY_OCCUPATION_TARGET]
    manager, release_result = Mock(tp_cpu_group=object()), object()
    assert release(release_result, manager, Mock()) is release_result
    barrier.assert_called_once_with(group=manager.tp_cpu_group)
    with (
        adapter.region("weights", enable_cpu_backup=True),
        adapter.region("kv_cache", enable_cpu_backup=False),
    ):
        pass
    adapter.pause("weights")
    adapter.resume("weights")

    client.weight_region.assert_called_once_with()
    client.kv_cache_region.assert_called_once_with()
    client.publish_weights.assert_called_once_with([target, draft])
    client.suspend.assert_called_once_with()
    client.resume.assert_called_once_with()
