# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from unittest.mock import Mock, call

import pytest

# Guard on the plugin itself, not on "sglang": the runtime image ships a
# top-level sglang package without sglang.srt, so importorskip("sglang")
# passes there and the plugin's own sglang.srt imports then fail collection.
plugin = pytest.importorskip(
    "gpu_memory_service.v1.integrations.sglang.plugin",
    reason="SGLang is required",
)

from sglang.srt.plugins.hook_registry import HookRegistry, HookType  # noqa: E402

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.sglang,
    pytest.mark.core,
]

# The plugin inlines its hook targets as string literals, so name them here.
_FACTORY_TARGET = (
    "sglang.srt.utils.torch_memory_saver_adapter.TorchMemorySaverAdapter.create"
)
_INITIAL_MODEL_LOAD_TARGET = (
    "sglang.srt.model_executor.model_runner_components.load_model_utils."
    "load_model_with_memory_saver"
)
_INIT_ALL_CUDA_GRAPHS_TARGET = (
    "sglang.srt.managers.scheduler.Scheduler.init_all_cuda_graphs"
)
_RELEASE_MEMORY_OCCUPATION_TARGET = (
    "sglang.srt.managers.scheduler_components.weight_updater."
    "SchedulerWeightUpdaterManager.release_memory_occupation"
)
_CREATE_DSA_INDEX_KEY_CACHE_TARGET = (
    "sglang.srt.mem_cache.memory_pool.DSATokenToKVPool._create_index_key_cache"
)
_LAYER_SPLIT_DSA_INDEX_KEY_CACHE_TARGET = (
    "sglang.srt.mem_cache.dsa_cache_layer_split."
    "LayerSplitDSATokenToKVPool._create_index_key_cache"
)


def test_sglang_hooks_capture_models_and_delegate_memory_control(monkeypatch):
    hooks = {}
    hook_types = {}

    def capture_hook(_registry, target, hook, hook_type=HookType.AFTER, **_kwargs):
        hooks[target] = hook
        hook_types[target] = hook_type

    monkeypatch.setattr(HookRegistry, "register", classmethod(capture_hook))
    client = Mock()
    client.weight_region.return_value = nullcontext()
    client.kv_cache_region.return_value = nullcontext()
    adapter = object.__new__(plugin.GMSV1MemorySaverAdapter)
    adapter._client = client
    adapter._models = []
    monkeypatch.setattr(plugin, "_adapter", lambda: adapter)
    barrier = Mock()
    monkeypatch.setattr(plugin.torch.distributed, "barrier", barrier)
    monkeypatch.setenv("DYN_GMS_USE_V1", "true")

    plugin.register_gms_v1_plugin()

    assert hooks.keys() == {
        _INITIAL_MODEL_LOAD_TARGET,
        _INIT_ALL_CUDA_GRAPHS_TARGET,
        _FACTORY_TARGET,
        _CREATE_DSA_INDEX_KEY_CACHE_TARGET,
        _LAYER_SPLIT_DSA_INDEX_KEY_CACHE_TARGET,
        _RELEASE_MEMORY_OCCUPATION_TARGET,
    }
    assert hook_types[_INIT_ALL_CUDA_GRAPHS_TARGET] is HookType.BEFORE
    target, draft = object(), object()
    observe_model = hooks[_INITIAL_MODEL_LOAD_TARGET]
    observe_model(Mock(model=target))
    observe_model(Mock(model=draft), is_draft_worker=True)
    hooks[_INIT_ALL_CUDA_GRAPHS_TARGET](Mock())
    client.publish_weights.assert_called_once_with([target, draft])

    original_factory = Mock()
    assert hooks[_FACTORY_TARGET](original_factory, enable=True) is adapter
    with (
        adapter.region("weights", enable_cpu_backup=True),
        adapter.region("kv_cache", enable_cpu_backup=False),
    ):
        pass
    adapter.pause("weights")
    adapter.resume("weights")

    client.weight_region.assert_called_once_with()
    client.kv_cache_region.assert_called_once_with()
    assert client.publish_weights.call_args_list == [
        call([target, draft]),
        call([target, draft]),
    ]
    client.suspend.assert_called_once_with()
    client.resume.assert_called_once_with()

    release = hooks[_RELEASE_MEMORY_OCCUPATION_TARGET]
    manager, release_result = Mock(tp_cpu_group=object()), object()
    assert release(release_result, manager, Mock()) is release_result
    barrier.assert_called_once_with(group=manager.tp_cpu_group)
