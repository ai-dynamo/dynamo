# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only test for the GMS V1 SGLang TP release fence.

Scope is deliberately narrow. The mocked SGLang/vLLM engine tests were removed
in "refactor(sglang): inline GMS V1 hooks and reject non-LLM workers"; this
covers only the TP release barrier, which is a correctness invariant for TP>1
rather than an engine-wiring detail.

Without the fence, SGLang can let the response-producing rank acknowledge
release_memory_occupation while a peer rank still owns its KV-cache socket, so
snapshot publication reaches CRIU with a half-external stream and yields a
checkpoint that cannot be restored. That failure only manifests at TP>1.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

if not HAS_TORCH:
    pytest.skip("torch is required", allow_module_level=True)

pytest.importorskip("sglang", reason="SGLang is required")

import gpu_memory_service.v1.integrations.sglang.plugin as plugin  # noqa: E402

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.sglang,
    pytest.mark.core,
]


def test_release_fence_barriers_on_the_tp_group(monkeypatch):
    """Every rank must clear the TP barrier before release is acknowledged."""
    barrier = Mock()
    monkeypatch.setattr(plugin.torch.distributed, "barrier", barrier)

    manager = SimpleNamespace(tp_cpu_group=object())
    release_result = object()

    returned = plugin._after_release_memory_occupation(release_result, manager)

    # The hook must be transparent: it fences, it does not alter the result.
    assert returned is release_result
    barrier.assert_called_once_with(group=manager.tp_cpu_group)


def test_release_fence_is_registered_as_an_after_hook():
    """The fence must run AFTER release, otherwise it cannot fence anything."""
    from sglang.srt.plugins.hook_registry import HookType

    registered: dict[str, object] = {}

    class _Recorder:
        @classmethod
        def register(cls, target, hook, hook_type=HookType.AFTER, **_kwargs):
            registered[target] = hook_type

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(plugin, "HookRegistry", _Recorder)
        monkey.setenv("DYN_GMS_USE_V1", "true")
        plugin.register_gms_v1_plugin()
    finally:
        monkey.undo()

    assert (
        registered.get(plugin._RELEASE_MEMORY_OCCUPATION_TARGET) is HookType.AFTER
    ), "TP release fence must be an AFTER hook"
    # The LayerSplit DSA hook overrides the parent method, so both must register.
    assert plugin._CREATE_DSA_INDEX_BUFFERS_TARGET in registered
    assert plugin._CREATE_LAYER_SPLIT_DSA_INDEX_BUFFERS_TARGET in registered
