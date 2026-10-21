# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

weight_updater = pytest.importorskip(
    "sglang.srt.managers.scheduler_components.weight_updater",
    reason="SGLang is required",
)

from sglang.srt.managers.io_struct import (  # noqa: E402
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
)
from sglang.srt.plugins.hook_registry import HookRegistry, HookType  # noqa: E402

from dynamo.sglang import memory_occupation  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

# SGLang's own scheduler-side resume, so the tests exercise the code that crashes.
_ORIGINAL = weight_updater.SchedulerWeightUpdaterManager.resume_memory_occupation


@pytest.fixture
def resume_hook(monkeypatch):
    hooks = {}

    def capture_hook(_registry, target, hook, hook_type=HookType.AFTER, **_kwargs):
        hooks[target] = (hook, hook_type)

    monkeypatch.setattr(HookRegistry, "register", classmethod(capture_hook))
    memory_occupation.register_memory_occupation_plugin()

    hook, hook_type = hooks[memory_occupation._RESUME_MEMORY_OCCUPATION_TARGET]
    assert hook_type is HookType.AROUND
    return hook


def _manager(offload_tags):
    # Only the attributes resume reaches for kv_cache and cuda_graph tags.
    return SimpleNamespace(
        offload_tags=set(offload_tags),
        memory_saver_adapter=Mock(),
        scheduler=None,
    )


def test_redundant_resume_is_a_noop(resume_hook):
    manager = _manager([])

    # Without the hook, SGLang's resume crashes the scheduler on this input.
    with pytest.raises(KeyError, match="kv_cache"):
        _ORIGINAL(manager, ResumeMemoryOccupationReqInput())

    result = resume_hook(_ORIGINAL, manager, ResumeMemoryOccupationReqInput())

    assert isinstance(result, ResumeMemoryOccupationReqOutput)
    manager.memory_saver_adapter.resume.assert_not_called()
    assert manager.offload_tags == set()


def test_resume_skips_tags_that_were_not_released(resume_hook):
    manager = _manager(["kv_cache"])

    resume_hook(_ORIGINAL, manager, ResumeMemoryOccupationReqInput())

    manager.memory_saver_adapter.resume.assert_called_once_with("kv_cache")
    assert manager.offload_tags == set()


def test_resume_of_released_tags_is_unchanged(resume_hook):
    manager = _manager(["kv_cache", "cuda_graph"])

    resume_hook(
        _ORIGINAL,
        manager,
        ResumeMemoryOccupationReqInput(tags=["kv_cache", "cuda_graph"]),
    )

    assert manager.memory_saver_adapter.resume.call_args_list == [
        call("cuda_graph"),
        call("kv_cache"),
    ]
    assert manager.offload_tags == set()
