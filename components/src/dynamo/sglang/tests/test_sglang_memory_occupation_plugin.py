# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

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
    """Expose the hook without modifying SGLang's global registry."""
    hooks = {}

    def capture_hook(_registry, target, hook, hook_type=HookType.AFTER, **_kwargs):
        """Keep registrations local so each test receives a fresh hook."""
        hooks[target] = (hook, hook_type)

    monkeypatch.setattr(HookRegistry, "register", classmethod(capture_hook))
    memory_occupation.register_memory_occupation_plugin()

    hook, hook_type = hooks[memory_occupation._RESUME_MEMORY_OCCUPATION_TARGET]
    assert hook_type is HookType.AROUND
    return hook


def _manager(offload_tags):
    """Provide the scheduler state reached when resuming KV or graph memory."""
    return SimpleNamespace(
        offload_tags=set(offload_tags),
        memory_saver_adapter=Mock(),
        scheduler=None,
    )


def test_redundant_resume_is_a_noop(resume_hook):
    """An empty offload set must return success without calling the adapter."""
    manager = _manager([])

    result = resume_hook(_ORIGINAL, manager, ResumeMemoryOccupationReqInput())

    assert isinstance(result, ResumeMemoryOccupationReqOutput)
    manager.memory_saver_adapter.resume.assert_not_called()
    assert manager.offload_tags == set()


def test_resume_skips_tags_that_were_not_released(resume_hook):
    """The default all-tags request must resume only the released KV cache."""
    manager = _manager(["kv_cache"])

    resume_hook(_ORIGINAL, manager, ResumeMemoryOccupationReqInput())

    manager.memory_saver_adapter.resume.assert_called_once_with("kv_cache")
    assert manager.offload_tags == set()
