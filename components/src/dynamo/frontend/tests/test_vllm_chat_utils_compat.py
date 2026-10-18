# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat processing must import with either supported vLLM module layout."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

pytest.importorskip("vllm.config")

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    "module_name",
    ["dynamo.frontend.prepost", "dynamo.frontend.vllm_processor"],
)
def test_chat_processor_imports_with_installed_vllm(module_name):
    module = importlib.import_module(module_name)
    assert module.__name__ == module_name


def _load_compat_module():
    source = Path(__file__).parents[1] / "_vllm_chat_utils.py"
    spec = importlib.util.spec_from_file_location("chat_utils_layout_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("modern", [True, False])
def test_helpers_follow_available_layout(monkeypatch, modern):
    old = ModuleType("vllm.entrypoints.chat_utils")
    old.load_chat_template = Mock()
    old.make_tool_call_id = Mock()
    new = ModuleType("vllm.renderers.chat_utils")
    new.load_chat_template = Mock()
    new.make_tool_call_id = Mock()
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.chat_utils", old)
    monkeypatch.setitem(
        sys.modules, "vllm.renderers.chat_utils", new if modern else None
    )
    module = _load_compat_module()
    expected = new if modern else old
    assert module.load_chat_template is expected.load_chat_template
    assert module.make_tool_call_id is expected.make_tool_call_id


def test_layout_fallback_does_not_hide_missing_transitive_dependency(monkeypatch):
    original = __import__

    def missing_dependency(name, *args, **kwargs):
        if name == "vllm.renderers.chat_utils":
            raise ModuleNotFoundError("dependency missing", name="missing_dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", missing_dependency)
    with pytest.raises(ModuleNotFoundError) as exc:
        _load_compat_module()
    assert exc.value.name == "missing_dependency"
