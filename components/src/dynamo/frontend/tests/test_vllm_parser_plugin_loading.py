#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM processor component - loading plugins.

Tests for parser plugin loading during vLLM frontend initialization.
Note that this file is separated from test_vllm_processor_unit.py
because test_vllm_processor_unit.py assumes Qwen3 parser and
will skip tests for any other parsers.
"""

import importlib
from argparse import Namespace
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import pytest

# Needs vllm packages, but never touches a GPU.
pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
]


@pytest.fixture
def processor_module(monkeypatch):
    """Import lazily so pytest collection does not import the processor."""
    module = importlib.import_module("dynamo.frontend.vllm_processor")
    # To avoid unnecessary errors in test target `EngineFactory.__init__()`,
    # clear an environment variable, which is used inside of `__init__()`.
    monkeypatch.delenv("DYN_VLLM_STREAM_INTERVAL", raising=False)
    return module


@pytest.mark.parametrize(
    "plugin_flags, expected_paths",
    [
        pytest.param({}, [], id="attributes-absent"),
        pytest.param(
            {
                "tool_parser_plugin": None,
                "reasoning_parser_plugin": None,
            },
            [],
            id="all-none",
        ),
        pytest.param(
            {
                "tool_parser_plugin": "",
                "reasoning_parser_plugin": "",
            },
            [],
            id="all-empty",
        ),
        pytest.param(
            {"tool_parser_plugin": "./tool_parser.py"},
            ["./tool_parser.py"],
            id="tool-only",
        ),
        pytest.param(
            {"reasoning_parser_plugin": "/workspace/custom_thinking_parser.py"},
            ["/workspace/custom_thinking_parser.py"],
            id="reasoning-only",
        ),
        pytest.param(
            {
                "tool_parser_plugin": "./tool_parser.py",
                "reasoning_parser_plugin": "/workspace/custom_thinking_parser.py",
            },
            ["./tool_parser.py", "/workspace/custom_thinking_parser.py"],
            id="both",
        ),
    ],
)
def test_engine_factory_loads_configured_plugins(
    processor_module,
    monkeypatch,
    plugin_flags,
    expected_paths,
):
    """
    Test EngineFactory can properly handle multiple patterns of combination
    of each plugin option and given value.
    """
    # Return a real module object without executing plugin code or touching
    # vLLM's global parser registries.
    importer = Mock(side_effect=lambda _: ModuleType("test_parser_plugin"))
    monkeypatch.setattr(processor_module, "import_plugin", importer)

    factory = processor_module.EngineFactory(
        config=SimpleNamespace(),
        flags=Namespace(**plugin_flags),
    )

    assert factory is not None
    assert importer.call_count == len(expected_paths)
    importer.assert_has_calls(
        [call(path) for path in expected_paths],
        any_order=True,
    )


@pytest.mark.parametrize(
    "flag_name",
    ["tool_parser_plugin", "reasoning_parser_plugin"],
    ids=["tool", "reasoning"],
)
@pytest.mark.parametrize(
    "failure",
    [
        # vLLM can log an import error and return None without raising.
        None,
        # Some errors can propagate directly from import_plugin().
        ImportError("plugin dependency is missing"),
        SyntaxError("plugin syntax is wrong"),
        RuntimeError("unknown error happened"),
    ],
)
def test_engine_factory_rejects_plugin_import_failure(
    processor_module,
    monkeypatch,
    flag_name,
    failure,
):
    """
    Test EngineFactory can properly raise RuntimeError
    when given parser plugin file has any issue.
    """
    plugin_path = "./broken_parser.py"
    if failure is None:
        importer = Mock(return_value=failure)
    else:
        importer = Mock(side_effect=failure)

    monkeypatch.setattr(processor_module, "import_plugin", importer)

    with pytest.raises(RuntimeError) as exc_info:
        processor_module.EngineFactory(
            config=SimpleNamespace(),
            flags=Namespace(**{flag_name: plugin_path}),
        )

    importer.assert_called_once_with(plugin_path)
    assert plugin_path in str(exc_info.value)

    if failure is not None:
        assert exc_info.value.__cause__ is failure
