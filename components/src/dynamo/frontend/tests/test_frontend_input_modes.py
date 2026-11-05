# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.metadata
import sys
from types import ModuleType, SimpleNamespace

import pytest

from dynamo.frontend.main import parse_args

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture(autouse=True)
def clear_input_mode_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DYN_INTERACTIVE",
        "DYN_KSERVE_GRPC_SERVER",
        "DYN_CHAT_PROCESSOR",
        "DYN_FRONTEND_ROUTE_EXTENSIONS",
        "DYN_PREPROCESS_WORKERS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("mode", ["--interactive", "--kserve-grpc-server"])
@pytest.mark.parametrize("processor", ["vllm", "sglang"])
def test_reject_python_chat_processor_in_non_http_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
    processor: str,
) -> None:
    # The error must describe the unsupported mode even without backend packages.
    monkeypatch.setitem(sys.modules, processor, None)
    monkeypatch.setattr(
        sys, "argv", ["dynamo.frontend", mode, "--dyn-chat-processor", processor]
    )

    with pytest.raises(SystemExit) as exc:
        parse_args()

    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "--dyn-chat-processor" in error
    assert mode in error


@pytest.mark.parametrize("mode", ["--interactive", "--kserve-grpc-server"])
def test_native_chat_processor_keeps_non_http_modes(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setattr(sys, "argv", ["dynamo.frontend", mode])

    config, _, _ = parse_args()

    assert config.chat_processor == "dynamo"


def test_http_sglang_processor_keeps_parser_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dynamo.frontend",
            "--dyn-chat-processor",
            "sglang",
            "--tool-call-parser",
            "qwen",
        ],
    )

    config, _, sglang_flags = parse_args()

    assert config.chat_processor == "sglang"
    assert sglang_flags.tool_call_parser == "qwen"


@pytest.mark.parametrize("layout", ["common", "launchers", "openai"])
@pytest.mark.parametrize("version", ["0.1.dev24216", "0.32.0"])
def test_http_vllm_processor_detects_cli_layout_without_version_checks(
    monkeypatch: pytest.MonkeyPatch, layout: str, version: str
) -> None:
    # Model the modern development wheel and the older supported layout,
    # including version strings that disagree with the installed modules.
    modules = {}
    for name in (
        "vllm",
        "vllm.utils",
        "vllm.platforms",
        "vllm.engine",
        "vllm.engine.arg_utils",
        "vllm.entrypoints",
        "vllm.entrypoints.common",
        "vllm.entrypoints.launchers",
        "vllm.entrypoints.openai",
        f"vllm.entrypoints.{layout}.cli_args",
    ):
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent:
            monkeypatch.setattr(modules[parent], child, module, raising=False)

    class FrontendArgs:
        @staticmethod
        def add_cli_args(parser):
            parser.add_argument("--tool-call-parser")
            return parser

    class AsyncEngineArgs:
        @staticmethod
        def add_cli_args(parser):
            parser.add_argument("--max-model-len", type=int)
            return parser

    modules["vllm.utils"].FlexibleArgumentParser = argparse.ArgumentParser
    modules["vllm.platforms"].current_platform = SimpleNamespace(device_type="cpu")
    modules["vllm.engine.arg_utils"].AsyncEngineArgs = AsyncEngineArgs
    modules[f"vllm.entrypoints.{layout}.cli_args"].FrontendArgs = FrontendArgs
    for other_layout in {"common", "launchers", "openai"} - {layout}:
        monkeypatch.setitem(
            sys.modules, f"vllm.entrypoints.{other_layout}.cli_args", None
        )
    installed_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: version if name == "vllm" else installed_version(name),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dynamo.frontend",
            "--dyn-chat-processor",
            "vllm",
            "--max-model-len",
            "2048",
            "--tool-call-parser",
            "kimi_k3",
        ],
    )

    config, vllm_flags, _ = parse_args()

    assert config.chat_processor == "vllm"
    assert vllm_flags.max_model_len == 2048
    assert vllm_flags.tool_call_parser == "kimi_k3"
