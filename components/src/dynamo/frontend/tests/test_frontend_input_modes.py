# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
import sys
import types

import pytest

from dynamo.frontend.main import (
    CHAT_PROCESSOR_ENTRYPOINT_GROUP,
    parse_args,
    setup_custom_chat_engine_factory,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture(autouse=True)
def clear_input_mode_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DYN_INTERACTIVE",
        "DYN_KSERVE_GRPC_SERVER",
        "DYN_CHAT_PROCESSOR",
        "DYN_CHAT_PROCESSOR_CONFIG",
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

    config, _, _, _ = parse_args()

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

    config, _, sglang_flags, _ = parse_args()

    assert config.chat_processor == "sglang"
    assert sglang_flags.tool_call_parser == "qwen"


@pytest.fixture
def custom_chat_processor(monkeypatch: pytest.MonkeyPatch) -> dict:
    """A custom chat processor module whose provider records its call."""
    calls: dict = {}

    async def chat_engine_factory(instance_id, mdc, routed_engine):
        raise AssertionError("not called by these tests")

    def provider(config, options):
        calls["config"] = config
        calls["options"] = options
        return chat_engine_factory

    module = types.ModuleType("fake_chat_processor")
    module.provider = provider  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_chat_processor", module)
    entry_point = importlib.metadata.EntryPoint(
        name="fake",
        value="fake_chat_processor:provider",
        group=CHAT_PROCESSOR_ENTRYPOINT_GROUP,
    )
    entry_points = importlib.metadata.EntryPoints([entry_point])
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entry_points)
    calls["factory"] = chat_engine_factory
    return calls


@pytest.mark.parametrize("name", ["fake", "fake_chat_processor:provider"])
def test_custom_chat_processor_gets_its_config(
    monkeypatch: pytest.MonkeyPatch, custom_chat_processor: dict, name: str
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dynamo.frontend",
            "--dyn-chat-processor",
            name,
            "--dyn-chat-processor-config",
            '{"render_workers": 8}',
        ],
    )

    config, _, _, provider = parse_args()
    factory = setup_custom_chat_engine_factory(config, provider)

    assert factory is custom_chat_processor["factory"]
    assert custom_chat_processor["config"] is config
    assert custom_chat_processor["options"] == {"render_workers": 8}


@pytest.mark.parametrize("extra_args", [[], ["--enable-auto-tool-choice"]])
def test_unknown_chat_processor_is_named_at_parse_time(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    custom_chat_processor: dict,
    extra_args: list[str],
) -> None:
    # A mistyped name must be reported as such, also when the flags of the
    # processor the user meant come with it.
    monkeypatch.setattr(
        sys,
        "argv",
        ["dynamo.frontend", "--dyn-chat-processor", "missing", *extra_args],
    )

    with pytest.raises(SystemExit) as exc:
        parse_args()

    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "Unknown chat processor 'missing'" in error
    assert "Available: fake" in error


def test_chat_processor_config_needs_a_custom_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["dynamo.frontend", "--dyn-chat-processor-config", '{"render_workers": 8}'],
    )
    with pytest.raises(ValueError, match="only applies to a custom"):
        parse_args()
