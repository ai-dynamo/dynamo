# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

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
