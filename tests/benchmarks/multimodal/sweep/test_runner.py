# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from benchmarks.multimodal.sweep.runner import _build_aiperf_cmd

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def _command(uuid_and_strip: bool) -> list[str]:
    return _build_aiperf_cmd(
        model="test-model",
        port=12345,
        sweep_mode="concurrency",
        sweep_value=4,
        conversation_num=2,
        warmup_count=1,
        input_file="input.jsonl",
        osl=10,
        artifact_dir=Path("artifacts"),
        uuid_and_strip=uuid_and_strip,
    )


def test_uuid_and_strip_is_opt_in() -> None:
    command = _command(uuid_and_strip=False)

    assert "--uuid-and-strip" not in command
    assert "--endpoint-type" not in command


def test_uuid_and_strip_uses_chat_endpoint() -> None:
    command = _command(uuid_and_strip=True)

    assert command[-3:] == ["--endpoint-type", "chat", "--uuid-and-strip"]
