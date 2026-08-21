# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ThunderAgent router CLI configuration."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from dynamo.thunderagent_router.__main__ import _publish_sglang_generate_capability
from dynamo.thunderagent_router.args import parse_args

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_sglang_generate_capability_is_opt_in() -> None:
    base = ["--endpoint", "dynamo.sglang.generate"]
    assert parse_args(base).publish_sglang_generate is False
    assert (
        parse_args([*base, "--publish-sglang-generate"]).publish_sglang_generate is True
    )
    assert (
        parse_args([*base, "--no-publish-sglang-generate"]).publish_sglang_generate
        is False
    )


def test_sglang_generate_capability_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("DYN_THUNDERAGENT_PUBLISH_SGLANG_GENERATE", "true")
    assert (
        parse_args(["--endpoint", "dynamo.sglang.generate"]).publish_sglang_generate
        is True
    )


def test_publish_sglang_generate_capability_uses_backend_contract(monkeypatch) -> None:
    module = ModuleType("dynamo.sglang.engine_generate")
    module.SGLANG_GENERATE_CAPABILITY = "sglang_generate"
    monkeypatch.setitem(sys.modules, module.__name__, module)

    published: list[tuple[str, str]] = []

    class RuntimeConfig:
        def set_engine_specific(self, key: str, value: str) -> None:
            published.append((key, value))

    runtime_config = RuntimeConfig()

    _publish_sglang_generate_capability(runtime_config)  # type: ignore[arg-type]
    assert published == [("sglang_generate", "true")]
