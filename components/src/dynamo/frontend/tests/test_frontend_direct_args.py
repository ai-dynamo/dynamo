# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture(autouse=True)
def clear_direct_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DYN_DIRECT",
        "DYN_DIRECT_BACKEND",
        "DYN_ENGINE_URLS",
        "DYN_INTERACTIVE",
        "DYN_KSERVE_GRPC_SERVER",
    ):
        monkeypatch.delenv(name, raising=False)


def parse_frontend_config(argv: list[str]) -> FrontendConfig:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args(argv))
    config.validate()
    return config


def test_direct_mode_collects_engine_urls() -> None:
    config = parse_frontend_config(
        [
            "--direct",
            "--engine-url",
            "http://prefill-0:30001",
            "--engine-url",
            "http://decode-0:30001",
        ]
    )
    assert config.direct
    assert config.direct_backend == "sglang"
    assert config.engine_urls == ["http://prefill-0:30001", "http://decode-0:30001"]


def test_engine_urls_on_the_command_line_replace_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DYN_DIRECT", "true")
    monkeypatch.setenv("DYN_ENGINE_URLS", "http://a:30001 http://b:30001")
    assert parse_frontend_config([]).engine_urls == [
        "http://a:30001",
        "http://b:30001",
    ]
    assert parse_frontend_config(
        ["--engine-url", "http://c:30001", "--engine-url", "http://d:30001"]
    ).engine_urls == ["http://c:30001", "http://d:30001"]


def test_direct_mode_needs_an_engine() -> None:
    with pytest.raises(ValueError, match="--direct requires at least one --engine-url"):
        parse_frontend_config(["--direct"])


@pytest.mark.parametrize("mode", ["-i", "--kserve-grpc-server"])
def test_direct_mode_is_http_only(mode: str) -> None:
    with pytest.raises(ValueError, match="--direct is only supported by the HTTP"):
        parse_frontend_config([mode, "--direct", "--engine-url", "http://a:30001"])


def test_engine_urls_require_direct_mode() -> None:
    with pytest.raises(ValueError, match="--engine-url requires --direct"):
        parse_frontend_config(["--engine-url", "http://a:30001"])
