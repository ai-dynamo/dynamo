# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture(autouse=True)
def clear_reject_media_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DYN_REJECT_MEDIA", raising=False)
    monkeypatch.delenv("DYN_REJECT_MEDIA_STATUS_CODE", raising=False)


def parse_frontend_config(args: list[str]) -> FrontendConfig:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args(args))
    config.validate()
    return config


def test_reject_media_is_disabled_by_default() -> None:
    config = parse_frontend_config([])

    assert config.reject_media is False
    assert config.reject_media_status_code == 415


def test_reject_media_cli_configuration() -> None:
    config = parse_frontend_config(
        ["--reject-media", "--reject-media-status-code", "429"]
    )

    assert config.reject_media is True
    assert config.reject_media_status_code == 429


def test_reject_media_environment_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DYN_REJECT_MEDIA", "true")
    monkeypatch.setenv("DYN_REJECT_MEDIA_STATUS_CODE", "503")

    config = parse_frontend_config([])

    assert config.reject_media is True
    assert config.reject_media_status_code == 503


@pytest.mark.parametrize("status", ["200", "302", "600"])
def test_reject_media_status_code_must_be_an_error_status(status: str) -> None:
    with pytest.raises(ValueError, match="--reject-media-status-code"):
        parse_frontend_config(["--reject-media-status-code", status])
