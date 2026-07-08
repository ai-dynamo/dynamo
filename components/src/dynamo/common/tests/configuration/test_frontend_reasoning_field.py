# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

_ENV_VAR = "DYN_CHAT_COMPLETIONS_REASONING_FIELD"


def _parse_frontend_config(argv: list[str]) -> FrontendConfig:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args(argv))
    config.validate()
    return config


def test_reasoning_field_defaults_to_reasoning_content(monkeypatch) -> None:
    monkeypatch.delenv(_ENV_VAR, raising=False)

    config = _parse_frontend_config([])

    assert config.chat_completions_reasoning_field == "reasoning_content"


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_reasoning_field_accepts_environment_values(monkeypatch, field: str) -> None:
    monkeypatch.setenv(_ENV_VAR, field)

    config = _parse_frontend_config([])

    assert config.chat_completions_reasoning_field == field


def test_reasoning_field_cli_overrides_environment(monkeypatch) -> None:
    monkeypatch.setenv(_ENV_VAR, "invalid")

    config = _parse_frontend_config(["--chat-completions-reasoning-field", "reasoning"])

    assert config.chat_completions_reasoning_field == "reasoning"


@pytest.mark.parametrize("field", ["", "Reasoning", " reasoning", "reasoning_content "])
def test_reasoning_field_rejects_invalid_environment_values(
    monkeypatch, field: str
) -> None:
    monkeypatch.setenv(_ENV_VAR, field)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args([]))

    with pytest.raises(ValueError, match="chat-completions-reasoning-field"):
        config.validate()


def test_reasoning_field_rejects_invalid_cli_value(monkeypatch) -> None:
    monkeypatch.delenv(_ENV_VAR, raising=False)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--chat-completions-reasoning-field", "invalid"])
