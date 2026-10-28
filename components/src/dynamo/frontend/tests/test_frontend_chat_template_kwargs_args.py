# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig
from dynamo.frontend.main import _fold_vllm_default_chat_template_kwargs

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def parse_frontend_config(args: list[str]) -> FrontendConfig:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args(args))
    config.validate()
    return config


def test_default_chat_template_kwargs_parse_a_json_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DYN_DEFAULT_CHAT_TEMPLATE_KWARGS", raising=False)
    assert parse_frontend_config([]).default_chat_template_kwargs is None

    config = parse_frontend_config(
        ["--default-chat-template-kwargs", '{"clear_thinking": false}']
    )
    assert config.default_chat_template_kwargs == {"clear_thinking": False}

    monkeypatch.setenv(
        "DYN_DEFAULT_CHAT_TEMPLATE_KWARGS", '{"preserve_thinking": true}'
    )
    assert parse_frontend_config([]).default_chat_template_kwargs == {
        "preserve_thinking": True
    }


def test_default_chat_template_kwargs_reject_a_non_object() -> None:
    with pytest.raises(SystemExit):
        parse_frontend_config(["--default-chat-template-kwargs", '["a"]'])


@pytest.mark.parametrize("processor", ["dynamo", "sglang", "vllm"])
def test_default_chat_template_kwargs_accept_thinking_keys_on_every_processor(
    processor: str,
) -> None:
    kwargs = '{"enable_thinking": false, "reasoning_effort": "high"}'
    config = parse_frontend_config(
        ["--dyn-chat-processor", processor, "--default-chat-template-kwargs", kwargs]
    )
    assert config.default_chat_template_kwargs == {
        "enable_thinking": False,
        "reasoning_effort": "high",
    }


@pytest.mark.parametrize("processor", ["dynamo", "sglang", "vllm"])
def test_default_chat_template_kwargs_reject_chat_template(processor: str) -> None:
    with pytest.raises(ValueError, match="cannot set chat_template"):
        parse_frontend_config(
            [
                "--dyn-chat-processor",
                processor,
                "--default-chat-template-kwargs",
                '{"chat_template": "x"}',
            ]
        )


def test_vllm_spellings_fold_into_the_frontend_flag() -> None:
    config = parse_frontend_config(
        ["--default-chat-template-kwargs", '{"enable_thinking": false}']
    )
    # vLLM's parser read --default_chat_template_kwargs or the dotted form.
    vllm_flags = argparse.Namespace(
        default_chat_template_kwargs={"enable_thinking": True, "clear_thinking": False}
    )
    _fold_vllm_default_chat_template_kwargs(config, vllm_flags)
    assert config.default_chat_template_kwargs == {
        "enable_thinking": False,
        "clear_thinking": False,
    }

    with pytest.raises(ValueError, match="cannot set chat_template"):
        _fold_vllm_default_chat_template_kwargs(
            config,
            argparse.Namespace(default_chat_template_kwargs={"chat_template": "x"}),
        )
