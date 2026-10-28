# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


def parse_frontend_config(argv: list[str]) -> FrontendConfig:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args(argv))
    config.validate()
    return config


def test_isolate_tool_call_markers_is_off_by_default() -> None:
    assert parse_frontend_config([]).isolate_tool_call_markers is False


def test_isolate_tool_call_markers_with_sglang_processor() -> None:
    config = parse_frontend_config(
        ["--dyn-chat-processor", "sglang", "--isolate-tool-call-markers"]
    )
    assert config.isolate_tool_call_markers is True


def test_isolate_tool_call_markers_rejects_other_processors() -> None:
    with pytest.raises(ValueError, match="--dyn-chat-processor=sglang"):
        parse_frontend_config(
            ["--dyn-chat-processor", "dynamo", "--isolate-tool-call-markers"]
        )
