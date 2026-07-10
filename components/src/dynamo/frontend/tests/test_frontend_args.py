# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _parse_frontend_config(argv: list[str]) -> FrontendConfig:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    return FrontendConfig.from_cli_args(parser.parse_args(argv))


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], False),
        (["--enable-engine-apis"], True),
        (["--no-enable-engine-apis"], False),
    ],
)
def test_enable_engine_apis_cli(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], expected: bool
) -> None:
    monkeypatch.delenv("DYN_VLLM_ENABLE_INFERENCE_V1_GENERATE", raising=False)

    config = _parse_frontend_config(argv)

    assert config.enable_engine_apis is expected


def test_enable_engine_apis_env_and_cli_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DYN_VLLM_ENABLE_INFERENCE_V1_GENERATE", "1")

    assert _parse_frontend_config([]).enable_engine_apis is True
    assert (
        _parse_frontend_config(["--no-enable-engine-apis"]).enable_engine_apis is False
    )
