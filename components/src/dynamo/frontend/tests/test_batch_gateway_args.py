# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture(autouse=True)
def clear_batch_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DYN_BATCH_GATEWAY_URL",
        "DYN_INTERACTIVE",
        "DYN_KSERVE_GRPC_SERVER",
        "DYN_FRONTEND_ROUTE_EXTENSIONS",
        "DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD",
        "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD",
        "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC",
    ):
        monkeypatch.delenv(name, raising=False)


def parse_frontend_config(argv: list[str]) -> FrontendConfig:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args(argv))
    config.validate()
    return config


@pytest.mark.parametrize("argv", [[], ["-i"], ["--kserve-grpc-server"]])
def test_batch_proxy_disabled_without_configuration(argv: list[str]) -> None:
    assert parse_frontend_config(argv).batch_gateway_url is None


def test_batch_proxy_cli_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DYN_BATCH_GATEWAY_URL", "http://environment-batch")
    assert parse_frontend_config([]).batch_gateway_url == "http://environment-batch"
    config = parse_frontend_config(["--batch-gateway-url", "http://configured-batch"])
    assert config.batch_gateway_url == "http://configured-batch"
    assert config.__dict__["batch_gateway_url"] == "http://configured-batch"


@pytest.mark.parametrize("mode", ["-i", "--kserve-grpc-server"])
def test_batch_proxy_rejects_non_http_input(mode: str) -> None:
    with pytest.raises(ValueError, match="--batch-gateway-url.*HTTP frontend"):
        parse_frontend_config([mode, "--batch-gateway-url", "http://configured-batch"])
