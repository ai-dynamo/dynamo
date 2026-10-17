# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def parse_frontend_config(args: list[str]) -> FrontendConfig:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args(args))
    config.validate()
    return config


def test_forward_routes_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DYN_HTTP_FORWARD_ROUTES", "/a=http://h:1 /b=http://h:2")
    assert parse_frontend_config([]).forward_routes == [
        "/a=http://h:1",
        "/b=http://h:2",
    ]


def test_forward_route_cli_replaces_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DYN_HTTP_FORWARD_ROUTES", "/a=http://stale:1")
    config = parse_frontend_config(
        ["--forward-route", "/a=http://h:1", "--forward-route", "/c=http://h:3"]
    )
    assert config.forward_routes == ["/a=http://h:1", "/c=http://h:3"]


@pytest.mark.parametrize("mode_flag", ["--interactive", "--kserve-grpc-server"])
def test_forward_route_rejected_outside_http(
    monkeypatch: pytest.MonkeyPatch, mode_flag: str
) -> None:
    monkeypatch.delenv("DYN_HTTP_FORWARD_ROUTES", raising=False)
    with pytest.raises(ValueError, match=f"--forward-route.*{mode_flag}"):
        parse_frontend_config(["--forward-route", "/a=http://h:1", mode_flag])
