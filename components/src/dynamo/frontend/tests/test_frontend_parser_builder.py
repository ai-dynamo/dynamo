# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``build_frontend_parser``: another frontend's parser, against an explicit env."""

import argparse
import logging
import os
import sys

import pytest

from dynamo.common.configuration.utils import env_or_default
from dynamo.frontend.frontend_args import FrontendConfig, build_frontend_parser

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_reads_only_the_given_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DYN_ROUTER_MODE", "kv")
    monkeypatch.setenv("DYN_ROUTER_TEMPERATURE", "0.9")
    before = dict(os.environ)

    parser, env_dests = build_frontend_parser({"DYN_ROUTER_MODE": "random"})
    namespace = parser.parse_args([])

    assert namespace.router_mode == "random"
    assert namespace.router_temperature == 0.0
    assert env_dests["DYN_ROUTER_MODE"] == "router_mode"
    assert env_dests["DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT"] == "overlap_score_credit"
    assert dict(os.environ) == before


def test_media_defaults_use_the_given_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DYN_REJECT_MEDIA", "0")
    monkeypatch.setenv("DYN_REJECT_MEDIA_STATUS_CODE", "500")

    parser, env_dests = build_frontend_parser(
        {"DYN_REJECT_MEDIA": "1", "DYN_REJECT_MEDIA_STATUS_CODE": "429"}
    )
    namespace = parser.parse_args([])

    assert namespace.reject_media is True
    assert namespace.reject_media_status_code == 429
    assert env_dests["DYN_REJECT_MEDIA"] == "reject_media"
    assert env_dests["DYN_REJECT_MEDIA_STATUS_CODE"] == "reject_media_status_code"


def test_no_registered_env_var_is_read_from_os_environ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An argument registered without ``env=`` would take its default from
    # the parsing process's os.environ. Record every env var the
    # registrations look up, wherever ``env_or_default`` was imported.
    registered: set[str] = set()

    def recording_env_or_default(env_var, default, value_type=None, *, env=None):
        registered.add(env_var)
        return env_or_default(env_var, default, value_type, env=env)

    for name, module in list(sys.modules.items()):
        if name.startswith("dynamo.") and (
            vars(module).get("env_or_default") is env_or_default
        ):
            monkeypatch.setattr(module, "env_or_default", recording_env_or_default)
    build_frontend_parser({})
    assert "DYN_ROUTER_MODE" in registered
    for name in registered:
        monkeypatch.delenv(name, raising=False)
    defaults = vars(build_frontend_parser({})[0].parse_args([]))

    # Two sentinels, so that a leaked env var changes at least one default.
    for sentinel in ("0", "1"):
        for name in registered:
            monkeypatch.setenv(name, sentinel)
        parser, _ = build_frontend_parser({})
        assert vars(parser.parse_args([])) == defaults, sentinel


def test_parses_without_notices_or_local_files(
    caplog: pytest.LogCaptureFixture, recwarn: pytest.WarningsRecorder
) -> None:
    env = {"DYN_ADMISSION_CONTROL": "none", "DYN_ENFORCE_DISAGG": "1"}
    parser = argparse.ArgumentParser(add_help=False)
    with caplog.at_level(logging.WARNING):
        assert build_frontend_parser(env, parser)[0] is parser
        namespace, unknown = parser.parse_known_args(
            [
                "--router-temperature",
                "0.5",
                "--admission-control",
                "none",
                "--enforce-disagg",
                "--model-path",
                "/not/a/local/dir",
                "--ais-perf-config",
                "/not/a/local/file.yaml",
            ]
        )

    assert namespace.router_temperature == 0.5
    assert namespace.model_path == "/not/a/local/dir"
    assert not hasattr(namespace, "admission_control")
    # The AISimulate perf-model flags name local files and are not registered.
    assert unknown == ["--ais-perf-config", "/not/a/local/file.yaml"]
    assert not caplog.records
    assert not recwarn


def _clear_plane_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the process's plane and TCP TLS settings out of a test."""
    for name in ("DYN_REQUEST_PLANE", "DYN_RESPONSE_PLANE"):
        # setenv records the original value even when the name is absent, so
        # the delenv below restores it instead of leaving a later export behind.
        monkeypatch.setenv(name, "unset")
        monkeypatch.delenv(name)
    for name in list(os.environ):
        if name.startswith("DYN_TCP_TLS_"):
            monkeypatch.delenv(name)


def _frontend_config(argv: list[str]) -> FrontendConfig:
    parser, _ = build_frontend_parser({})
    return FrontendConfig.from_cli_args(parser.parse_args(argv))


def test_request_plane_velo_requires_velo_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_plane_env(monkeypatch)

    with pytest.raises(ValueError, match="DYN_RESPONSE_PLANE"):
        _frontend_config(["--request-plane", "velo"]).validate()


def test_request_plane_velo_accepts_velo_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_plane_env(monkeypatch)

    config = _frontend_config(["--request-plane", "velo", "--response-plane", "velo"])
    config.validate()

    assert config.request_plane == "velo"
    assert config.response_plane == "velo"


def test_request_plane_velo_accepts_velo_response_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_plane_env(monkeypatch)
    monkeypatch.setenv("DYN_RESPONSE_PLANE", "velo")

    config = _frontend_config(["--request-plane", "velo"])
    config.validate()

    # The parser builder ignores os.environ, so validate() supplied the fallback.
    assert config.response_plane is None


def test_uppercase_request_plane_env_requires_velo_response() -> None:
    parser, _ = build_frontend_parser(
        {"DYN_REQUEST_PLANE": "VELO", "DYN_RESPONSE_PLANE": "tcp"}
    )
    config = FrontendConfig.from_cli_args(parser.parse_args([]))

    with pytest.raises(ValueError, match="DYN_RESPONSE_PLANE"):
        config.validate()


def test_plane_values_are_normalized() -> None:
    parser, _ = build_frontend_parser(
        {"DYN_REQUEST_PLANE": " VELO ", "DYN_RESPONSE_PLANE": " VELO "}
    )
    config = FrontendConfig.from_cli_args(parser.parse_args([]))
    config.validate()

    assert config.request_plane == "velo"
    assert config.response_plane == "velo"


@pytest.mark.parametrize("request_plane", ["tcp", "nats"])
def test_non_velo_request_plane_accepts_velo_response(
    monkeypatch: pytest.MonkeyPatch, request_plane: str
) -> None:
    _clear_plane_env(monkeypatch)

    config = _frontend_config(
        ["--request-plane", request_plane, "--response-plane", "velo"]
    )
    config.validate()

    assert config.request_plane == request_plane


def test_request_plane_help_documents_velo_requirement() -> None:
    parser, _ = build_frontend_parser({})

    help_text = " ".join(parser.format_help().split())
    assert "'tcp' is the default" in help_text
    assert "velo requires --response-plane velo" in help_text
