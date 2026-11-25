# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``build_mocker_parser``: another mocker's parser, against an explicit env."""

import argparse
import os

import pytest

from dynamo.mocker.args import (
    build_mocker_parser,
    parse_args,
    set_ais_perf_model_attributes,
)
from dynamo.mocker.config import old_ais_perf_inputs
from tests.utils.ais_perf_model import AIS_PERF_MODEL_ENV

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.unit,
    pytest.mark.usefixtures("no_ais_environment"),
]


def options(parser):
    """Return ``{option string: dest}`` of every option ``parser`` knows."""
    # argparse has no public list of a parser's options.
    return {
        option: action.dest for option, action in parser._option_string_actions.items()
    }


def test_registers_the_options_parse_args_registers(monkeypatch):
    parsers = []
    parse = argparse.ArgumentParser.parse_args

    def record(parser, *args, **kwargs):
        parsers.append(parser)
        return parse(parser, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(argparse.ArgumentParser, "parse_args", record)
        parse_args([])
    (process_parser,) = parsers

    expected = options(process_parser)
    for option in ("-h", "--help", "--version"):
        del expected[option]
    assert options(build_mocker_parser({})) == expected
    assert "--ais-perf-model" in expected
    assert "--router-mode" in expected


def test_reads_only_the_given_env(monkeypatch):
    # What this process has set: every variable the mocker reads itself, and
    # one of those the router group reads.
    ours = {
        "DYN_DISCOVERY_BACKEND": "file",
        "DYN_REQUEST_PLANE": "nats",
        "DYN_RESPONSE_PLANE": "quic",
        "DYN_EVENT_PLANE": "zmq",
        "DYN_ROUTER_MODE": "kv",
    }
    for name, value in ours.items():
        monkeypatch.setenv(name, value)
    before = dict(os.environ)

    namespace = build_mocker_parser({"DYN_DISCOVERY_BACKEND": "mem"}).parse_args([])

    assert namespace.discovery_backend == "mem"
    assert namespace.request_plane == "tcp"
    assert namespace.response_plane is None
    assert namespace.event_plane is None
    assert namespace.router_mode is None
    assert dict(os.environ) == before


def test_reads_the_argument_list_of_another_mocker(monkeypatch):
    # The variable of the process that reads is not the other mocker's.
    monkeypatch.setenv(AIS_PERF_MODEL_ENV, "ours.yaml")
    parser = build_mocker_parser({})

    namespace, rest = parser.parse_known_args(
        ["--ais-sys", "h100_sxm", "--aic-perf-model", "--not-a-mocker-flag", "x"]
    )
    # The other mocker's variable is empty, which counts as unset.
    set_ais_perf_model_attributes(parser, namespace, "")

    assert rest == ["--not-a-mocker-flag", "x"]
    assert namespace.ais_perf_model_without_value is True
    assert namespace.ais_perf_model_value is None
    assert namespace.ais_perf_model_source is None
    assert old_ais_perf_inputs(namespace) == [
        "--ais-perf-model without a value",
        "--ais-system",
    ]


def test_conflicting_uses_are_an_error_of_the_given_parser():
    class Raising(argparse.ArgumentParser):
        def error(self, message):
            raise ValueError(message)

    parser = build_mocker_parser({}, Raising(add_help=False))

    namespace = parser.parse_args(["--ais-perf-model"])
    with pytest.raises(ValueError, match="without a value cannot be combined with"):
        set_ais_perf_model_attributes(parser, namespace, "theirs.yaml")


def _clear_plane_env(monkeypatch):
    """Keep the process's plane and TCP TLS settings out of a test."""
    for name in ("DYN_REQUEST_PLANE", "DYN_RESPONSE_PLANE"):
        # setenv records the original value even when the name is absent, so
        # the delenv below restores it instead of leaving a later export behind.
        monkeypatch.setenv(name, "unset")
        monkeypatch.delenv(name)
    for name in list(os.environ):
        if name.startswith("DYN_TCP_TLS_"):
            monkeypatch.delenv(name)


def test_request_plane_velo_requires_velo_response(monkeypatch, capsys):
    _clear_plane_env(monkeypatch)

    with pytest.raises(SystemExit) as error:
        parse_args(["--request-plane", "velo"])

    assert error.value.code == 2
    assert "DYN_RESPONSE_PLANE" in capsys.readouterr().err


def test_request_plane_velo_accepts_velo_response(monkeypatch):
    _clear_plane_env(monkeypatch)

    args = parse_args(["--request-plane", "velo", "--response-plane", "velo"])

    assert args.request_plane == "velo"
    assert args.response_plane == "velo"


def test_request_plane_velo_accepts_velo_response_from_env(monkeypatch):
    _clear_plane_env(monkeypatch)
    monkeypatch.setenv("DYN_RESPONSE_PLANE", "velo")

    args = parse_args(["--request-plane", "velo"])

    assert args.response_plane == "velo"


@pytest.mark.parametrize("request_plane", ["tcp", "nats"])
def test_non_velo_request_plane_accepts_velo_response(monkeypatch, request_plane):
    _clear_plane_env(monkeypatch)

    args = parse_args(["--request-plane", request_plane, "--response-plane", "velo"])

    assert args.request_plane == request_plane


def test_uppercase_request_plane_env_requires_velo_response(monkeypatch, capsys):
    _clear_plane_env(monkeypatch)
    monkeypatch.setenv("DYN_REQUEST_PLANE", "VELO")

    with pytest.raises(SystemExit) as error:
        parse_args(["--response-plane", "tcp"])

    assert error.value.code == 2
    assert "DYN_RESPONSE_PLANE" in capsys.readouterr().err


def test_plane_values_are_normalized(monkeypatch):
    _clear_plane_env(monkeypatch)
    monkeypatch.setenv("DYN_REQUEST_PLANE", " VELO ")
    monkeypatch.setenv("DYN_RESPONSE_PLANE", " VELO ")

    args = parse_args([])

    assert args.request_plane == "velo"
    assert args.response_plane == "velo"


def test_request_plane_help_documents_velo_requirement():
    help_text = " ".join(build_mocker_parser({}).format_help().split())

    assert "'tcp' is the default" in help_text
    assert "velo requires --response-plane velo" in help_text
