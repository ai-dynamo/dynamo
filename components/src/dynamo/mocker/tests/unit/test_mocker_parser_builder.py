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

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.unit,
]

ENV_VAR = "DYN_AIS_PERF_MODEL"


@pytest.fixture(autouse=True)
def no_env_value(monkeypatch):
    """Keep a value set in the shell that runs the tests out of them."""
    monkeypatch.delenv(ENV_VAR, raising=False)


class RecordingEnviron(dict):
    """A copy of the environment that records every name looked up."""

    def __init__(self, environ):
        super().__init__(environ)
        self.names = set()

    def get(self, name, default=None):
        self.names.add(name)
        return super().get(name, default)

    def __getitem__(self, name):
        self.names.add(name)
        return super().__getitem__(name)

    def __contains__(self, name):
        self.names.add(name)
        return super().__contains__(name)


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


def test_registers_on_the_given_parser():
    parser = argparse.ArgumentParser(add_help=False)
    assert build_mocker_parser({}, parser) is parser
    assert parser.parse_args(["--model-path", "m"]).model_path == "m"


def test_reads_only_the_given_env(monkeypatch):
    monkeypatch.setenv("DYN_DISCOVERY_BACKEND", "file")
    monkeypatch.setenv("DYN_ROUTER_MODE", "kv")
    before = dict(os.environ)

    env = {"DYN_DISCOVERY_BACKEND": "mem", "DYN_REQUEST_PLANE": "nats"}
    namespace = build_mocker_parser(env).parse_args([])

    assert namespace.discovery_backend == "mem"
    assert namespace.request_plane == "nats"
    assert namespace.router_mode is None
    assert dict(os.environ) == before


def test_no_env_default_is_read_from_os_environ(monkeypatch):
    # Every variable parse_args looks up for this process.
    environ = RecordingEnviron(os.environ)
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", environ)
        parse_args([])
    names = {name for name in environ.names if name.startswith("DYN_")}
    assert {"DYN_DISCOVERY_BACKEND", "DYN_EVENT_PLANE", "DYN_ROUTER_MODE"} <= names

    for name in names:
        monkeypatch.delenv(name, raising=False)
    defaults = vars(build_mocker_parser({}).parse_args([]))

    # Two values, so that a variable that is read changes at least one default.
    for value in ("0", "1"):
        for name in names:
            monkeypatch.setenv(name, value)
        assert vars(build_mocker_parser({}).parse_args([])) == defaults, value


def test_reads_the_argument_list_of_another_mocker(monkeypatch):
    # The variable of the process that reads is not the other mocker's.
    monkeypatch.setenv(ENV_VAR, "ours.yaml")
    parser = build_mocker_parser({})

    namespace, rest = parser.parse_known_args(
        ["--ais-sys", "h100_sxm", "--aic-perf-model", "--not-a-mocker-flag", "x"]
    )
    set_ais_perf_model_attributes(parser, namespace, None)

    assert rest == ["--not-a-mocker-flag", "x"]
    assert namespace.ais_perf_model_without_value is True
    assert namespace.ais_perf_model_value is None
    assert namespace.ais_perf_model_source is None
    assert old_ais_perf_inputs(namespace) == [
        "--ais-perf-model without a value",
        "--ais-system",
    ]

    namespace = parser.parse_args(["--ais-nextn", "2"])
    set_ais_perf_model_attributes(parser, namespace, "theirs.yaml")

    assert namespace.ais_perf_model_value == "theirs.yaml"
    assert namespace.ais_perf_model_source == ENV_VAR
    assert old_ais_perf_inputs(namespace) == []


def test_conflicting_uses_are_an_error_of_the_given_parser():
    class Raising(argparse.ArgumentParser):
        def error(self, message):
            raise ValueError(message)

    parser = build_mocker_parser({}, Raising(add_help=False))

    namespace = parser.parse_args(["--ais-perf-model"])
    with pytest.raises(
        ValueError,
        match=f"^--ais-perf-model without a value cannot be combined with {ENV_VAR}$",
    ):
        set_ais_perf_model_attributes(parser, namespace, "theirs.yaml")

    namespace = parser.parse_args(["--ais-perf-model", "a.yaml", "--ais-perf-model"])
    with pytest.raises(ValueError, match="^--ais-perf-model is given with and without"):
        set_ais_perf_model_attributes(parser, namespace, None)
