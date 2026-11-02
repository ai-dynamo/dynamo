# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``build_frontend_parser``: another frontend's parser, against an explicit env."""

import argparse
import logging
import os

import pytest

from dynamo.frontend.frontend_args import build_frontend_parser

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
