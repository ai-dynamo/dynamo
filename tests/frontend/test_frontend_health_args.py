# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration coverage for model-aware HTTP health checks."""

import argparse

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def parse_config(args):
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    return FrontendConfig.from_cli_args(parser.parse_args(args))


@pytest.mark.parametrize(
    ("env", "args", "expected"),
    [
        (None, [], False),
        (None, ["--health-check-requires-model"], True),
        ("true", [], True),
        ("true", ["--no-health-check-requires-model"], False),
        ("false", ["--health-check-requires-model"], True),
    ],
)
def test_health_check_requires_model_config(monkeypatch, env, args, expected):
    monkeypatch.delenv("DYN_HEALTH_CHECK_REQUIRES_MODEL", raising=False)
    if env is not None:
        monkeypatch.setenv("DYN_HEALTH_CHECK_REQUIRES_MODEL", env)
    assert parse_config(args).health_check_requires_model is expected
