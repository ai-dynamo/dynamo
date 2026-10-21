# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def parse_config(argv: list[str]) -> FrontendConfig:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args(argv))
    config.validate()
    return config


def test_realtime_budget_default_environment_and_cli(monkeypatch):
    monkeypatch.delenv("DYN_MIGRATION_REALTIME_BUDGET_MIB", raising=False)
    assert parse_config([]).migration_realtime_budget_mib == 512
    monkeypatch.setenv("DYN_MIGRATION_REALTIME_BUDGET_MIB", "64")
    assert parse_config([]).migration_realtime_budget_mib == 64
    config = parse_config(["--migration-realtime-budget-mib", "0"])
    assert config.migration_realtime_budget_mib == 0
    assert config.__dict__["migration_realtime_budget_mib"] == 0


@pytest.mark.parametrize("value", ["-1", str(2**32)])
def test_realtime_budget_rejects_invalid_cli_and_environment(monkeypatch, value):
    with pytest.raises(ValueError, match="--migration-realtime-budget-mib"):
        parse_config(["--migration-realtime-budget-mib", value])
    monkeypatch.setenv("DYN_MIGRATION_REALTIME_BUDGET_MIB", value)
    with pytest.raises(ValueError, match="--migration-realtime-budget-mib"):
        parse_config([])
