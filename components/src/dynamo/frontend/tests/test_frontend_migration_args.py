# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture(autouse=True)
def clear_migration_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DYN_MIGRATION_LIMIT",
        "DYN_MIGRATION_LIMIT_DRAINING",
        "DYN_MIGRATION_MAX_SEQ_LEN",
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


def test_draining_limit_omission_matches_zero() -> None:
    omitted = parse_frontend_config(["--migration-limit", "2"])
    explicit = parse_frontend_config(
        ["--migration-limit", "2", "--migration-limit-draining", "0"]
    )
    assert omitted.migration_limit == explicit.migration_limit == 2
    assert omitted.migration_limit_draining == explicit.migration_limit_draining == 0


def test_draining_limit_cli_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DYN_MIGRATION_LIMIT", "2")
    monkeypatch.setenv("DYN_MIGRATION_LIMIT_DRAINING", "5")
    assert parse_frontend_config([]).migration_limit_draining == 5
    config = parse_frontend_config(["--migration-limit-draining", "0"])
    assert config.migration_limit == 2
    assert config.migration_limit_draining == 0


@pytest.mark.parametrize("value", [-1, 2**32])
@pytest.mark.parametrize("source", ["cli", "env"])
def test_draining_limit_rejects_values_outside_u32(
    monkeypatch: pytest.MonkeyPatch, value: int, source: str
) -> None:
    argv = []
    if source == "env":
        monkeypatch.setenv("DYN_MIGRATION_LIMIT_DRAINING", str(value))
    else:
        argv = ["--migration-limit-draining", str(value)]
    with pytest.raises(ValueError, match="--migration-limit-draining must be between"):
        parse_frontend_config(argv)


def test_draining_limit_accepts_u32_maximum() -> None:
    assert (
        parse_frontend_config(
            ["--migration-limit-draining", str(2**32 - 1)]
        ).migration_limit_draining
        == 2**32 - 1
    )
