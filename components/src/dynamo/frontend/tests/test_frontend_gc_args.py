# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend GC flags: --dyn-freeze-gc-heap / --dyn-gc-pause-log-ms."""

import argparse

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("DYN_FREEZE_GC_HEAP", "DYN_GC_PAUSE_LOG_MS"):
        monkeypatch.delenv(name, raising=False)


def parse_frontend_config(args: list[str]) -> FrontendConfig:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args(args))
    config.validate()
    return config


def test_freeze_defaults_on_and_pause_log_off() -> None:
    config = parse_frontend_config([])
    assert config.freeze_gc_heap is True
    assert config.gc_pause_log_ms == 0.0


@pytest.mark.parametrize(
    ("args", "env_value", "expected"),
    [
        (["--no-dyn-freeze-gc-heap"], None, False),
        (["--dyn-freeze-gc-heap"], None, True),
        ([], "false", False),
        ([], "0", False),
        # CLI wins over the environment in both directions.
        (["--dyn-freeze-gc-heap"], "false", True),
        (["--no-dyn-freeze-gc-heap"], "true", False),
    ],
)
def test_freeze_flag_and_env(
    monkeypatch: pytest.MonkeyPatch, args, env_value, expected
) -> None:
    if env_value is not None:
        monkeypatch.setenv("DYN_FREEZE_GC_HEAP", env_value)
    assert parse_frontend_config(args).freeze_gc_heap is expected


def test_pause_log_threshold_from_cli_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert (
        parse_frontend_config(["--dyn-gc-pause-log-ms", "50"]).gc_pause_log_ms == 50.0
    )
    monkeypatch.setenv("DYN_GC_PAUSE_LOG_MS", "12.5")
    assert parse_frontend_config([]).gc_pause_log_ms == 12.5
    assert parse_frontend_config(["--dyn-gc-pause-log-ms", "0"]).gc_pause_log_ms == 0.0


@pytest.mark.parametrize("value", ["-1", "nan"])
def test_pause_log_threshold_rejects_negative_and_nan(value: str) -> None:
    with pytest.raises(ValueError, match="--dyn-gc-pause-log-ms must be >= 0"):
        parse_frontend_config(["--dyn-gc-pause-log-ms", value])


def test_help_lists_flags_and_env() -> None:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    help_text = parser.format_help()
    for needle in (
        "--dyn-freeze-gc-heap",
        "--no-dyn-freeze-gc-heap",
        "DYN_FREEZE_GC_HEAP",
        "--dyn-gc-pause-log-ms",
        "DYN_GC_PAUSE_LOG_MS",
    ):
        assert needle in help_text
