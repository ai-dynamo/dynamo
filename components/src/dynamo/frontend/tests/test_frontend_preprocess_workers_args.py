# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from pathlib import Path

import pytest

from dynamo.frontend import frontend_args
from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def parse_frontend_config(args: list[str]) -> FrontendConfig:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args(args))
    config.validate()
    return config


@pytest.mark.parametrize("cpu_count, expected", [(1, 0), (2, 1), (3, 2), (16, 2)])
def test_sglang_auto_workers_follow_cpu_budget(
    monkeypatch: pytest.MonkeyPatch, cpu_count: int, expected: int
) -> None:
    monkeypatch.delenv("DYN_PREPROCESS_WORKERS", raising=False)
    monkeypatch.setattr(
        frontend_args.os,
        "sched_getaffinity",
        lambda _pid: set(range(cpu_count)),
        raising=False,
    )
    monkeypatch.setattr(frontend_args, "_cpu_quota_count", lambda: None)

    config = parse_frontend_config(["--dyn-chat-processor", "sglang"])

    assert config.preprocess_workers == expected


def test_sglang_auto_workers_respect_cgroup_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DYN_PREPROCESS_WORKERS", raising=False)
    monkeypatch.setattr(
        frontend_args.os,
        "sched_getaffinity",
        lambda _pid: set(range(16)),
        raising=False,
    )
    monkeypatch.setattr(frontend_args, "_cpu_quota_count", lambda: 2)

    assert (
        parse_frontend_config(["--dyn-chat-processor", "sglang"]).preprocess_workers
        == 1
    )


@pytest.mark.parametrize("processor", ["dynamo", "vllm"])
def test_other_processors_keep_zero_default_without_cpu_probe(
    monkeypatch: pytest.MonkeyPatch, processor: str
) -> None:
    monkeypatch.delenv("DYN_PREPROCESS_WORKERS", raising=False)
    monkeypatch.setattr(
        frontend_args,
        "_default_sglang_preprocess_workers",
        lambda: pytest.fail("CPU probe should be SGLang-only"),
    )

    assert (
        parse_frontend_config(["--dyn-chat-processor", processor]).preprocess_workers
        == 0
    )


def test_explicit_zero_disables_sglang_auto_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DYN_PREPROCESS_WORKERS", "2")
    monkeypatch.setattr(
        frontend_args,
        "_default_sglang_preprocess_workers",
        lambda: pytest.fail("explicit value must take precedence"),
    )

    assert (
        parse_frontend_config(
            ["--dyn-chat-processor", "sglang", "--dyn-preprocess-workers", "0"]
        ).preprocess_workers
        == 0
    )

    monkeypatch.setenv("DYN_PREPROCESS_WORKERS", "0")
    assert (
        parse_frontend_config(["--dyn-chat-processor", "sglang"]).preprocess_workers
        == 0
    )


def test_explicit_worker_count_is_not_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DYN_PREPROCESS_WORKERS", raising=False)
    assert (
        parse_frontend_config(
            ["--dyn-chat-processor", "sglang", "--dyn-preprocess-workers", "4"]
        ).preprocess_workers
        == 4
    )


def test_negative_worker_count_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DYN_PREPROCESS_WORKERS", raising=False)
    with pytest.raises(ValueError, match="--dyn-preprocess-workers must be >= 0"):
        parse_frontend_config(
            ["--dyn-chat-processor", "sglang", "--dyn-preprocess-workers", "-1"]
        )


def test_cpu_count_falls_back_when_affinity_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_affinity(_pid: int) -> set[int]:
        raise OSError("affinity unavailable")

    monkeypatch.setattr(os, "sched_getaffinity", no_affinity, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    monkeypatch.setattr(frontend_args, "_cpu_quota_count", lambda: None)

    assert frontend_args._default_sglang_preprocess_workers() == 1


@pytest.mark.parametrize(
    "quota, expected",
    [("200000 100000", 2), ("150000 100000", 1), ("max 100000", None)],
)
def test_cgroup_v2_cpu_quota(
    monkeypatch: pytest.MonkeyPatch, quota: str, expected: int | None
) -> None:
    original_read_text = Path.read_text

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        if str(path) == "/sys/fs/cgroup/cpu.max":
            return quota
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)

    assert frontend_args._cpu_quota_count() == expected


def test_cgroup_v1_cpu_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "200000",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000",
    }

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        if str(path) == "/sys/fs/cgroup/cpu.max":
            raise FileNotFoundError
        return values[str(path)]

    monkeypatch.setattr(Path, "read_text", read_text)

    assert frontend_args._cpu_quota_count() == 2
