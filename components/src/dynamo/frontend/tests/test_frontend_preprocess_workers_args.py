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


def mock_proc_cgroup(
    monkeypatch: pytest.MonkeyPatch, membership: str, mountinfo: str
) -> None:
    original_read_text = Path.read_text

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        if str(path) == "/proc/self/cgroup":
            return membership
        if str(path) == "/proc/self/mountinfo":
            return mountinfo
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)


@pytest.mark.parametrize(
    "parent_quota, child_quota, expected",
    [
        ("100000 100000", "max 100000", 1),
        ("200000 100000", "50000 50000", 1),
        ("200000 100000", "150000 50000", 2),
        ("max 100000", "max 100000", None),
    ],
)
def test_cgroup_v2_uses_tightest_visible_quota(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    parent_quota: str,
    child_quota: str,
    expected: int | None,
) -> None:
    root = tmp_path / "cgroup"
    child = root / "pod" / "container"
    child.mkdir(parents=True)
    (child.parent / "cpu.max").write_text(parent_quota)
    (child / "cpu.max").write_text(child_quota)
    mock_proc_cgroup(
        monkeypatch,
        "0::/pod/container",
        f"31 24 0:28 / {root} rw - cgroup2 cgroup rw",
    )

    assert frontend_args._cpu_quota_count() == expected


def test_cgroup_v2_namespaced_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "cgroup mount"
    root.mkdir()
    (root / "cpu.max").write_text("200000 100000")
    encoded_root = str(root).replace(" ", r"\040")
    mock_proc_cgroup(
        monkeypatch,
        "0::/",
        f"31 24 0:28 / {encoded_root} rw - cgroup2 cgroup rw",
    )

    assert frontend_args._cpu_quota_count() == 2


def test_cgroup_ambiguous_mount_is_conservative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cpu.max").write_text("800000 100000")
    mock_proc_cgroup(
        monkeypatch,
        "0::/",
        f"31 24 0:28 /kubepods/pod {root} rw - cgroup2 cgroup rw",
    )

    assert frontend_args._cpu_quota_count() == 1


def test_cgroup_v1_uses_ancestor_quota(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "cpu,cpuacct"
    child = root / "pod" / "container"
    child.mkdir(parents=True)
    (child.parent / "cpu.cfs_quota_us").write_text("100000")
    (child.parent / "cpu.cfs_period_us").write_text("100000")
    (child / "cpu.cfs_quota_us").write_text("-1")
    (child / "cpu.cfs_period_us").write_text("100000")
    mock_proc_cgroup(
        monkeypatch,
        "2:cpu,cpuacct:/pod/container",
        f"31 24 0:28 / {root} rw - cgroup cgroup rw,cpu,cpuacct",
    )

    assert frontend_args._cpu_quota_count() == 1
