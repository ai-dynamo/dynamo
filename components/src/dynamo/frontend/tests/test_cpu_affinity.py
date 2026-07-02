#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path

import pytest

from dynamo.frontend import cpu_affinity
from dynamo.frontend.cpu_affinity import warn_if_frontend_cpu_affinity_spans_numa_nodes

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def _add_cpu(root: Path, cpu: int, node: int) -> None:
    (root / f"cpu{cpu}" / f"node{node}").mkdir(parents=True)


def test_warns_when_affinity_spans_multiple_numa_nodes(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    for cpu in (0, 1, 2, 3):
        _add_cpu(tmp_path, cpu, 0)
    for cpu in (8, 9):
        _add_cpu(tmp_path, cpu, 1)
    monkeypatch.setattr(cpu_affinity, "_CPU_SYSFS_ROOT", tmp_path)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3, 8, 9})
    logger = logging.getLogger("test.frontend.cpu_affinity.multiple")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        warn_if_frontend_cpu_affinity_spans_numa_nodes(logger)

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "Frontend CPU affinity spans multiple NUMA nodes" in message
    assert "Available CPUs: 0-3,8-9" in message
    assert "node0" in message
    assert "node1" in message


@pytest.mark.parametrize(
    ("cpus", "cpu_nodes", "debug_message"),
    [
        pytest.param({0, 1}, [(0, 0), (1, 0)], None, id="same_node"),
        pytest.param({8}, [], "no NUMA node links found", id="cpu_missing_from_sysfs"),
    ],
)
def test_does_not_warn_without_multiple_detected_nodes(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    cpus: set[int],
    cpu_nodes: list[tuple[int, int]],
    debug_message: str | None,
):
    for cpu, node in cpu_nodes:
        _add_cpu(tmp_path, cpu, node)
    monkeypatch.setattr(cpu_affinity, "_CPU_SYSFS_ROOT", tmp_path)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: cpus)
    logger = logging.getLogger("test.frontend.cpu_affinity.silent")

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        warn_if_frontend_cpu_affinity_spans_numa_nodes(logger)

    assert not [
        record for record in caplog.records if record.levelno >= logging.WARNING
    ]
    if debug_message is None:
        assert caplog.records == []
    else:
        assert len(caplog.records) == 1
        assert debug_message in caplog.records[0].getMessage()


def test_affinity_failure_logs_debug_traceback_without_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    def fail_affinity(_pid: int) -> set[int]:
        raise OSError("not permitted")

    monkeypatch.setattr(os, "sched_getaffinity", fail_affinity)
    logger = logging.getLogger("test.frontend.cpu_affinity.failure")

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        warn_if_frontend_cpu_affinity_spans_numa_nodes(logger)

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.DEBUG
    assert "not permitted" in caplog.records[0].getMessage()
    assert caplog.records[0].exc_info is not None
