#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path

import pytest

from dynamo.frontend.cpu_affinity import warn_if_frontend_cpu_affinity_spans_numa_nodes

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def _add_cpu(root: Path, cpu: int, node: int) -> None:
    (root / f"cpu{cpu}" / f"node{node}").mkdir(parents=True)


def test_warns_when_affinity_spans_multiple_numa_nodes(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    _add_cpu(tmp_path, 0, 0)
    _add_cpu(tmp_path, 4, 1)
    logger = logging.getLogger("test.frontend.cpu_affinity.multiple")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        warn_if_frontend_cpu_affinity_spans_numa_nodes(
            logger,
            cpu_root=tmp_path,
            affinity_getter=lambda _pid: {0, 4},
        )

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert message.startswith("=" * 80)
    assert "WARNING: Frontend CPU affinity spans multiple NUMA nodes!" in message
    assert "Available CPUs: [0, 4]" in message
    assert "NUMA nodes: [0, 1]" in message


@pytest.mark.parametrize("cpus", [{0, 1}, {0, 8}])
def test_stays_silent_without_multiple_detected_nodes(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    cpus: set[int],
):
    _add_cpu(tmp_path, 0, 0)
    _add_cpu(tmp_path, 1, 0)
    logger = logging.getLogger("test.frontend.cpu_affinity.silent")

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        warn_if_frontend_cpu_affinity_spans_numa_nodes(
            logger,
            cpu_root=tmp_path,
            affinity_getter=lambda _pid: cpus,
        )

    assert caplog.records == []


def test_stays_silent_when_affinity_detection_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)

    warn_if_frontend_cpu_affinity_spans_numa_nodes(
        logging.getLogger("test.frontend.cpu_affinity.unavailable")
    )

    assert caplog.records == []
