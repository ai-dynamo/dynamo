#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path

import pytest

from dynamo.frontend.cpu_affinity import (
    NumaAffinityStatus,
    detect_cpu_numa_affinity,
    format_cpu_list,
    log_frontend_cpu_affinity,
    parse_cpu_list,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _write_node(root: Path, node: int, cpus: str) -> None:
    node_path = root / f"node{node}"
    node_path.mkdir()
    (node_path / "cpulist").write_text(cpus)


def test_parse_cpu_list_supports_ranges_and_single_ids():
    assert parse_cpu_list("0-2,8,16-17\n") == {0, 1, 2, 8, 16, 17}


@pytest.mark.parametrize("value", ["0-", "2-1", "1--2", "cpu3", "0,,2"])
def test_parse_cpu_list_rejects_invalid_values(value: str):
    with pytest.raises(ValueError):
        parse_cpu_list(value)


def test_format_cpu_list_compacts_ranges_and_sorts_values():
    assert format_cpu_list([8, 2, 1, 0, 7, 4]) == "0-2,4,7-8"
    assert format_cpu_list([]) == "none"


def test_detects_affinity_confined_to_one_numa_node(tmp_path: Path):
    _write_node(tmp_path, 0, "0-3,8-11")
    _write_node(tmp_path, 1, "4-7,12-15")

    result = detect_cpu_numa_affinity(
        node_root=tmp_path,
        affinity_getter=lambda _pid: {0, 2, 8},
    )

    assert result.status == NumaAffinityStatus.SINGLE_NODE
    assert result.available_cpus == (0, 2, 8)
    assert result.numa_nodes == (0,)
    assert result.unmapped_cpus == ()
    assert result.reason is None


def test_detects_affinity_spanning_multiple_numa_nodes(tmp_path: Path):
    _write_node(tmp_path, 0, "0-3")
    _write_node(tmp_path, 1, "4-7")

    result = detect_cpu_numa_affinity(
        node_root=tmp_path,
        affinity_getter=lambda _pid: {1, 4, 6},
    )

    assert result.status == NumaAffinityStatus.MULTIPLE_NODES
    assert result.available_cpus == (1, 4, 6)
    assert result.numa_nodes == (0, 1)
    assert result.unmapped_cpus == ()


def test_multiple_nodes_remains_conclusive_with_an_unmapped_cpu(tmp_path: Path):
    _write_node(tmp_path, 0, "0-3")
    _write_node(tmp_path, 1, "4-7")

    result = detect_cpu_numa_affinity(
        node_root=tmp_path,
        affinity_getter=lambda _pid: {1, 4, 9},
    )

    assert result.status == NumaAffinityStatus.MULTIPLE_NODES
    assert result.numa_nodes == (0, 1)
    assert result.unmapped_cpus == (9,)


def test_partial_cpu_mapping_is_unknown(tmp_path: Path):
    _write_node(tmp_path, 0, "0-3")

    result = detect_cpu_numa_affinity(
        node_root=tmp_path,
        affinity_getter=lambda _pid: {1, 4},
    )

    assert result.status == NumaAffinityStatus.UNKNOWN
    assert result.numa_nodes == (0,)
    assert result.unmapped_cpus == (4,)
    assert result.reason is not None
    assert "did not map all available CPUs" in result.reason


@pytest.mark.parametrize("create_root", [False, True])
def test_missing_or_empty_topology_is_unknown(tmp_path: Path, create_root: bool):
    node_root = tmp_path / "nodes"
    if create_root:
        node_root.mkdir()

    result = detect_cpu_numa_affinity(
        node_root=node_root,
        affinity_getter=lambda _pid: {0, 1},
    )

    assert result.status == NumaAffinityStatus.UNKNOWN
    assert result.available_cpus == (0, 1)
    assert result.unmapped_cpus == (0, 1)
    assert result.reason is not None
    assert "failed to read NUMA topology" in result.reason


def test_unsupported_affinity_api_is_unknown(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)

    result = detect_cpu_numa_affinity()

    assert result.status == NumaAffinityStatus.UNKNOWN
    assert result.reason == "os.sched_getaffinity is unavailable"


def test_affinity_syscall_error_is_unknown(tmp_path: Path):
    def fail_affinity(_pid: int) -> set[int]:
        raise OSError("not permitted")

    result = detect_cpu_numa_affinity(
        node_root=tmp_path,
        affinity_getter=fail_affinity,
    )

    assert result.status == NumaAffinityStatus.UNKNOWN
    assert result.reason == "sched_getaffinity failed: not permitted"


def test_logging_fails_open_on_unexpected_detection_error(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    def fail_affinity(_pid: int) -> set[int]:
        raise RuntimeError("unexpected failure")

    test_logger = logging.getLogger("test.frontend.cpu_affinity.fail_open")
    with caplog.at_level(logging.WARNING, logger=test_logger.name):
        result = log_frontend_cpu_affinity(
            test_logger,
            node_root=tmp_path,
            affinity_getter=fail_affinity,
        )

    records = [record for record in caplog.records if record.name == test_logger.name]
    assert result.status == NumaAffinityStatus.UNKNOWN
    assert result.reason == "unexpected affinity detection error: unexpected failure"
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING


@pytest.mark.parametrize(
    ("available_cpus", "expected_status", "expected_level", "expected_message"),
    [
        (
            {0, 1},
            NumaAffinityStatus.SINGLE_NODE,
            logging.INFO,
            "confined to one NUMA node",
        ),
        (
            {1, 4},
            NumaAffinityStatus.MULTIPLE_NODES,
            logging.WARNING,
            "spans multiple NUMA nodes",
        ),
        (
            {1, 8},
            NumaAffinityStatus.UNKNOWN,
            logging.WARNING,
            "Unable to determine",
        ),
    ],
)
def test_logs_exactly_one_affinity_result(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    available_cpus: set[int],
    expected_status: NumaAffinityStatus,
    expected_level: int,
    expected_message: str,
):
    _write_node(tmp_path, 0, "0-3")
    _write_node(tmp_path, 1, "4-7")
    test_logger = logging.getLogger("test.frontend.cpu_affinity")

    with caplog.at_level(logging.INFO, logger=test_logger.name):
        result = log_frontend_cpu_affinity(
            test_logger,
            node_root=tmp_path,
            affinity_getter=lambda _pid: available_cpus,
        )

    records = [record for record in caplog.records if record.name == test_logger.name]
    assert result.status == expected_status
    assert len(records) == 1
    assert records[0].levelno == expected_level
    assert expected_message in records[0].getMessage()
