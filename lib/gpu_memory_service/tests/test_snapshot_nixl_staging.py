# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NIXL/POSIX staging restore planning."""

import pytest

try:
    from gpu_memory_service.snapshot.backends.nixl_staging import (
        NixlFileGroup,
        NixlWorkGroup,
        _split_work_groups,
    )
    from gpu_memory_service.snapshot.transfer import FileTransferSource
except ModuleNotFoundError:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def _work_group(name: str, byte_count: int) -> NixlWorkGroup:
    source = FileTransferSource(
        allocation_id=name,
        file_path=f"/checkpoint/{name}.bin",
        file_offset=0,
        byte_count=byte_count,
    )
    file_group: NixlFileGroup = (source.file_path, [source])
    return name, [file_group]


def test_split_work_groups_limits_bucket_count():
    work_groups = [_work_group(f"shard-{idx}", 1) for idx in range(142)]

    buckets = _split_work_groups(work_groups, worker_count=8)

    assert len(buckets) == 8
    assert sum(len(file_groups) for _name, file_groups in buckets) == 142
    expected = sorted(f"shard-{idx}" for idx in range(142))
    actual = sorted(
        source.allocation_id
        for _bucket_name, file_groups in buckets
        for _file_path, sources in file_groups
        for source in sources
    )
    assert actual == expected


def test_split_work_groups_keeps_existing_groups_when_not_over_limit():
    work_groups = [_work_group("a", 1), _work_group("b", 2)]

    assert _split_work_groups(work_groups, worker_count=8) == work_groups


def test_split_work_groups_balances_by_bytes():
    work_groups = [
        _work_group("large", 10),
        _work_group("medium-a", 6),
        _work_group("medium-b", 6),
        _work_group("small-a", 1),
        _work_group("small-b", 1),
    ]

    buckets = _split_work_groups(work_groups, worker_count=2)
    bucket_sizes = sorted(
        sum(source.byte_count for _path, sources in file_groups for source in sources)
        for _name, file_groups in buckets
    )

    assert bucket_sizes == [12, 12]
