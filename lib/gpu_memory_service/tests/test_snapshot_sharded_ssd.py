# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for sharded SSD restore planning."""

import pytest

try:
    from gpu_memory_service.snapshot.backends.sharded_ssd import (
        SHARDED_SSD_QUEUES_PER_ROOT_CONFIG_KEY,
        _group_sources_by_root,
        _queues_per_root_from_config,
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


def _source(path: str, byte_count: int) -> FileTransferSource:
    return FileTransferSource(
        allocation_id=path.rsplit("/", 1)[-1],
        file_path=path,
        file_offset=0,
        byte_count=byte_count,
    )


def test_group_sources_by_root_preserves_single_queue_default():
    sources = [
        _source("/mnt/nvme0/gms/shard_0000.bin", 1),
        _source("/mnt/nvme0/gms/shard_0001.bin", 1),
        _source("/mnt/nvme1/gms/shard_0002.bin", 1),
    ]

    groups = _group_sources_by_root(
        sources,
        ["/mnt/nvme0/gms", "/mnt/nvme1/gms"],
    )

    assert list(groups) == ["/mnt/nvme0/gms", "/mnt/nvme1/gms"]
    assert [file_path for file_path, _sources in groups["/mnt/nvme0/gms"]] == [
        "/mnt/nvme0/gms/shard_0000.bin",
        "/mnt/nvme0/gms/shard_0001.bin",
    ]


def test_group_sources_by_root_splits_each_root_into_balanced_queues():
    sources = [
        _source("/mnt/nvme0/gms/shard_0000.bin", 10),
        _source("/mnt/nvme0/gms/shard_0001.bin", 6),
        _source("/mnt/nvme0/gms/shard_0002.bin", 6),
        _source("/mnt/nvme0/gms/shard_0003.bin", 2),
        _source("/mnt/nvme1/gms/shard_0004.bin", 8),
        _source("/mnt/nvme1/gms/shard_0005.bin", 8),
    ]

    groups = _group_sources_by_root(
        sources,
        ["/mnt/nvme0/gms", "/mnt/nvme1/gms"],
        queues_per_root=2,
    )

    assert set(groups) == {
        "/mnt/nvme0/gms#q0",
        "/mnt/nvme0/gms#q1",
        "/mnt/nvme1/gms#q0",
        "/mnt/nvme1/gms#q1",
    }
    assert sorted(
        sum(source.byte_count for _path, grouped in file_groups for source in grouped)
        for file_groups in groups.values()
    ) == [8, 8, 12, 12]


def test_group_sources_by_root_rejects_paths_outside_configured_roots():
    source = _source("/mnt/other/gms/shard_0000.bin", 1)

    with pytest.raises(RuntimeError, match="not.*under.*sharded SSD root"):
        _group_sources_by_root([source], ["/mnt/nvme0/gms"], queues_per_root=2)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (None, 1),
        ("", 1),
        ("2", 2),
        (4, 4),
    ],
)
def test_queues_per_root_from_config(raw_value, expected):
    config = {}
    if raw_value is not None:
        config[SHARDED_SSD_QUEUES_PER_ROOT_CONFIG_KEY] = raw_value

    assert _queues_per_root_from_config(config) == expected


@pytest.mark.parametrize("raw_value", ["0", "-1", "not-an-int"])
def test_queues_per_root_from_config_rejects_invalid_values(raw_value):
    with pytest.raises(ValueError, match=SHARDED_SSD_QUEUES_PER_ROOT_CONFIG_KEY):
        _queues_per_root_from_config(
            {SHARDED_SSD_QUEUES_PER_ROOT_CONFIG_KEY: raw_value}
        )
