# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NIXL/POSIX staging restore planning."""

import threading

import pytest

try:
    from gpu_memory_service.snapshot.backends.nixl_staging import (
        DEFAULT_POSIX_IOS_POOL_SIZE,
        DEFAULT_POSIX_KERNEL_QUEUE_SIZE,
        POSIX_IOS_POOL_SIZE_CONFIG_KEY,
        POSIX_KERNEL_QUEUE_SIZE_CONFIG_KEY,
        NixlFileGroup,
        NixlWorkGroup,
        _NixlPosixStagingTransferSession,
        _posix_backend_params_from_config,
        _split_work_groups,
    )
    from gpu_memory_service.snapshot.transfer import (
        FileTransferSource,
        GMSTransferTarget,
    )
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


def test_posix_backend_params_default_to_bounded_pool():
    assert _posix_backend_params_from_config({}) == {
        "ios_pool_size": str(DEFAULT_POSIX_IOS_POOL_SIZE),
        "kernel_queue_size": str(DEFAULT_POSIX_KERNEL_QUEUE_SIZE),
    }


def test_posix_backend_params_allow_override():
    assert _posix_backend_params_from_config(
        {
            POSIX_IOS_POOL_SIZE_CONFIG_KEY: "64",
            POSIX_KERNEL_QUEUE_SIZE_CONFIG_KEY: 16,
        }
    ) == {
        "ios_pool_size": "64",
        "kernel_queue_size": "16",
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (POSIX_IOS_POOL_SIZE_CONFIG_KEY, "0"),
        (POSIX_KERNEL_QUEUE_SIZE_CONFIG_KEY, "-1"),
        (POSIX_IOS_POOL_SIZE_CONFIG_KEY, "not-an-int"),
    ],
)
def test_posix_backend_params_reject_invalid_values(key, value):
    with pytest.raises(ValueError, match=key):
        _posix_backend_params_from_config({key: value})


def test_posix_backend_params_are_forwarded_to_nixl_agent(monkeypatch):
    from gpu_memory_service.snapshot.backends import nixl_staging

    source = FileTransferSource(
        allocation_id="alloc-0",
        file_path="/checkpoint/shard.bin",
        file_offset=0,
        byte_count=4096,
    )
    target = GMSTransferTarget(
        allocation_id="alloc-0",
        va=0x1000,
        device=0,
        byte_count=4096,
    )
    captured = {}

    class FakeApi:
        @staticmethod
        def agent_config_type(*, backends):
            captured["config_backends"] = backends
            return {"backends": backends}

        @staticmethod
        def agent_type(agent_name, _config):
            captured["agent_name"] = agent_name
            return FakeAgent()

    class FakeAgent:
        def create_backend(self, backend_name, backend_params=None):
            captured["backend_name"] = backend_name
            captured["backend_params"] = backend_params

    def group_sources(_sources):
        return {"file": [(source.file_path, [source])]}

    def fake_restore_file_groups_with_nixl_staging(**kwargs):
        captured["restore_agent"] = kwargs["agent"]
        captured["restore_agent_name"] = kwargs["agent_name"]
        captured["restore_file_groups"] = kwargs["file_groups"]
        return source.byte_count

    monkeypatch.setattr(
        nixl_staging.cuda_utils,
        "cuda_runtime_set_device",
        lambda _device: None,
    )
    monkeypatch.setattr(
        nixl_staging,
        "restore_file_groups_with_nixl_staging",
        fake_restore_file_groups_with_nixl_staging,
    )
    monkeypatch.setattr(
        nixl_staging,
        "load_nixl_api",
        lambda: FakeApi(),
    )
    monkeypatch.setattr(
        nixl_staging,
        "make_pinned_copy_slots",
        lambda _count: [],
    )

    session = _NixlPosixStagingTransferSession(
        backend_name="test-backend",
        device=0,
        max_workers=1,
        group_sources=group_sources,
        group_kind="file",
        warn_under_parallelized=False,
        posix_backend_params={"ios_pool_size": "64", "kernel_queue_size": "16"},
        sources=[source],
    )

    session.restore({"alloc-0": target})

    assert captured["config_backends"] == []
    assert captured["backend_name"] == "POSIX"
    assert captured["backend_params"] == {
        "ios_pool_size": "64",
        "kernel_queue_size": "16",
    }
    assert captured["restore_agent_name"] == captured["agent_name"]
    assert captured["restore_file_groups"] == [(source.file_path, [source])]


def test_staging_prep_starts_before_restore(monkeypatch):
    from gpu_memory_service.snapshot.backends import nixl_staging

    source = FileTransferSource(
        allocation_id="alloc-0",
        file_path="/checkpoint/shard.bin",
        file_offset=0,
        byte_count=4096,
    )
    prep_started = threading.Event()
    allow_finish = threading.Event()

    class FakeApi:
        @staticmethod
        def agent_config_type(*, backends):
            return {"backends": backends}

        @staticmethod
        def agent_type(_agent_name, _config):
            return FakeAgent()

    class FakeAgent:
        def create_backend(self, _backend_name, backend_params=None):
            self.backend_params = backend_params

    def group_sources(_sources):
        return {"file": [(source.file_path, [source])]}

    def fake_load_nixl_api():
        prep_started.set()
        assert allow_finish.wait(timeout=1.0)
        return FakeApi()

    monkeypatch.setattr(
        nixl_staging.cuda_utils,
        "cuda_runtime_set_device",
        lambda _device: None,
    )
    monkeypatch.setattr(nixl_staging, "load_nixl_api", fake_load_nixl_api)
    monkeypatch.setattr(nixl_staging, "make_pinned_copy_slots", lambda _count: [])

    session = _NixlPosixStagingTransferSession(
        backend_name="test-backend",
        device=0,
        max_workers=1,
        group_sources=group_sources,
        group_kind="file",
        warn_under_parallelized=False,
        posix_backend_params={"ios_pool_size": "64", "kernel_queue_size": "16"},
        sources=[source],
    )
    try:
        assert prep_started.wait(timeout=1.0)
    finally:
        allow_finish.set()
        session.close()
