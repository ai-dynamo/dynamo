# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NIXL/POSIX staging restore planning."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

try:
    from gpu_memory_service.snapshot.backends.nixl_common import (
        NixlFileGroup,
        NixlWorkGroup,
        split_work_groups,
    )
    from gpu_memory_service.snapshot.backends.nixl_staging import (
        _PreparedNixlGroup,
        _NixlPosixStagingTransferSession,
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

    buckets = split_work_groups(work_groups, worker_count=8)

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


def test_split_work_groups_balances_by_bytes():
    work_groups = [
        _work_group("large", 10),
        _work_group("medium-a", 6),
        _work_group("medium-b", 6),
        _work_group("small-a", 1),
        _work_group("small-b", 1),
    ]

    buckets = split_work_groups(work_groups, worker_count=2)
    bucket_sizes = sorted(
        sum(source.byte_count for _path, sources in file_groups for source in sources)
        for _name, file_groups in buckets
    )

    assert bucket_sizes == [12, 12]


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


def test_transfer_gate_runs_after_all_prep_and_before_transfer(monkeypatch):
    sources = [
        FileTransferSource(
            allocation_id=f"alloc-{idx}",
            file_path=f"/checkpoint/shard-{idx}.bin",
            file_offset=0,
            byte_count=4096,
        )
        for idx in range(2)
    ]
    gate_entered = threading.Event()
    release_gate = threading.Event()
    transfers: list[str] = []

    class FakeGate:
        def wait(self, participant):
            assert participant == 0
            gate_entered.set()
            assert release_gate.wait(timeout=1.0)

    def group_sources(_sources):
        return {
            source.allocation_id: [(source.file_path, [source])] for source in sources
        }

    def prepare_group(_session, _worker_index, group_name, file_groups):
        return _PreparedNixlGroup(
            group_name=group_name,
            file_groups=file_groups,
            agent=object(),
            agent_name=f"agent-{group_name}",
            slots=[],
            prep_elapsed_s=0.0,
        )

    monkeypatch.setattr(
        _NixlPosixStagingTransferSession,
        "_prepare_group",
        prepare_group,
    )
    session = _NixlPosixStagingTransferSession(
        backend_name="test-backend",
        device=0,
        max_workers=2,
        group_sources=group_sources,
        group_kind="file",
        warn_under_parallelized=False,
        posix_backend_params={},
        sources=sources,
        restore_transfer_gate=FakeGate(),
    )
    monkeypatch.setattr(
        session,
        "_restore_prepared_group",
        lambda prepared, _targets: transfers.append(prepared.group_name),
    )
    targets = {
        source.allocation_id: GMSTransferTarget(
            allocation_id=source.allocation_id,
            va=idx * 4096,
            device=0,
            byte_count=source.byte_count,
        )
        for idx, source in enumerate(sources)
    }

    with ThreadPoolExecutor(max_workers=1) as pool:
        restore = pool.submit(session.restore, targets)
        assert gate_entered.wait(timeout=1.0)
        assert transfers == []
        release_gate.set()
        restore.result(timeout=1.0)

    assert sorted(transfers) == ["alloc-0", "alloc-1"]
