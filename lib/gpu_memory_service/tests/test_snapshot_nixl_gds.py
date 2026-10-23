# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for direct NIXL/GDS restore integrity handling."""

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from blake3 import blake3
from gpu_memory_service.snapshot.backends import nixl_gds
from gpu_memory_service.snapshot.backends.nixl_gds import _NixlGDSTransferSession
from gpu_memory_service.snapshot.transfer import (
    FileIntegrity,
    FileTransferSource,
    GMSTransferTarget,
    blake3_threads_per_shard,
    verify_file_source_digests,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


@pytest.mark.parametrize(
    ("worker_count", "expected_threads"),
    [(1, 4), (4, 4), (8, 2), (16, 1), (32, 1)],
)
def test_blake3_threads_are_bounded_per_rank(worker_count, expected_threads):
    assert blake3_threads_per_shard(worker_count) == expected_threads


def test_blake3_threads_require_an_active_worker():
    with pytest.raises(ValueError, match="worker_count must be positive"):
        blake3_threads_per_shard(0)


def test_one_integrity_record_covers_multiple_allocation_extents(tmp_path):
    shard_path = tmp_path / "shard.bin"
    shard_path.write_bytes(b"abcdefgh")
    sources = [
        FileTransferSource("first", str(shard_path), 0, 4),
        FileTransferSource("second", str(shard_path), 4, 4),
    ]
    integrity = {str(shard_path): FileIntegrity(8, blake3(b"abcdefgh").hexdigest())}

    assert verify_file_source_digests(sources, integrity, max_workers=1) == (1, 8)
    with pytest.raises(RuntimeError, match="not contiguous"):
        verify_file_source_digests(sources[1:], integrity, max_workers=1)


def _session_and_targets(shard_path, expected_data):
    source = FileTransferSource(
        allocation_id="weight",
        file_path=str(shard_path),
        file_offset=0,
        byte_count=len(expected_data),
    )
    session = _NixlGDSTransferSession(
        agent=object(),
        agent_name="test-agent",
        device=0,
        max_workers=1,
        sources=[source],
        integrity_by_path={
            str(shard_path): FileIntegrity(
                len(expected_data), blake3(expected_data).hexdigest()
            )
        },
    )
    targets = {"weight": GMSTransferTarget("weight", 100, 0, len(expected_data))}
    return session, targets


def test_gds_verifies_before_starting_direct_transfer(tmp_path, monkeypatch):
    expected_data = b"checkpoint-bytes"
    shard_path = tmp_path / "shard.bin"
    shard_path.write_bytes(expected_data)
    session, targets = _session_and_targets(shard_path, expected_data)
    transfers = []
    monkeypatch.setattr(
        nixl_gds,
        "run_bounded_nixl_transfers",
        lambda **kwargs: transfers.append(kwargs["items"]),
    )

    session.restore(targets)

    assert len(transfers) == 1


def test_gds_rejects_same_length_mutation_before_direct_transfer(
    tmp_path,
    monkeypatch,
):
    expected_data = b"checkpoint-bytes"
    shard_path = tmp_path / "shard.bin"
    shard_path.write_bytes(expected_data)
    session, targets = _session_and_targets(shard_path, expected_data)
    shard_path.write_bytes(b"X" + expected_data[1:])

    def unexpected_transfer(**_kwargs):
        pytest.fail("direct transfer must not start after digest mismatch")

    monkeypatch.setattr(
        nixl_gds,
        "run_bounded_nixl_transfers",
        unexpected_transfer,
    )

    with pytest.raises(RuntimeError, match="BLAKE3 mismatch"):
        session.restore(targets)
