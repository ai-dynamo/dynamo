# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NIXL UCX restore planning and streaming."""

import base64
import threading

import pytest

try:
    from gpu_memory_service.snapshot.backends import nixl_ucx
    from gpu_memory_service.snapshot.backends.nixl_common import NixlTransferResources
    from gpu_memory_service.snapshot.backends.nixl_ucx import _NixlUCXTransferSession
    from gpu_memory_service.snapshot.model import AllocationEntry
    from gpu_memory_service.snapshot.transfer import (
        GMSTransferTarget,
        RemoteTransferSource,
        build_remote_transfer_sources,
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


def test_build_remote_transfer_sources_validates_manifest_sizes():
    allocations = [
        AllocationEntry(
            allocation_id="alloc-a",
            size=4096,
            aligned_size=4096,
            tag="weight",
            tensor_file="unused",
        )
    ]

    sources = build_remote_transfer_sources(
        allocations,
        {
            "alloc-a": {
                "va": "8192",
                "device": 3,
                "byte_count": 4096,
            }
        },
        remote_agent="peer",
    )

    assert sources == [
        RemoteTransferSource(
            allocation_id="alloc-a",
            remote_agent="peer",
            va=8192,
            device=3,
            byte_count=4096,
        )
    ]

    with pytest.raises(RuntimeError, match="source size mismatch"):
        build_remote_transfer_sources(
            allocations,
            {"alloc-a": {"va": 8192, "byte_count": 1}},
            remote_agent="peer",
        )


def test_load_remote_peer_metadata_accepts_base64_metadata():
    raw_metadata = b"peer-metadata"
    payload = nixl_ucx.load_remote_peer_metadata(
        {
            "agent_name": "peer",
            "metadata": base64.b64encode(raw_metadata).decode("ascii"),
            "sources": {"alloc-a": {"va": 1, "byte_count": 2}},
        }
    )

    assert payload["agent_name"] == "peer"
    assert payload["metadata"] == raw_metadata
    assert payload["sources"] == {"alloc-a": {"va": 1, "byte_count": 2}}


def test_ucx_streaming_starts_ready_allocation_before_all_targets(monkeypatch):
    sources = [
        RemoteTransferSource(
            allocation_id="alloc-a",
            remote_agent="peer",
            va=0xA000,
            device=0,
            byte_count=4096,
        ),
        RemoteTransferSource(
            allocation_id="alloc-b",
            remote_agent="peer",
            va=0xB000,
            device=0,
            byte_count=4096,
        ),
    ]
    targets = {
        source.allocation_id: GMSTransferTarget(
            allocation_id=source.allocation_id,
            va=0x1000 + index * 0x1000,
            device=0,
            byte_count=source.byte_count,
        )
        for index, source in enumerate(sources)
    }

    started = []
    first_started = threading.Event()

    class FakeAgent:
        def check_xfer_state(self, handle):
            return "DONE"

        def release_xfer_handle(self, _handle):
            return None

        def deregister_memory(self, _registration):
            return None

    def fake_start_transfer(_agent, handle, label, _backend_name):
        started.append((handle, label))
        if label == "alloc-a":
            first_started.set()

    monkeypatch.setattr(nixl_ucx, "start_transfer", fake_start_transfer)
    monkeypatch.setattr(
        nixl_ucx,
        "wait_for_transfer_done",
        lambda *_args, **_kwargs: None,
    )

    session = _NixlUCXTransferSession(
        agent=FakeAgent(),
        remote_agent="peer",
        device=0,
        max_workers=1,
        sources=sources,
    )
    next_handle = iter(["handle-a", "handle-b"])

    def fake_prepare_source_transfer(source, _target):
        return NixlTransferResources(
            handle=next(next_handle),
            label=source.allocation_id,
        )

    monkeypatch.setattr(session, "_prepare_source_transfer", fake_prepare_source_transfer)

    try:
        session.submit_targets({"alloc-a": targets["alloc-a"]})
        assert first_started.wait(timeout=1.0)
        assert started == [("handle-a", "alloc-a")]

        session.submit_targets({"alloc-b": targets["alloc-b"]})
        session.finish_restore()
        assert started == [
            ("handle-a", "alloc-a"),
            ("handle-b", "alloc-b"),
        ]
    finally:
        session.close()
