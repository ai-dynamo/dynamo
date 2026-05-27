# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NIXL GDS restore scheduling."""

import threading

import pytest

try:
    from gpu_memory_service.snapshot.backends import nixl_gds
    from gpu_memory_service.snapshot.backends.nixl_common import NixlTransferResources
    from gpu_memory_service.snapshot.backends.nixl_gds import _NixlGDSTransferSession
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


def test_gds_streaming_starts_ready_file_before_all_targets(monkeypatch):
    sources = [
        FileTransferSource(
            allocation_id="alloc-a",
            file_path="/checkpoint/shard-a.bin",
            file_offset=0,
            byte_count=4096,
        ),
        FileTransferSource(
            allocation_id="alloc-b",
            file_path="/checkpoint/shard-b.bin",
            file_offset=0,
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
    cuda_context_calls = []
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
        if label.endswith("shard-a.bin"):
            first_started.set()

    monkeypatch.setattr(nixl_gds, "start_transfer", fake_start_transfer)
    monkeypatch.setattr(
        nixl_gds.cuda_utils,
        "cuda_runtime_set_device",
        lambda device: cuda_context_calls.append(
            (device, threading.current_thread().name)
        ),
    )
    monkeypatch.setattr(
        nixl_gds,
        "wait_for_transfer_done",
        lambda *_args, **_kwargs: None,
    )

    session = _NixlGDSTransferSession(
        agent=FakeAgent(),
        agent_name="fake-agent",
        device=0,
        max_workers=1,
        sources=sources,
    )
    next_handle = iter(["handle-a", "handle-b"])

    def fake_prepare_file_transfer(file_group, _targets):
        assert cuda_context_calls == [(0, "nixl-gds-streaming-scheduler")]
        file_path, _sources = file_group
        return NixlTransferResources(handle=next(next_handle), label=file_path)

    monkeypatch.setattr(session, "_prepare_file_transfer", fake_prepare_file_transfer)

    try:
        session.submit_targets({"alloc-a": targets["alloc-a"]})
        assert first_started.wait(timeout=1.0)
        assert [label for _handle, label in started] == ["/checkpoint/shard-a.bin"]

        session.submit_targets({"alloc-b": targets["alloc-b"]})
        session.finish_restore()
        assert [label for _handle, label in started] == [
            "/checkpoint/shard-a.bin",
            "/checkpoint/shard-b.bin",
        ]
        assert cuda_context_calls == [(0, "nixl-gds-streaming-scheduler")]
    finally:
        session.close()
