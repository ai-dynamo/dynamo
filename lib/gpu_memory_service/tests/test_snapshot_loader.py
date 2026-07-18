# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GMS snapshot loader CLI."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import pytest

try:
    from gpu_memory_service.cli.snapshot import loader
    from gpu_memory_service.common.snapshot_profile import (
        SNAPSHOT_PROFILE_ENV,
        SnapshotProfile,
        snapshot_profile_enabled,
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


def test_list_checkpoint_devices_requires_exact_visible_device_match(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "device-2").mkdir()
    (tmp_path / "device-0").mkdir()
    (tmp_path / "device-0-copy").mkdir()
    (tmp_path / "not-a-device").mkdir()
    (tmp_path / "device-1").write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(loader.cuda_utils, "list_devices", lambda: [0, 2])

    assert loader._list_checkpoint_devices(str(tmp_path)) == [0, 2]


@pytest.mark.parametrize(
    ("visible_devices", "checkpoint_dirs", "expected"),
    [
        ([0, 1], ["device-0"], "missing=1"),
        ([0], ["device-0", "device-1"], "extra=1"),
        ([7], [], "missing=7"),
        ([2], ["device-02"], "missing=2"),
    ],
)
def test_list_checkpoint_devices_rejects_mismatched_checkpoints(
    tmp_path,
    monkeypatch,
    visible_devices,
    checkpoint_dirs,
    expected,
):
    for dirname in checkpoint_dirs:
        (tmp_path / dirname).mkdir()
    monkeypatch.setattr(loader.cuda_utils, "list_devices", lambda: visible_devices)

    with pytest.raises(RuntimeError, match=expected):
        loader._list_checkpoint_devices(str(tmp_path))


def test_load_device_sets_cuda_context_before_storage_client(monkeypatch):
    calls = []
    monkeypatch.delenv(SNAPSHOT_PROFILE_ENV, raising=False)

    class FakeStorageClient:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def load_to_gms(self, input_dir, *, max_workers, clear_existing):
            calls.append(
                (
                    "load_to_gms",
                    {
                        "input_dir": input_dir,
                        "max_workers": max_workers,
                        "clear_existing": clear_existing,
                    },
                )
            )

    monkeypatch.setattr(loader, "get_socket_path", lambda device: f"/tmp/gms-{device}")
    monkeypatch.setattr(loader, "GMSStorageClient", FakeStorageClient)
    monkeypatch.setattr(
        loader.cuda_utils,
        "cuda_runtime_set_device",
        lambda device: calls.append(("set_device", device)),
    )

    loader._load_device(
        "/checkpoints/run/versions/1",
        3,
        16,
        "nixl",
        [],
        2,
    )

    assert calls[0] == ("set_device", 3)
    assert calls[1][0] == "init"
    assert calls[1][1]["socket_path"] == "/tmp/gms-3"
    assert calls[1][1]["device"] == 3
    assert calls[2] == (
        "load_to_gms",
        {
            "input_dir": "/checkpoints/run/versions/1/device-3",
            "max_workers": 16,
            "clear_existing": True,
        },
    )


def test_load_device_profiles_first_claimed_cuda_set_device_and_current_context(
    monkeypatch,
    caplog,
):
    calls = []

    class FakeStorageClient:
        def __init__(self, **_kwargs):
            calls.append("storage_client")

        def load_to_gms(self, *_args, **_kwargs):
            calls.append("load")

    monkeypatch.setenv(SNAPSHOT_PROFILE_ENV, "1")
    monkeypatch.setattr(loader, "get_socket_path", lambda device: f"/tmp/gms-{device}")
    monkeypatch.setattr(loader, "GMSStorageClient", FakeStorageClient)
    monkeypatch.setattr(
        loader.cuda_utils,
        "cuda_runtime_set_device",
        lambda device: calls.append(("cudaSetDevice", device)),
    )
    monkeypatch.setattr(
        loader.cuda_utils,
        "cuda_current_context",
        lambda: calls.append("cuCtxGetCurrent") or 17,
    )
    loader._reset_cuda_set_device_profile_state()

    with caplog.at_level(logging.INFO):
        loader._load_device(
            "/checkpoints/run/versions/1",
            3,
            16,
            "nixl",
            [],
            2,
        )

    assert calls[:3] == [
        ("cudaSetDevice", 3),
        "cuCtxGetCurrent",
        "storage_client",
    ]
    payloads = [
        json.loads(record.getMessage().removeprefix("GMS_SNAPSHOT_PROFILE "))
        for record in caplog.records
        if record.getMessage().startswith("GMS_SNAPSHOT_PROFILE ")
    ]
    phases = {payload["phase"]: payload for payload in payloads}
    assert phases["cuda_set_device"]["first_claimed_for_process_profile"]
    assert (
        phases["first_claimed_cuda_set_device"]["semantics"]
        == "bookkeeping_claim_before_unsynchronized_call"
    )
    assert phases["current_context_established"]["current_context"] == 17


def test_snapshot_profile_gating_and_event_schema(monkeypatch, caplog):
    monkeypatch.delenv(SNAPSHOT_PROFILE_ENV, raising=False)
    assert not snapshot_profile_enabled()
    disabled = SnapshotProfile("loader", logger=loader.logger)
    with disabled.phase("disabled"):
        pass
    assert "GMS_SNAPSHOT_PROFILE" not in caplog.text

    monkeypatch.setenv(SNAPSHOT_PROFILE_ENV, "1")
    assert snapshot_profile_enabled()
    profile = SnapshotProfile("loader", logger=loader.logger, device=3)
    with caplog.at_level(logging.INFO):
        with profile.phase("manifest_read", count=2, byte_count=17):
            pass

    message = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("GMS_SNAPSHOT_PROFILE ")
    )
    payload = json.loads(message.removeprefix("GMS_SNAPSHOT_PROFILE "))
    assert payload["event"] == "gms_snapshot_profile"
    assert payload["component"] == "loader"
    assert payload["phase"] == "manifest_read"
    assert payload["device"] == 3
    assert payload["count"] == 2
    assert payload["bytes"] == 17
    assert payload["wall_end_ns"] >= payload["wall_start_ns"]
    assert payload["duration_ns"] >= 0
    assert payload["cpu_duration_ns"] >= 0
    with pytest.raises(TypeError, match="JSON scalar"):
        profile.phase("invalid", unsupported=[])


def test_snapshot_profile_aggregate_accounting(caplog):
    profile = SnapshotProfile("loader", logger=loader.logger, enabled=True, device=0)
    profile.add_aggregate(
        "allocate_rpc",
        wall_start_ns=100,
        wall_end_ns=200,
        duration_ns=70,
        cpu_duration_ns=20,
        count=1,
        byte_count=4096,
    )
    profile.add_aggregate(
        "allocate_rpc",
        wall_start_ns=150,
        wall_end_ns=300,
        duration_ns=90,
        cpu_duration_ns=30,
        count=2,
        byte_count=8192,
    )

    with caplog.at_level(logging.INFO):
        profile.emit_aggregates()

    message = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("GMS_SNAPSHOT_PROFILE ")
    )
    payload = json.loads(message.removeprefix("GMS_SNAPSHOT_PROFILE "))
    assert payload["kind"] == "aggregate"
    assert payload["wall_start_ns"] == 100
    assert payload["wall_end_ns"] == 300
    assert payload["duration_ns"] == 160
    assert payload["cpu_duration_ns"] == 50
    assert payload["count"] == 3
    assert payload["bytes"] == 12288
    assert payload["duration_semantics"] == "cumulative"

    caplog.clear()
    with caplog.at_level(logging.INFO):
        profile.emit_aggregates()
    assert "GMS_SNAPSHOT_PROFILE" not in caplog.text


def test_snapshot_profile_preserves_exceptions_and_aggregates_concurrently(caplog):
    profile = SnapshotProfile("loader", logger=loader.logger, enabled=True)

    with pytest.raises(ValueError, match="boom"):
        with profile.phase("failing_phase"):
            raise ValueError("boom")

    def record() -> None:
        with profile.aggregate("concurrent", byte_count=4):
            pass

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: record(), range(40)))

    with caplog.at_level(logging.INFO):
        profile.emit_aggregates()

    payloads = [
        json.loads(record.getMessage().removeprefix("GMS_SNAPSHOT_PROFILE "))
        for record in caplog.records
        if record.getMessage().startswith("GMS_SNAPSHOT_PROFILE ")
    ]
    failing = next(
        payload for payload in payloads if payload["phase"] == "failing_phase"
    )
    aggregate = next(
        payload for payload in payloads if payload["phase"] == "concurrent"
    )
    assert failing["status"] == "error"
    assert failing["error_type"] == "ValueError"
    assert aggregate["count"] == 40
    assert aggregate["bytes"] == 160


def test_snapshot_profile_flushes_only_matching_session(caplog):
    profile = SnapshotProfile("server", logger=loader.logger, enabled=True)
    for session in ("session_1", "session_2"):
        with profile.aggregate("allocation", session=session):
            pass

    with caplog.at_level(logging.INFO):
        profile.emit_aggregates(session="session_1")
    first_payloads = [
        json.loads(record.getMessage().removeprefix("GMS_SNAPSHOT_PROFILE "))
        for record in caplog.records
        if record.getMessage().startswith("GMS_SNAPSHOT_PROFILE ")
    ]
    assert [payload["session"] for payload in first_payloads] == ["session_1"]

    caplog.clear()
    with caplog.at_level(logging.INFO):
        profile.emit_aggregates()
    remaining_payloads = [
        json.loads(record.getMessage().removeprefix("GMS_SNAPSHOT_PROFILE "))
        for record in caplog.records
        if record.getMessage().startswith("GMS_SNAPSHOT_PROFILE ")
    ]
    assert [payload["session"] for payload in remaining_payloads] == ["session_2"]
