# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.common.snapshot.constants import (
    SNAPSHOT_CONTROL_DIR_ENV,
    SNAPSHOT_RESTORE_CONTEXT_FILE,
)
from dynamo.vllm.snapshot_nixl import (
    SNAPSHOT_NIXL_MODULE,
    SnapshotNixlMixin,
    configure_snapshot_nixl_connector,
    read_handshake_metadata,
    write_handshake_metadata,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_configure_snapshot_nixl_connector_direct_and_nested():
    direct = SimpleNamespace(
        kv_connector="NixlConnector", kv_connector_module_path=None
    )
    configure_snapshot_nixl_connector(direct)
    assert direct.kv_connector_module_path == SNAPSHOT_NIXL_MODULE

    nixl_child = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
    wrapped = SimpleNamespace(
        kv_connector="PdConnector",
        kv_connector_extra_config={
            "connectors": [
                {"kv_connector": "DynamoConnector"},
                nixl_child,
            ]
        },
    )
    configure_snapshot_nixl_connector(wrapped)
    assert nixl_child["kv_connector_module_path"] == SNAPSHOT_NIXL_MODULE


def test_handshake_round_trip(tmp_path):
    payload = {"engine_id": "inc-9", "agent": b"new-agent"}
    write_handshake_metadata("inc-9", 0, 0, payload, control_dir=str(tmp_path))
    assert (
        read_handshake_metadata("inc-9", control_dir=str(tmp_path))[(0, 0)] == payload
    )


class _FakeWorker:
    tp_rank = 0

    def __init__(self, vllm_config=None, engine_id="old", kv_cache_config=None):
        self.engine_id = engine_id
        self.xfer_handshake_metadata = {"engine_id": engine_id, "agent": b"meta"}
        self.registered = None

    def register_kv_caches(self, caches):
        self.registered = caches


class _FakeScheduler:
    def __init__(self, vllm_config=None, engine_id="old", kv_cache_config=None):
        self.handshake = None

    def set_xfer_handshake_metadata(self, metadata):
        self.handshake = metadata


class _FakeConnector(SnapshotNixlMixin):
    def __init__(self, role):
        self._bound_incarnation_id = None
        self._kv_caches = {"layer": object()}
        self.engine_id = "old-engine"
        self.connector_worker = _FakeWorker() if role == "worker" else None
        self.connector_scheduler = _FakeScheduler() if role == "scheduler" else None
        self.kv_cache_config = object()
        self.shut_down = False
        self._vllm_config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="old-engine"),
            parallel_config=SimpleNamespace(data_parallel_index=0),
        )

    def shutdown(self):
        self.shut_down = True


def test_worker_rebind_writes_new_engine_handshake(monkeypatch, tmp_path):
    monkeypatch.setenv(SNAPSHOT_CONTROL_DIR_ENV, str(tmp_path))
    (tmp_path / SNAPSHOT_RESTORE_CONTEXT_FILE).write_text(
        '{"incarnation_id":"inc-restore","env":{}}\n'
    )
    monkeypatch.setattr(
        "dynamo.vllm.snapshot_nixl.apply_snapshot_restore_env", lambda: {}
    )
    connector = _FakeConnector("worker")
    SnapshotNixlMixin._maybe_rebind(connector)

    assert connector.engine_id == "inc-restore_dp0"
    assert connector.connector_worker.registered is connector._kv_caches
    assert (
        read_handshake_metadata("inc-restore", control_dir=str(tmp_path))[(0, 0)][
            "engine_id"
        ]
        == "inc-restore_dp0"
    )
    SnapshotNixlMixin._maybe_rebind(connector)
    assert connector._bound_incarnation_id == "inc-restore"


def test_scheduler_rebind_reads_handshake(monkeypatch, tmp_path):
    monkeypatch.setenv(SNAPSHOT_CONTROL_DIR_ENV, str(tmp_path))
    (tmp_path / SNAPSHOT_RESTORE_CONTEXT_FILE).write_text(
        '{"incarnation_id":"inc-restore","env":{}}\n'
    )
    write_handshake_metadata(
        "inc-restore", 0, 0, {"engine_id": "inc-restore_dp0"}, control_dir=str(tmp_path)
    )
    monkeypatch.setattr(
        "dynamo.vllm.snapshot_nixl.apply_snapshot_restore_env", lambda: {}
    )
    connector = _FakeConnector("scheduler")
    SnapshotNixlMixin._maybe_rebind(connector)
    assert connector.connector_scheduler.handshake[(0, 0)]["engine_id"] == (
        "inc-restore_dp0"
    )
