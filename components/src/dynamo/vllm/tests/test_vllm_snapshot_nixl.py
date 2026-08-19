# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace

import pytest

from dynamo.common.snapshot.constants import SNAPSHOT_CONTROL_DIR_ENV
from dynamo.common.snapshot.restore_context import RestoreIdentity
from dynamo.vllm.snapshot_nixl import (
    SNAPSHOT_NIXL_MODULE,
    SnapshotNixlMixin,
    configure_snapshot_nixl_connector,
    engine_id_for_restore,
    read_handshake_metadata,
    write_handshake_metadata,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_configure_snapshot_nixl_connector_direct():
    kv_cfg = SimpleNamespace(
        kv_connector="NixlConnector", kv_connector_module_path=None
    )
    assert configure_snapshot_nixl_connector(kv_cfg) is True
    assert kv_cfg.kv_connector == "NixlConnector"
    assert kv_cfg.kv_connector_module_path == SNAPSHOT_NIXL_MODULE


def test_configure_snapshot_nixl_connector_nested_pd():
    nixl_child = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
    kv_cfg = SimpleNamespace(
        kv_connector="PdConnector",
        kv_connector_extra_config={
            "connectors": [
                {"kv_connector": "DynamoConnector", "kv_role": "kv_both"},
                nixl_child,
            ]
        },
    )
    assert configure_snapshot_nixl_connector(kv_cfg) is True
    assert nixl_child["kv_connector_module_path"] == SNAPSHOT_NIXL_MODULE
    assert (
        "kv_connector_module_path"
        not in kv_cfg.kv_connector_extra_config["connectors"][0]
    )


def test_engine_id_for_restore_preserves_dp_suffix():
    assert engine_id_for_restore("old_dp1", "inc-9") == "inc-9_dp1"
    assert engine_id_for_restore("old", "inc-9") == "inc-9"


def test_handshake_round_trip(tmp_path):
    payload = {"engine_id": "inc-9", "agent": b"new-agent"}
    write_handshake_metadata("inc-9", 0, 0, payload, control_dir=str(tmp_path))
    found = read_handshake_metadata(
        "inc-9", 1, 1, control_dir=str(tmp_path), timeout=0.2
    )
    assert found[(0, 0)] == payload


class _FakeWorker:
    tp_rank = 0

    def __init__(self, vllm_config=None, engine_id="old", kv_cache_config=None):
        self.engine_id = engine_id
        self.xfer_handshake_metadata = {
            "engine_id": engine_id,
            "agent": b"agent-" + engine_id.encode(),
        }
        self.registered = None

    def register_kv_caches(self, caches):
        self.registered = caches

    def register_cross_layers_kv_caches(self, kv_cache):
        return None


class _FakeScheduler:
    def __init__(self, vllm_config=None, engine_id="old", kv_cache_config=None):
        self.engine_id = engine_id
        self.handshake = None

    def set_xfer_handshake_metadata(self, metadata):
        self.handshake = metadata


class _FakeConnector(SnapshotNixlMixin):
    def __init__(self, role):
        self._bound_incarnation_id = None
        self._registered_kv_caches = {"layer": object()}
        self._registered_cross_layer_kv = None
        self.engine_id = "old-engine_dp0"
        self.connector_worker = _FakeWorker() if role == "worker" else None
        self.connector_scheduler = _FakeScheduler() if role == "scheduler" else None
        self.kv_cache_config = object()
        self.shut_down = False
        self._vllm_config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="old-engine_dp0"),
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
                rank=0,
                pipeline_parallel_rank=0,
            ),
        )

    def shutdown(self):
        self.shut_down = True


def test_worker_rebind_changes_engine_id_and_handshake(monkeypatch, tmp_path):
    monkeypatch.setenv(SNAPSHOT_CONTROL_DIR_ENV, str(tmp_path))
    identity = RestoreIdentity(
        incarnation_id="inc-restore",
        env={"POD_IP": "10.0.0.5", "VLLM_NIXL_SIDE_CHANNEL_HOST": "10.0.0.5"},
    )
    connector = _FakeConnector("worker")
    monkeypatch.setattr(
        "dynamo.vllm.snapshot_nixl.load_restore_identity", lambda: identity
    )

    SnapshotNixlMixin._maybe_rebind(connector)

    assert connector.shut_down is True
    assert connector.engine_id == "inc-restore_dp0"
    assert connector.connector_worker.engine_id == "inc-restore_dp0"
    assert connector.connector_worker.registered is connector._registered_kv_caches
    assert os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "10.0.0.5"
    found = read_handshake_metadata(
        "inc-restore", 1, 1, control_dir=str(tmp_path), timeout=0.2
    )
    assert found[(0, 0)]["engine_id"] == "inc-restore_dp0"

    SnapshotNixlMixin._maybe_rebind(connector)
    assert connector._bound_incarnation_id == "inc-restore"


def test_scheduler_rebind_reads_handshake_and_restarts_listener(monkeypatch, tmp_path):
    monkeypatch.setenv(SNAPSHOT_CONTROL_DIR_ENV, str(tmp_path))
    identity = RestoreIdentity(incarnation_id="inc-restore", env={"POD_IP": "10.0.0.5"})
    write_handshake_metadata(
        "inc-restore",
        0,
        0,
        {"engine_id": "inc-restore", "agent": b"new-agent"},
        control_dir=str(tmp_path),
    )
    connector = _FakeConnector("scheduler")
    monkeypatch.setattr(
        "dynamo.vllm.snapshot_nixl.load_restore_identity", lambda: identity
    )

    SnapshotNixlMixin._maybe_rebind(connector)

    assert connector.shut_down is True
    assert connector.engine_id == "inc-restore_dp0"
    assert connector.connector_scheduler.handshake[(0, 0)]["engine_id"] == "inc-restore"
