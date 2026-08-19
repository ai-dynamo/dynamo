# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import pytest
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    TransferBackend,
)

from dynamo.common.snapshot.restore_context import RestoreIdentity
from dynamo.sglang.snapshot_nixl import (
    SnapshotNixlKVManager,
    SnapshotNixlKVReceiver,
    install_snapshot_nixl,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


class _DummyAgent:
    def __init__(self, name):
        self.name = name

    def get_agent_metadata(self):
        return f"meta-{self.name}".encode()


def test_install_snapshot_nixl_returns_snapshot_classes():
    install_snapshot_nixl()
    from sglang.srt.disaggregation import utils as disagg_utils

    assert (
        disagg_utils.get_kv_class(TransferBackend.NIXL, KVClassType.MANAGER)
        is SnapshotNixlKVManager
    )
    assert (
        disagg_utils.get_kv_class(TransferBackend.NIXL, KVClassType.RECEIVER)
        is SnapshotNixlKVReceiver
    )
    install_snapshot_nixl()
    assert getattr(disagg_utils.get_kv_class, "_dyn_snapshot_nixl") is True


def test_maybe_rebind_mints_new_agent_name(monkeypatch):
    created = []

    def fake_agent(name, _config):
        agent = _DummyAgent(name)
        created.append(agent)
        return agent

    monkeypatch.setattr(
        "dynamo.sglang.snapshot_nixl.envs.SGLANG_DISAGGREGATION_NIXL_BACKEND",
        SimpleNamespace(get=lambda: "UCX"),
    )
    fake_nixl = SimpleNamespace(
        nixl_agent=fake_agent, nixl_agent_config=lambda **_kw: {}
    )
    monkeypatch.setitem(sys.modules, "nixl._api", fake_nixl)

    identity = RestoreIdentity(incarnation_id="inc-restore", env={"POD_IP": "10.0.0.5"})
    monkeypatch.setattr(
        "dynamo.sglang.snapshot_nixl.load_restore_identity", lambda: identity
    )

    manager = SnapshotNixlKVManager.__new__(SnapshotNixlKVManager)
    manager.agent = _DummyAgent("old-agent")
    manager.local_ip = "10.0.0.5"
    manager._bound_incarnation_id = None
    manager.registered = False
    manager.disaggregation_mode = DisaggregationMode.DECODE

    def register():
        manager.registered = True

    manager.register_buffer_to_engine = register

    SnapshotNixlKVManager._maybe_rebind(manager)

    assert manager.registered is True
    assert manager.agent.name != "old-agent"
    assert manager._bound_incarnation_id == "inc-restore"
    assert created and created[0].name == manager.agent.name

    SnapshotNixlKVManager._maybe_rebind(manager)
    assert len(created) == 1
