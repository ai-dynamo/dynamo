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

from dynamo.sglang.snapshot_nixl import SnapshotNixlKVManager, install_snapshot_nixl

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


class _DummyAgent:
    def __init__(self, name):
        self.name = name


def test_install_snapshot_nixl_returns_manager():
    install_snapshot_nixl()
    from sglang.srt.disaggregation import utils as disagg_utils

    assert (
        disagg_utils.get_kv_class(TransferBackend.NIXL, KVClassType.MANAGER)
        is SnapshotNixlKVManager
    )
    install_snapshot_nixl()
    assert getattr(disagg_utils.get_kv_class, "_dyn_snapshot_nixl") is True


def test_maybe_rebind_mints_new_agent_name(monkeypatch, tmp_path):
    created = []

    def fake_agent(name, _config):
        agent = _DummyAgent(name)
        created.append(agent)
        return agent

    monkeypatch.setenv("DYN_SNAPSHOT_CONTROL_DIR", str(tmp_path))
    (tmp_path / "restore-context.json").write_text(
        '{"incarnation_id":"inc-restore","env":{}}\n'
    )
    monkeypatch.setattr(
        "dynamo.sglang.snapshot_nixl.envs.SGLANG_DISAGGREGATION_NIXL_BACKEND",
        SimpleNamespace(get=lambda: "UCX"),
    )
    monkeypatch.setitem(
        sys.modules,
        "nixl._api",
        SimpleNamespace(nixl_agent=fake_agent, nixl_agent_config=lambda **_kw: {}),
    )

    manager = SnapshotNixlKVManager.__new__(SnapshotNixlKVManager)
    manager._nixl_agent = _DummyAgent("old-agent")
    manager._bound_incarnation_id = None
    manager.registered = False
    manager.disaggregation_mode = DisaggregationMode.DECODE
    manager.register_buffer_to_engine = lambda: setattr(manager, "registered", True)

    SnapshotNixlKVManager._maybe_rebind(manager)

    assert manager.registered is True
    assert manager.agent.name != "old-agent"
    assert manager._bound_incarnation_id == "inc-restore"
    SnapshotNixlKVManager._maybe_rebind(manager)
    assert len(created) == 1
