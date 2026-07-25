# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from types import SimpleNamespace

import pytest

import dynamo.vllm.snapshot as snapshot_mod

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.asyncio
async def test_snapshot_capture_exits_without_unwind_and_restore_returns(
    monkeypatch, caplog
):
    class SnapshotCaptured(Exception):
        pass

    snapshot_config = object()
    engine = (object(),)
    restore_results = iter((False, True))
    controllers = []
    exit_codes = []

    class FakeSnapshotController:
        def __init__(self, engine, pause_controller, snapshot_config, pause_args):
            self.engine = engine
            self.pause_controller = pause_controller
            self.snapshot_config = snapshot_config
            self.pause_args = pause_args
            controllers.append(self)

        async def wait_for_restore(self):
            return next(restore_results)

    def fake_exit(code):
        exit_codes.append(code)
        raise SnapshotCaptured

    monkeypatch.setattr(
        snapshot_mod.SnapshotConfig, "from_env", lambda: snapshot_config
    )
    monkeypatch.setattr(snapshot_mod, "configure_snapshot_capture_env", lambda: None)
    monkeypatch.setattr(
        snapshot_mod, "VllmEnginePauseController", lambda engine_client: engine_client
    )
    monkeypatch.setattr(
        snapshot_mod, "EngineSnapshotController", FakeSnapshotController
    )
    monkeypatch.setattr(snapshot_mod.gc, "collect", lambda: None)
    monkeypatch.setattr(snapshot_mod.os, "_exit", fake_exit)
    caplog.set_level(logging.INFO)

    config = SimpleNamespace(
        headless=False,
        engine_args=SimpleNamespace(enable_sleep_mode=False),
    )

    with pytest.raises(SnapshotCaptured):
        await snapshot_mod.prepare_snapshot_engine(config, lambda _: engine)

    assert exit_codes == [0]
    assert "without destroying the engine" in caplog.text

    restored_controller = await snapshot_mod.prepare_snapshot_engine(
        config, lambda _: engine
    )

    assert restored_controller is controllers[1]
    assert exit_codes == [0]
