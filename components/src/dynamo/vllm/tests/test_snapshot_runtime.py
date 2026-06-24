# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import types
from types import SimpleNamespace

import pytest

from dynamo.common.snapshot.constants import SNAPSHOT_CONTROL_DIR_ENV
from dynamo.vllm import snapshot
from dynamo.vllm.flashinfer_snapshot import (
    DYN_VLLM_REQUIRE_FLASHINFER_SNAPSHOT_RESOURCES,
)
from dynamo.vllm.snapshot import (
    prepare_snapshot_engine,
    verify_snapshot_worker_identity,
)
from dynamo.vllm.snapshot_worker_config import (
    DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER,
    SNAPSHOT_WORKER_CLASS,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _identity(**overrides):
    identity = {
        "module": "dynamo.vllm.snapshot_worker",
        "class": "SnapshotWorker",
        "qualified_class": SNAPSHOT_WORKER_CLASS,
        "rank": 0,
        "local_rank": 0,
        "pid": 1234,
        "flashinfer_resource_count": 1,
        "flashinfer_resources": [
            {
                "name": "get_ep_group().device_communicator.all2all_manager",
                "kind": "two_sided_manager",
                "class": "flashinfer.Manager",
            }
        ],
    }
    identity.update(overrides)
    return identity


class _FakeEngineClient:
    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc
        self.calls = []

    async def collective_rpc(self, method, timeout=None):
        self.calls.append((method, timeout))
        if self.exc is not None:
            raise self.exc
        return self.result


@pytest.mark.asyncio
async def test_verify_snapshot_worker_identity_calls_collective_rpc(monkeypatch):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    engine_client = _FakeEngineClient([_identity()])

    await verify_snapshot_worker_identity(engine_client)

    assert engine_client.calls == [("snapshot_worker_identity", 30)]


@pytest.mark.asyncio
async def test_verify_snapshot_worker_identity_fails_wrong_class(monkeypatch):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    engine_client = _FakeEngineClient(
        [_identity(qualified_class="vllm.v1.worker.gpu_worker.Worker")]
    )

    with pytest.raises(RuntimeError, match="expected"):
        await verify_snapshot_worker_identity(engine_client)


@pytest.mark.asyncio
async def test_verify_snapshot_worker_identity_fails_missing_method(monkeypatch):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    engine_client = _FakeEngineClient(exc=AttributeError("missing"))

    with pytest.raises(RuntimeError, match="could not verify"):
        await verify_snapshot_worker_identity(engine_client)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"flashinfer_resource_count": 0},
        {"flashinfer_resource_count": -1},
        {"flashinfer_resource_count": None},
        {"flashinfer_resource_count": "1"},
        {"flashinfer_resource_count": True},
        {"flashinfer_resource_count": False},
        {"flashinfer_resource_count": 1.5},
        {},
    ],
)
async def test_verify_snapshot_worker_identity_strict_invalid_resources(
    monkeypatch, overrides
):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    monkeypatch.setenv(DYN_VLLM_REQUIRE_FLASHINFER_SNAPSHOT_RESOURCES, "1")
    identity = _identity()
    if overrides:
        identity.update(overrides)
    else:
        identity.pop("flashinfer_resource_count")
    engine_client = _FakeEngineClient([identity])

    with pytest.raises(RuntimeError, match="invalid flashinfer_resource_count"):
        await verify_snapshot_worker_identity(engine_client)


@pytest.mark.asyncio
async def test_prepare_snapshot_engine_enables_sleep_without_enforce_eager(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(SNAPSHOT_CONTROL_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(snapshot, "configure_snapshot_capture_env", lambda: None)
    monkeypatch.setattr(
        snapshot,
        "configure_flashinfer_snapshot_worker",
        lambda _: False,
    )

    async def verify_snapshot_worker_identity(_engine_client):
        return None

    monkeypatch.setattr(
        snapshot,
        "verify_snapshot_worker_identity",
        verify_snapshot_worker_identity,
    )

    async def wait_for_restore(self):
        return True

    monkeypatch.setattr(
        snapshot.EngineSnapshotController,
        "wait_for_restore",
        wait_for_restore,
    )
    handlers_module = types.ModuleType("dynamo.vllm.handlers")
    handlers_module.VllmEnginePauseController = lambda engine_client: SimpleNamespace(
        engine_client=engine_client
    )
    monkeypatch.setitem(sys.modules, "dynamo.vllm.handlers", handlers_module)

    config = SimpleNamespace(
        headless=False,
        engine_args=SimpleNamespace(enable_sleep_mode=False, enforce_eager=False),
    )
    engine = (_FakeEngineClient([_identity()]), object())

    controller = await prepare_snapshot_engine(config, lambda _: engine)

    assert controller is not None
    assert config.engine_args.enable_sleep_mode is True
    assert config.engine_args.enforce_eager is False
