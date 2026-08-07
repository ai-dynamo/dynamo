# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import dynamo.vllm.snapshot as snapshot_mod
from dynamo.vllm.args import parse_args
from dynamo.vllm.worker_factory import WorkerFactory

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.fault_tolerance,
    pytest.mark.gpu_1,
    pytest.mark.xpu_1,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.timeout(180),
    pytest.mark.pre_merge,
]

_BASE_ARGS = [
    "--model",
    "Qwen/Qwen3-0.6B",
    "--load-format",
    "gms",
    "--request-plane",
    "tcp",
]


@pytest.mark.asyncio
async def test_paused_restore_rejects_resumed_engine():
    factory = WorkerFactory(Mock(), Mock(), AsyncMock(), Mock(), Mock())

    with pytest.raises(RuntimeError, match="snapshot restore engine is not paused"):
        await factory._wake_with_failover_lock(
            Mock(is_paused=False),
            Mock(),
            SimpleNamespace(gms_shadow_mode=True),
            restore_paused=True,
        )


@pytest.mark.asyncio
async def test_restore_refresh_sets_shadow_mode_before_election(monkeypatch):
    config = parse_args(_BASE_ARGS)
    assert config.gms_shadow_mode is False
    monkeypatch.setenv("DYN_VLLM_GMS_SHADOW_MODE", "1")
    monkeypatch.setattr(
        snapshot_mod,
        "refresh_snapshot_restore_config",
        AsyncMock(return_value=config),
    )

    config = await snapshot_mod.refresh_vllm_snapshot_restore_config(
        config,
        _BASE_ARGS,
        restore_paused=True,
    )

    election = AsyncMock(return_value=(False, Mock()))
    factory = WorkerFactory(Mock(), Mock(), AsyncMock(), Mock(), Mock())
    factory._wake_with_failover_lock = election  # type: ignore[assignment]
    await factory._wake_with_failover_lock(Mock(), Mock(), config)
    election.assert_awaited_once()
    assert election.await_args.args[2].gms_shadow_mode is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "create_method"),
    [
        ("decode", "_create_decode_worker"),
        ("prefill", "_create_prefill_worker"),
    ],
)
async def test_paused_restore_reuses_pause_controller_without_repausing(
    monkeypatch, mode, create_method
):
    entered = asyncio.Event()
    release = asyncio.Event()
    election_call = {}

    async def blocked_election(*args, **kwargs):
        election_call["args"] = args
        election_call["kwargs"] = kwargs
        entered.set()
        await release.wait()
        return True, Mock()

    engine_config = SimpleNamespace(
        additional_config={},
        cache_config=SimpleNamespace(num_gpu_blocks=1),
        model_config=SimpleNamespace(max_model_len=1024),
        shutdown_timeout=5.0,
    )
    controller = SimpleNamespace(
        engine=(Mock(), engine_config, Mock(), "/tmp/prom", Mock()),
        pause_controller=Mock(is_paused=True, needs_resume_recovery=True),
        snapshot_config=Mock(restore_paused=True),
    )
    kv_setup = AsyncMock()
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.configure_kv_event_block_size", kv_setup
    )
    monkeypatch.setattr(
        "dynamo.vllm.worker_factory.get_dp_range_for_worker", lambda _config: (0, 1)
    )
    kv_publisher, fpm_relay, routes = Mock(), Mock(), Mock()
    model_registration, model_discovery = (
        AsyncMock(),
        AsyncMock(side_effect=RuntimeError("stop-after-election")),
    )
    factory = WorkerFactory(Mock(), kv_publisher, model_registration, fpm_relay, Mock())
    factory._wake_with_failover_lock = blocked_election  # type: ignore[assignment]
    factory._maybe_create_failover_metrics = Mock()  # type: ignore[assignment]
    factory._maybe_get_encode_worker_client = model_discovery  # type: ignore[assignment]
    factory.register_engine_routes = routes  # type: ignore[assignment]
    endpoint = Mock(connection_id=Mock(return_value="cid"), serve_endpoint=Mock())
    runtime = Mock(endpoint=Mock(return_value=endpoint))
    pause_controller = controller.pause_controller
    pause_controller.pause = AsyncMock()
    args = [*_BASE_ARGS, "--gms-shadow-mode"]
    if mode == "prefill":
        args.extend(["--disaggregation-mode", "prefill"])
    config = parse_args(args)
    task = asyncio.create_task(
        getattr(factory, create_method)(
            runtime, config, asyncio.Event(), [], snapshot_controller=controller
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    assert election_call["args"][0] is pause_controller
    assert election_call["kwargs"]["restore_paused"] is True
    pause_controller.pause.assert_not_awaited()
    blocked = (
        kv_setup,
        model_discovery,
        kv_publisher,
        fpm_relay,
        routes,
        model_registration,
        endpoint.serve_endpoint,
    )
    for publisher in blocked:
        publisher.assert_not_called()
    release.set()
    with pytest.raises(RuntimeError, match="stop-after-election"):
        await task
    model_discovery.assert_awaited_once()
