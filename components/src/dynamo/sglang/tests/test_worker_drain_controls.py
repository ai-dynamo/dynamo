# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.sglang.drain_controls import configure_worker_drain

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture
def controls(monkeypatch):
    monkeypatch.setenv("DYN_EXPERIMENTAL_SGLANG_WORKER_DRAIN", "1")
    monkeypatch.setenv("DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS", "0")
    handler = SimpleNamespace(
        generate_endpoint=SimpleNamespace(
            unregister_endpoint_instance=AsyncMock(),
            register_endpoint_instance=AsyncMock(),
            connection_id=Mock(return_value=7),
            client=AsyncMock(
                return_value=SimpleNamespace(instance_ids=Mock(return_value=[7]))
            ),
        ),
        config=SimpleNamespace(dynamo_args=SimpleNamespace(engine_routes=[])),
        engine=SimpleNamespace(tokenizer_manager=SimpleNamespace(is_pause=False)),
        _engine_route_lock=asyncio.Lock(),
    )
    runtime = SimpleNamespace(
        system_status_server_url=Mock(return_value="system-listener"),
        register_engine_http_handler=Mock(),
    )
    ready, terminal = asyncio.Event(), asyncio.Event()
    ready.set()
    return handler, runtime, ready, terminal


def test_controls_are_opt_in_and_require_a_system_listener(controls, monkeypatch):
    handler, runtime, ready, terminal = controls
    monkeypatch.delenv("DYN_EXPERIMENTAL_SGLANG_WORKER_DRAIN")
    assert (
        configure_worker_drain(handler, runtime, ready, asyncio.Event(), terminal)
        is None
    )
    runtime.system_status_server_url.assert_not_called()
    runtime.register_engine_http_handler.assert_not_called()
    monkeypatch.setenv("DYN_EXPERIMENTAL_SGLANG_WORKER_DRAIN", "1")
    runtime.system_status_server_url.return_value = None
    with pytest.raises(ValueError, match="DYN_SYSTEM_PORT"):
        configure_worker_drain(handler, runtime, ready, asyncio.Event(), terminal)


@pytest.mark.asyncio
async def test_control_methods_and_withdrawal_failure(controls):
    handler, runtime, ready, terminal = controls
    drain = configure_worker_drain(handler, runtime, ready, asyncio.Event(), terminal)
    prefix, control = runtime.register_engine_http_handler.call_args.args
    assert prefix == "worker_lifecycle"
    assert handler._worker_drain is drain
    response = await control({"path": "drain", "method": "GET", "body": None})
    assert response["status"] == 405
    handler.generate_endpoint.unregister_endpoint_instance.assert_not_called()
    response = await control({"path": "drain", "method": "POST", "body": [1]})
    assert response["status"] == 400
    handler.generate_endpoint.unregister_endpoint_instance.side_effect = RuntimeError(
        "discovery unavailable"
    )
    response = await control({"path": "drain", "method": "POST", "body": {}})
    assert response["status"] == 503
    assert (await drain.status({}))["state"] == "serving"
    assert not drain.withdrawn


@pytest.mark.asyncio
async def test_status_tracks_background_work_without_authorizing_deletion(controls):
    handler, runtime, ready, terminal = controls
    pending = asyncio.create_task(asyncio.Event().wait())
    handler._consume_tasks = {pending}
    try:
        configure_worker_drain(handler, runtime, ready, asyncio.Event(), terminal)
        control = runtime.register_engine_http_handler.call_args.args[1]
        response = await control({"path": "drain", "method": "POST", "body": None})
        assert response["status"] == 200
        assert response["body"]["admin_withdrawn"]
        response = await control({"path": "status", "method": "GET", "body": None})
        status = response["body"]
        assert status["background_tasks"] == 1
        assert status["inflight_requests"] == 0
        assert status["engine_quiescent"] is None
        assert not status["completion_supported"]
        assert status["state"] == "draining"
        terminal.set()
        response = await control({"path": "resume", "method": "POST", "body": {}})
        assert response["status"] == 409
        handler.generate_endpoint.register_endpoint_instance.assert_not_called()
    finally:
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending


def test_controls_do_not_shadow_a_configured_engine_route(controls):
    handler, runtime, ready, terminal = controls
    handler.config.dynamo_args.engine_routes = [
        "worker_lifecycle/status=get_server_info"
    ]
    with pytest.raises(ValueError, match="reserved"):
        configure_worker_drain(handler, runtime, ready, asyncio.Event(), terminal)
    runtime.register_engine_http_handler.assert_not_called()


@pytest.mark.asyncio
async def test_model_ready_does_not_allow_drain_before_own_endpoint_registered(
    controls,
):
    handler, runtime, ready, terminal = controls
    client = handler.generate_endpoint.client.return_value
    client.instance_ids.return_value = [8]
    drain = configure_worker_drain(handler, runtime, ready, asyncio.Event(), terminal)
    control = runtime.register_engine_http_handler.call_args.args[1]
    response = await control({"path": "drain", "method": "POST", "body": {}})
    assert response["status"] == 409
    handler.generate_endpoint.unregister_endpoint_instance.assert_not_called()
    client.instance_ids.return_value = [7, 8]
    response = await control({"path": "drain", "method": "POST", "body": {}})
    assert response["status"] == 200
    assert drain.withdrawn
    client.instance_ids.return_value = [8]
    response = await control({"path": "status", "method": "GET", "body": None})
    assert response["body"]["state"] == "draining"
