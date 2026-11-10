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
    monkeypatch.setenv("DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS", "0")
    handler = SimpleNamespace(
        generate_endpoint=SimpleNamespace(
            unregister_endpoint_instance=AsyncMock(),
            register_endpoint_instance=AsyncMock(),
            inflight_requests=Mock(return_value=0),
            connection_id=Mock(return_value=7),
            client=AsyncMock(
                return_value=SimpleNamespace(instance_ids=Mock(return_value=[7]))
            ),
        ),
        config=SimpleNamespace(
            dynamo_args=SimpleNamespace(engine_routes=[], response_plane="tcp")
        ),
        engine=SimpleNamespace(
            tokenizer_manager=SimpleNamespace(is_pause=False, rid_to_state={})
        ),
        _engine_route_lock=asyncio.Lock(),
    )
    runtime = SimpleNamespace(
        system_status_server_url=Mock(return_value="system-listener"),
        register_engine_http_handler=Mock(),
    )
    ready, terminal = asyncio.Event(), asyncio.Event()
    ready.set()
    return handler, runtime, ready, terminal


async def call(runtime, action, method, body=None):
    handlers = dict(
        call.args for call in runtime.register_engine_http_handler.call_args_list
    )
    return await handlers[action]({"path": "", "method": method, "body": body})


@pytest.mark.asyncio
async def test_control_methods_and_withdrawal_failure(controls):
    handler, runtime, ready, terminal = controls
    drain = configure_worker_drain(handler, runtime, ready, asyncio.Event(), terminal)
    assert {
        entry.args[0] for entry in runtime.register_engine_http_handler.call_args_list
    } == {"drain", "resume", "status"}
    assert handler._worker_drain is drain
    response = await call(runtime, "drain", "GET", None)
    assert response["status"] == 405
    handler.generate_endpoint.unregister_endpoint_instance.assert_not_called()
    response = await call(runtime, "drain", "POST", [1])
    assert response["status"] == 400
    handler.generate_endpoint.unregister_endpoint_instance.side_effect = RuntimeError(
        "discovery unavailable"
    )
    response = await call(runtime, "drain", "POST", {})
    assert response["status"] == 503
    assert (await drain.status({}))["state"] == "serving"
    assert not drain.withdrawn


@pytest.mark.asyncio
async def test_status_waits_for_ingress_and_sglang_before_reporting_drained(controls):
    handler, runtime, ready, terminal = controls
    configure_worker_drain(handler, runtime, ready, asyncio.Event(), terminal)
    inflight = handler.generate_endpoint.inflight_requests
    rid_to_state = handler.engine.tokenizer_manager.rid_to_state
    inflight.return_value = 1
    response = await call(runtime, "drain", "POST", None)
    assert response["status"] == 202
    assert response["body"]["reason"] == "dynamo_inflight"

    inflight.return_value = 0
    rid_to_state["pending"] = object()
    response = await call(runtime, "status", "GET")
    assert response["body"]["reason"] == "sglang_inflight"

    del handler.engine.tokenizer_manager.rid_to_state
    response = await call(runtime, "status", "GET")
    assert response["body"]["reason"] == "completion_check_unavailable"

    rid_to_state.clear()
    handler.engine.tokenizer_manager.rid_to_state = rid_to_state
    response = await call(runtime, "status", "GET")
    assert response["body"] == {
        "worker_id": "7",
        "state": "drained",
        "reason": None,
    }
    terminal.set()
    response = await call(runtime, "resume", "POST", {})
    assert response["status"] == 409
    handler.generate_endpoint.register_endpoint_instance.assert_not_called()


def test_controls_do_not_shadow_a_configured_engine_route(controls):
    handler, runtime, ready, terminal = controls
    handler.config.dynamo_args.engine_routes = ["status/alias=get_server_info"]
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

    async def create_client():
        await asyncio.sleep(0)
        return client

    handler.generate_endpoint.client.side_effect = create_client
    drain = configure_worker_drain(handler, runtime, ready, asyncio.Event(), terminal)
    responses = await asyncio.gather(
        call(runtime, "drain", "POST", {}),
        call(runtime, "status", "GET", None),
    )
    assert responses[0]["status"] == 409
    assert responses[1]["body"]["state"] == "starting"
    handler.generate_endpoint.unregister_endpoint_instance.assert_not_called()
    # Repeated startup polls must reuse the same discovery monitor.
    await call(runtime, "status", "GET", None)
    handler.generate_endpoint.client.assert_awaited_once()
    client.instance_ids.return_value = [7, 8]
    handler.generate_endpoint.inflight_requests.return_value = 1
    response = await call(runtime, "drain", "POST", {})
    assert response["status"] == 202
    assert drain.withdrawn
    client.instance_ids.return_value = [8]
    response = await call(runtime, "status", "GET", None)
    assert response["body"]["state"] == "draining"


@pytest.mark.asyncio
async def test_quic_rejects_unpublish_without_changing_worker_state(controls):
    handler, runtime, ready, terminal = controls
    handler.config.dynamo_args.response_plane = "quic"
    drain = configure_worker_drain(handler, runtime, ready, asyncio.Event(), terminal)
    response = await call(runtime, "drain", "POST", {})
    assert response["status"] == 409
    assert "TCP response plane" in response["body"]["error"]
    handler.generate_endpoint.unregister_endpoint_instance.assert_not_called()
    handler.generate_endpoint.client.assert_not_called()
    assert not drain.withdrawn
    response = await call(runtime, "status", "GET", None)
    assert response["body"]["state"] == "serving"
    response = await call(runtime, "resume", "POST", {})
    assert response["status"] == 200


def test_unsupported_worker_does_not_expose_manual_drain(controls):
    handler, runtime, ready, terminal = controls
    configure_worker_drain(
        handler, runtime, ready, asyncio.Event(), terminal, enable_admin=False
    )
    runtime.register_engine_http_handler.assert_not_called()


@pytest.mark.asyncio
async def test_paused_worker_reports_state_before_first_status_poll(controls):
    handler, runtime, ready, terminal = controls
    handler.engine.tokenizer_manager.is_pause = True
    handler.generate_endpoint.client.return_value.instance_ids.return_value = []
    configure_worker_drain(handler, runtime, ready, asyncio.Event(), terminal)

    response = await call(runtime, "status", "GET")
    assert response["body"]["state"] == "paused"
    response = await call(runtime, "drain", "POST")
    assert response["status"] == 409
    assert "paused" in response["body"]["error"]
    handler.generate_endpoint.unregister_endpoint_instance.assert_not_awaited()
