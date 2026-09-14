# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TritonHealthCheckPayload and the handler's probe short-circuit.

Covers both halves of the canary contract:
* Producer: TritonHealthCheckPayload stamps HEALTH_CHECK_KEY on to_dict() and
  survives DYN_HEALTH_CHECK_PAYLOAD env overrides.
* Consumer: RequestHandler.generate() short-circuits health probes via
  is_probe() and answers with Server.ready() / Model.ready() rather than
  running Triton inference.
"""

import asyncio
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
import tritonserver

from dynamo.health_check import HEALTH_CHECK_KEY
from dynamo.triton import handlers
from dynamo.triton.health_check import TritonHealthCheckPayload

pytestmark = [
    pytest.mark.unit,
    pytest.mark.triton,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


# --- Test helpers ---


def _make_handler(
    *,
    server_ready: bool | Exception = True,
    model_ready: bool = True,
    model_name: str = "test-model",
    output_metadata: Optional[list[dict[str, str]]] = None,
) -> handlers.RequestHandler:
    """Build a RequestHandler over configurable MagicMocks.

    ``server_ready`` accepts a bool or an Exception; when an Exception is
    passed, ``Server.ready()`` raises it. ``model_ready`` accepts only a
    bool. ``async_infer`` is wired to fail loudly so accidental inference
    is caught immediately.
    """
    if output_metadata is None:
        output_metadata = [{"name": "OUT", "datatype": "FP32"}]

    def _readiness(value: bool | Exception) -> MagicMock:
        m = MagicMock()
        if isinstance(value, Exception):
            m.side_effect = value
        else:
            m.return_value = value
        return m

    server = MagicMock()
    server.ready = _readiness(server_ready)

    model = MagicMock()
    model.ready = _readiness(model_ready)
    model.name = model_name
    model.metadata = MagicMock(return_value={"outputs": output_metadata})
    model.async_infer = MagicMock(
        side_effect=AssertionError("async_infer must not be called")
    )

    return handlers.RequestHandler(server, model)


def _probe_request(model_name: str = "test-model") -> dict[str, Any]:
    """Minimal probe request carrying the canary marker."""
    return {HEALTH_CHECK_KEY: True, "model": model_name}


def _run_generate(
    handler: handlers.RequestHandler, request: dict[str, Any]
) -> list[dict[str, Any]]:
    async def _collect() -> list[dict[str, Any]]:
        return [response async for response in handler.generate(request)]

    return asyncio.run(_collect())


# --- Payload contract (producer) ---


def test_default_payload_shape():
    """default_payload carries the model name and NOT the canary marker
    (the marker is stamped by to_dict())."""
    payload = TritonHealthCheckPayload("classifier_v2")
    assert payload.default_payload == {"model": "classifier_v2"}
    assert HEALTH_CHECK_KEY not in payload.default_payload


def test_to_dict_stamps_marker():
    """to_dict() returns a payload carrying HEALTH_CHECK_KEY=True alongside model name."""
    result = TritonHealthCheckPayload("m").to_dict()
    assert result[HEALTH_CHECK_KEY] is True
    assert result["model"] == "m"


def test_to_dict_stamps_marker_over_env_override(monkeypatch):
    """DYN_HEALTH_CHECK_PAYLOAD override cannot strip the canary marker."""
    monkeypatch.setenv("DYN_HEALTH_CHECK_PAYLOAD", '{"custom_field": 42}')
    result = TritonHealthCheckPayload("m").to_dict()
    assert result[HEALTH_CHECK_KEY] is True
    assert result["custom_field"] == 42


# --- Probe short-circuit (consumer) ---


def test_probe_short_circuits_without_invoking_inference():
    """A probe request skips inference and yields an empty-tensor response."""
    handler = _make_handler()

    responses = _run_generate(handler, _probe_request())

    assert len(responses) == 1
    assert responses[0]["model"] == "test-model"
    assert responses[0]["tensors"] == []


def test_probe_raises_when_server_not_ready():
    """Server.ready() False surfaces as RuntimeError."""
    handler = _make_handler(server_ready=False)

    with pytest.raises(RuntimeError, match="server not ready"):
        _run_generate(handler, _probe_request())


def test_probe_raises_when_model_not_ready():
    """Model.ready() False surfaces as RuntimeError naming the model."""
    handler = _make_handler(model_ready=False, model_name="detector_v3")

    with pytest.raises(RuntimeError, match="model detector_v3 not ready"):
        _run_generate(handler, _probe_request("detector_v3"))


def test_probe_normalizes_triton_error_from_stopped_server():
    """Server.ready() raising TritonError is collapsed to RuntimeError."""
    stopped = tritonserver.TritonError("server has been stopped")
    handler = _make_handler(server_ready=stopped)

    with pytest.raises(RuntimeError, match="triton not ready:.*stopped"):
        _run_generate(handler, _probe_request())
