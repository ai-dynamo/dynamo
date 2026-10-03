# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``LoraLifecycleChatPayload``.

Drive the load → unload → reload sequence against a fake worker system API and
frontend, without a GPU or a real worker. The fakes replace ``requests.get``,
``requests.post``, and ``requests.delete`` and serve ``/v1/loras`` and
``/v1/models`` from one shared "adapter loaded" flag.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from tests.utils.constants import DefaultPort
from tests.utils.payload_builder import lora_lifecycle_chat_payload

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]

LORA = "org/test-lora"
BASE = "org/base-model"
SYSTEM = f"http://localhost:{DefaultPort.SYSTEM1.value}"
FRONTEND = f"http://localhost:{DefaultPort.FRONTEND.value}"


def _response(status_code: int = 200, body: Any = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = str(body)
    resp.json.return_value = body
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(str(status_code))
    else:
        resp.raise_for_status.return_value = None
    return resp


class FakeDeployment:
    """One worker plus frontend whose state is a single loaded flag."""

    def __init__(self, unload_status: int = 200):
        self.loaded = False
        self.unload_status = unload_status
        self.calls: list[tuple[str, str, Any]] = []

    def get(self, url: str, timeout: float = 0) -> MagicMock:
        self.calls.append(("GET", url, None))
        if url == f"{SYSTEM}/v1/loras":
            return _response(body={"loras": {LORA: 1} if self.loaded else {}})
        if url == f"{FRONTEND}/v1/models":
            ids = [BASE, LORA] if self.loaded else [BASE]
            return _response(body={"data": [{"id": i} for i in ids]})
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url: str, json: Any = None, timeout: float = 0) -> MagicMock:
        self.calls.append(("POST", url, json))
        if url == f"{SYSTEM}/v1/loras":
            self.loaded = True
            return _response(body={"status": "success"})
        if url == f"{FRONTEND}/v1/chat/completions":
            if json["model"] == LORA and not self.loaded:
                return _response(404, {"message": "Model not found"})
            return _response(
                body={"choices": [{"message": {"content": "A neural network."}}]}
            )
        raise AssertionError(f"unexpected POST {url}")

    def delete(self, url: str, timeout: float = 0) -> MagicMock:
        self.calls.append(("DELETE", url, None))
        assert url == f"{SYSTEM}/v1/loras/{LORA}"
        if self.unload_status == 200:
            self.loaded = False
        return _response(self.unload_status, {"status": "error"})

    def chat_models(self) -> list[str]:
        return [
            body["model"]
            for method, url, body in self.calls
            if method == "POST" and url.endswith("/v1/chat/completions")
        ]


def _install(monkeypatch, deployment: FakeDeployment) -> None:
    monkeypatch.setattr(requests, "get", deployment.get)
    monkeypatch.setattr(requests, "post", deployment.post)
    monkeypatch.setattr(requests, "delete", deployment.delete)


def _payload():
    return lora_lifecycle_chat_payload(
        lora_name=LORA, s3_uri=f"s3://bucket/{LORA}", base_model=BASE
    )


def test_lifecycle_unloads_then_reloads_before_harness_request(monkeypatch):
    """url() runs adapter, unloaded-adapter, and base requests, then reloads."""
    deployment = FakeDeployment()
    _install(monkeypatch, deployment)

    url = _payload().url()

    assert url == f"{FRONTEND}/v1/chat/completions"
    assert deployment.loaded, "adapter must be reloaded for the harness request"
    assert [c[0] for c in deployment.calls].count("DELETE") == 1
    load_count = sum(
        1 for m, u, _ in deployment.calls if m == "POST" and u == f"{SYSTEM}/v1/loras"
    )
    assert load_count == 2
    assert deployment.chat_models() == [LORA, LORA, BASE]


def test_lifecycle_runs_once_across_repeats(monkeypatch):
    """Repeated url() calls do not unload or reload the adapter again."""
    deployment = FakeDeployment()
    _install(monkeypatch, deployment)
    payload = _payload()

    payload.url()
    calls_after_first = len(deployment.calls)
    payload.url()

    assert len(deployment.calls) == calls_after_first


def test_unload_failure_raises(monkeypatch):
    """A failed unload surfaces as an error instead of skipping the checks."""
    deployment = FakeDeployment(unload_status=500)
    _install(monkeypatch, deployment)

    with pytest.raises(RuntimeError, match="Failed to unload LoRA adapter"):
        _payload().url()
