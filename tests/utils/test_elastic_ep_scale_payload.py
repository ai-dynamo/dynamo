# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``ElasticEPScalePayload``.

Exercise the scale-control POST + chat-url handoff without a GPU or a real
worker. We monkeypatch ``requests.post`` to mimic the
``/engine/control/scale_elastic_ep`` response and assert the payload POSTs the
right body to the right route, scales exactly once across repeats, and then
yields the chat-completions URL on the frontend port.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.utils import payloads as payloads_mod
from tests.utils.payloads import ElasticEPScalePayload

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


def _make_payload(*, new_dp: int = 4, system_port: int = 8081) -> ElasticEPScalePayload:
    return ElasticEPScalePayload(
        body={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
        },
        new_data_parallel_size=new_dp,
        system_port=system_port,
        expected_response=["hello"],
    )


@pytest.fixture
def capture_post(monkeypatch):
    """Install a fake ``requests.post`` that echoes a successful scale response
    and records each call."""
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, json: Any = None, timeout: float = 0) -> Any:
        calls.append({"url": url, "json": json})
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "status": "ok",
            "message": f"Scaled to data_parallel_size={json['new_data_parallel_size']}",
            "new_data_parallel_size": json["new_data_parallel_size"],
        }
        return resp

    monkeypatch.setattr(payloads_mod.requests, "post", fake_post)
    return calls


def test_url_triggers_scale_then_returns_chat_url(capture_post):
    """url() POSTs to the control route on the system port, then returns the
    chat-completions URL on the frontend port."""
    payload = _make_payload(new_dp=4, system_port=8081)

    url = payload.url()

    assert len(capture_post) == 1
    assert (
        capture_post[0]["url"]
        == "http://localhost:8081/engine/control/scale_elastic_ep"
    )
    assert capture_post[0]["json"] == {"new_data_parallel_size": 4}
    assert url.endswith("/v1/chat/completions")


def test_scale_runs_once_across_repeats(capture_post):
    """The _scaled guard means repeated url() calls scale only once."""
    payload = _make_payload(new_dp=4)

    payload.url()
    payload.url()
    payload.url()

    assert len(capture_post) == 1


def test_scale_error_status_raises(monkeypatch):
    """A non-ok scale response surfaces as an assertion, not a silent pass."""

    def fake_post(url: str, json: Any = None, timeout: float = 0) -> Any:
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"status": "error", "message": "boom"}
        return resp

    monkeypatch.setattr(payloads_mod.requests, "post", fake_post)
    payload = _make_payload()

    with pytest.raises(AssertionError, match="scale_elastic_ep failed"):
        payload.url()
