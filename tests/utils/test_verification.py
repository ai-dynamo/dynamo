# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the deployment-free payload runner."""

import json

import pytest

from tests.utils.payloads import ChatPayload
from tests.utils.verification import (
    EngineLogError,
    EngineResponseError,
    ResponseValidationError,
    check_response,
    run_payloads,
    validate_expected_logs,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
]


class _FakeResponse:
    def __init__(self, payload=None, status_code=200, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.headers = {"content-type": "application/json"}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"status {self.status_code}")


class _FakeLogSource:
    def __init__(self, content: str) -> None:
        self._content = content
        self.log_path = "/tmp/fake.log"

    def read_logs(self) -> str:
        return self._content


def _chat_response(content="a fairly long answer about nothing at all", model="m"):
    return _FakeResponse(
        {
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": content}}],
        }
    )


def _chat_payload(**kwargs):
    kwargs.setdefault("body", {"messages": []})
    kwargs.setdefault("expected_response", [])
    kwargs.setdefault("expected_log", [])
    return ChatPayload(**kwargs)


# --- check_response ----------------------------------------------------------


def test_non_200_raises_engine_response_error():
    payload = _chat_payload()
    response = _FakeResponse({"error": "boom"}, status_code=500)
    with pytest.raises(EngineResponseError, match="500"):
        check_response(payload, response)


def test_content_assertion_failure_is_a_validation_error():
    """Only ResponseValidationError is retried, so the class matters."""
    payload = _chat_payload(expected_response=["unicorn"])
    with pytest.raises(ResponseValidationError, match="unicorn"):
        check_response(payload, _chat_response())


def test_min_content_length_is_enforced():
    payload = _chat_payload(min_content_length=100)
    with pytest.raises(ResponseValidationError, match="too short"):
        check_response(payload, _chat_response(content="short"))


def test_expected_model_mismatch_is_reported():
    payload = _chat_payload(expected_model="wanted")
    with pytest.raises(ResponseValidationError, match="Expected model 'wanted'"):
        check_response(payload, _chat_response(model="served"))


def test_require_content_field_rejects_a_refusal_that_is_long_enough():
    """extract_content falls back to `refusal`, so length alone is not enough."""
    payload = _chat_payload(min_content_length=10, require_content_field=True)
    response = _FakeResponse(
        {
            "model": "m",
            "choices": [
                {"message": {"role": "assistant", "refusal": "I cannot help with that"}}
            ],
        }
    )
    with pytest.raises(ResponseValidationError, match="missing 'content'"):
        check_response(payload, response)


def test_require_content_field_measures_message_content_not_the_fallback():
    payload = _chat_payload(min_content_length=100, require_content_field=True)
    response = _FakeResponse(
        {
            "model": "m",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "short",
                        "reasoning_content": "x" * 500,
                    }
                }
            ],
        }
    )
    with pytest.raises(ResponseValidationError, match="content"):
        check_response(payload, response)


def test_require_content_field_accepts_real_content():
    payload = _chat_payload(min_content_length=10, require_content_field=True)
    check_response(payload, _chat_response())


def test_require_content_field_rejects_empty_content_even_at_length_zero():
    """min_content_length=0 means "any real answer", not "empty counts"."""
    payload = _chat_payload(min_content_length=0, require_content_field=True)
    response = _FakeResponse(
        {
            "model": "m",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "thinking out loud",
                    }
                }
            ],
        }
    )
    with pytest.raises(ResponseValidationError, match="'content' is empty"):
        check_response(payload, response)


def test_expected_role_mismatch_is_reported():
    payload = _chat_payload(expected_role="assistant")
    response = _FakeResponse(
        {"model": "m", "choices": [{"message": {"role": "tool", "content": "hi"}}]}
    )
    with pytest.raises(ResponseValidationError, match="Expected role 'assistant'"):
        check_response(payload, response)


def test_expected_log_uses_the_supplied_log_source(monkeypatch):
    monkeypatch.setattr("tests.utils.verification.time.sleep", lambda _s: None)
    payload = _chat_payload(expected_log=[r"KV hit rate: \d+"])
    check_response(
        payload, _chat_response(), log_source=_FakeLogSource("KV hit rate: 42")
    )


def test_missing_log_pattern_raises(monkeypatch):
    monkeypatch.setattr("tests.utils.verification.time.sleep", lambda _s: None)
    payload = _chat_payload(expected_log=["never appears"])
    with pytest.raises(EngineLogError, match="never appears"):
        check_response(payload, _chat_response(), log_source=_FakeLogSource("nothing"))


def test_expected_log_without_a_log_source_says_the_test_is_deployment_coupled():
    payload = _chat_payload(expected_log=["anything"])
    with pytest.raises(EngineLogError, match="topology_dependent"):
        check_response(payload, _chat_response())


def test_empty_log_names_the_path():
    with pytest.raises(EngineLogError, match="/tmp/fake.log"):
        validate_expected_logs(["x"], _FakeLogSource(""))


# --- run_payloads ------------------------------------------------------------


def test_retries_validation_failures_up_to_max_attempts(monkeypatch):
    monkeypatch.setattr("tests.utils.verification.time.sleep", lambda _s: None)
    responses = [
        _chat_response(content="wrong"),
        _chat_response(content="wrong"),
        _chat_response(content="contains unicorn here"),
    ]
    sent = []

    def fake_send_request(**kwargs):
        sent.append(kwargs["url"])
        return responses[len(sent) - 1]

    monkeypatch.setattr("tests.utils.verification.send_request", fake_send_request)

    run_payloads(
        [_chat_payload(expected_response=["unicorn"], max_attempts=3, port=1234)]
    )
    assert len(sent) == 3
    assert sent[0] == "http://localhost:1234/v1/chat/completions"


def test_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr("tests.utils.verification.time.sleep", lambda _s: None)
    monkeypatch.setattr(
        "tests.utils.verification.send_request",
        lambda **kwargs: _chat_response(content="wrong"),
    )
    with pytest.raises(ResponseValidationError):
        run_payloads([_chat_payload(expected_response=["unicorn"], max_attempts=2)])


def test_transport_errors_are_not_retried_and_carry_diagnostics(monkeypatch):
    attempts = []

    def boom(**kwargs):
        attempts.append(1)
        raise ConnectionError("refused")

    monkeypatch.setattr("tests.utils.verification.send_request", boom)

    with pytest.raises(RuntimeError, match="pod is CrashLoopBackOff"):
        run_payloads(
            [_chat_payload(max_attempts=5)],
            describe_failure=lambda payload, error: f"pod is CrashLoopBackOff: {error}",
        )
    assert len(attempts) == 1, "transport failures must not be retried"


def test_default_failure_message_names_the_url(monkeypatch):
    monkeypatch.setattr(
        "tests.utils.verification.send_request",
        lambda **kwargs: (_ for _ in ()).throw(ConnectionError("refused")),
    )
    with pytest.raises(RuntimeError, match="http://localhost:4321"):
        run_payloads([_chat_payload(port=4321)])


def test_repeat_count_sends_the_payload_that_many_times(monkeypatch):
    sent = []
    monkeypatch.setattr(
        "tests.utils.verification.send_request",
        lambda **kwargs: (sent.append(kwargs["url"]), _chat_response())[1],
    )
    run_payloads([_chat_payload(repeat_count=3)])
    assert len(sent) == 3


def test_model_is_injected_but_never_overrides_an_explicit_one(monkeypatch):
    bodies = []
    monkeypatch.setattr(
        "tests.utils.verification.send_request",
        lambda **kwargs: (bodies.append(kwargs["payload"]), _chat_response())[1],
    )
    run_payloads(
        [_chat_payload(), _chat_payload(body={"model": "explicit"})], model="m"
    )
    assert bodies[0]["model"] == "m"
    assert bodies[1]["model"] == "explicit"


def test_payload_instances_are_not_mutated(monkeypatch):
    monkeypatch.setattr(
        "tests.utils.verification.send_request", lambda **kwargs: _chat_response()
    )
    payload = _chat_payload()
    run_payloads([payload], model="m")
    assert "model" not in payload.body, "shared payload instances must be reusable"


def test_headers_are_forwarded(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "tests.utils.verification.send_request",
        lambda **kwargs: (seen.update(kwargs), _chat_response())[1],
    )
    payload = _chat_payload()
    payload.headers = {"Host": "route.example.com"}
    run_payloads([payload])
    assert seen["headers"] == {"Host": "route.example.com"}


def test_final_validation_runs_after_the_last_iteration(monkeypatch):
    monkeypatch.setattr(
        "tests.utils.verification.send_request", lambda **kwargs: _chat_response()
    )
    calls = []

    class _WithFinal(ChatPayload):
        def final_validation(self):
            calls.append(1)

    run_payloads(
        [_WithFinal(body={}, expected_response=[], expected_log=[], repeat_count=2)]
    )
    assert calls == [1]
