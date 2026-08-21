# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Send payloads and assert on responses, with no knowledge of the deployment.

This is the verification half of a Dynamo functional test. It takes payloads
that already know their URL (see ``tests/utils/inference_endpoint.py``) and
runs the send/validate/retry loop. It never launches a process, applies a
Kubernetes manifest, or opens a port-forward, so the same code path serves
in-container tests (``tests/serve``) and cluster deploy tests
(``tests/deploy``).

Log-based assertions are the one thing a response cannot answer. Rather than
give this module a process handle, callers pass a ``log_source`` -- anything
with ``read_logs() -> str`` (the ``LogReadable`` protocol in
``tests/utils/router_logs.py``). A payload that sets ``expected_log`` without a
``log_source`` is a configuration error and is reported as one.

Like ``tests/utils/inference_endpoint.py`` this module imports no ``dynamo``
package, so it is importable on the bare runner the Kubernetes deploy job uses.
"""

from __future__ import annotations

import json
import logging
import re
import time
from copy import deepcopy
from typing import Any, Callable, Iterable, Optional, Sequence

import requests

from tests.utils.client import send_request
from tests.utils.payloads import BasePayload

logger = logging.getLogger(__name__)

# Backoff growth between validation retries. 1.5 keeps the worst-case sleep
# budget bounded for max_attempts up to ~6.
_RETRY_BACKOFF_FACTOR = 1.5


class EngineResponseError(Exception):
    """A response was unusable: non-200 status, or the handler raised.

    The class name is load-bearing: tests select it by name in
    ``@pytest.mark.flaky(only_rerun=["EngineResponseError"])``.
    """


class ResponseValidationError(EngineResponseError):
    """Validation/assertion failure during process_response.

    Subset of EngineResponseError raised only when payload.process_response
    asserts on response content (the case the in-process retry was designed
    for). Status (non-200) and handler errors continue to raise the parent
    EngineResponseError so they surface immediately and aren't masked by
    payload.max_attempts.
    """


class EngineLogError(Exception):
    """Expected log patterns were not found."""


def validate_expected_logs(patterns: Sequence[str], log_source: Any) -> None:
    """Assert every regex in ``patterns`` appears in ``log_source.read_logs()``.

    Raises:
        EngineLogError: if the log is empty or any pattern is missing.
    """
    content = log_source.read_logs() or ""
    if not content:
        log_path = getattr(log_source, "log_path", "<unknown>")
        raise EngineLogError(f"Log file not available or empty at path: {log_path}")

    missing = [p for p in patterns if not re.compile(p).search(content)]
    if missing:
        sample = content[-1000:] if len(content) > 1000 else content
        raise EngineLogError(
            f"Missing expected log patterns: {missing}\n\nLog sample:\n{sample}"
        )
    logger.info("SUCCESS: All expected log patterns: %s found", list(patterns))


def check_response(
    payload: BasePayload,
    response: requests.Response,
    *,
    log_source: Optional[Any] = None,
) -> None:
    """Validate one response against the payload's expectations.

    Args:
        payload: The payload that produced ``response``.
        response: The HTTP response.
        log_source: Object exposing ``read_logs() -> str``, required only when
            ``payload.expected_log`` is non-empty.

    Raises:
        EngineResponseError: non-200 status, or the response handler raised.
        ResponseValidationError: the handler's assertions failed (the only
            error the caller's retry loop re-issues).
        EngineLogError: expected log patterns are missing.
    """
    if response.status_code != 200:
        logger.error("Response returned non-200 status code: %d", response.status_code)

        error_msg = f"Response returned non-200 status code: {response.status_code}"
        try:
            error_data = response.json()
            if "error" in error_data:
                error_msg += f"\nError details: {error_data['error']}"
            logger.error("Response error details: %s", json.dumps(error_data, indent=2))
        except Exception:
            logger.error("Response text: %s", response.text[:500])

        raise EngineResponseError(error_msg)

    try:
        content = payload.process_response(response)

        logger.info(
            "Extracted content: \n%s",
            content[:200] + "..."
            if isinstance(content, str) and len(content) > 200
            else content,
        )
    except AssertionError as e:
        raise ResponseValidationError(str(e))
    except Exception as e:
        raise EngineResponseError(f"Failed to handle response: {e}")

    if payload.expected_log:
        if log_source is None:
            raise EngineLogError(
                f"{type(payload).__name__} sets expected_log={payload.expected_log!r} "
                "but no log_source was provided. Log assertions need a deployment "
                "handle, so the test is deployment-coupled: run it through a "
                "backend that exposes read_logs() and mark it topology_dependent."
            )
        # The kv event sometimes needs extra time to arrive and be reflected in
        # the log.
        time.sleep(0.5)
        validate_expected_logs(payload.expected_log, log_source)


def run_payloads(
    payloads: Iterable[BasePayload],
    *,
    log: Optional[logging.Logger] = None,
    log_source: Optional[Any] = None,
    model: Optional[str] = None,
    describe_failure: Optional[Callable[[BasePayload, BaseException], str]] = None,
) -> None:
    """Send each payload and assert on each response.

    Payloads are expected to already carry their target address (``base_url``,
    or ``host``/``port`` for the in-container suites). Nothing here knows how
    the deployment was created.

    Args:
        payloads: Payloads to run, in order. Each is deep-copied first, so
            shared/parametrized payload instances are never mutated.
        log: Logger for progress.
        log_source: Passed through to :func:`check_response` for
            ``expected_log`` assertions.
        model: Injected into each payload body when the payload has not already
            set a model.
        describe_failure: Builds the message for a transport-level failure.
            Backends use it to attach server logs, PIDs, or pod status.

    Raises:
        ResponseValidationError: a payload's assertions failed on every attempt.
        RuntimeError: a request failed at the transport layer (wrapping the
            original error).
    """
    log = log or logger

    for _payload in payloads:
        log.info("TESTING: Payload: %s", _payload.__class__.__name__)

        # Per-iteration copy so callers can safely reuse shared config
        # instances across parametrized cases.
        payload = deepcopy(_payload)
        if model is not None and hasattr(payload, "with_model"):
            payload = payload.with_model(model)

        for iteration in range(payload.repeat_count):
            # Resolve an iteration-specific body once so validation retries
            # resend the same request.
            request_body = payload.body_for_iteration(iteration)
            # Re-issue the request (deployment stays up) on validation failure
            # when payload.max_attempts > 1. See tests/README.md "Flaky Tests"
            # for when this is appropriate.
            last_err: Optional[ResponseValidationError] = None
            try:
                for attempt in range(payload.max_attempts):
                    try:
                        response = send_request(
                            url=payload.url(),
                            payload=request_body,
                            timeout=payload.timeout,
                            method=payload.method,
                            stream=payload.http_stream,
                            headers=payload.headers or None,
                        )
                        check_response(payload, response, log_source=log_source)
                        last_err = None
                        break
                    except ResponseValidationError as e:
                        last_err = e
                        if attempt < payload.max_attempts - 1:
                            wait = 1.0 * (_RETRY_BACKOFF_FACTOR**attempt)
                            log.warning(
                                "%s request failed (attempt %d/%d): %s — retrying in %.1fs",
                                type(payload).__name__,
                                attempt + 1,
                                payload.max_attempts,
                                e,
                                wait,
                            )
                            time.sleep(wait)
            except Exception as e:
                # Transport / connection failures (and payload.url() failures)
                # aren't retried by design; the inner loop only retries
                # ResponseValidationError. Re-raise with backend diagnostics so
                # a CI failure is diagnosable in one pass rather than yielding a
                # bare ReadTimeout.
                message = (
                    describe_failure(payload, e)
                    if describe_failure is not None
                    else _default_failure_message(payload, e)
                )
                raise RuntimeError(message) from e
            if last_err is not None:
                raise last_err

        # e.g. CachedTokensChatPayload asserts a cache-hit delta after the last
        # iteration.
        if hasattr(payload, "final_validation"):
            payload.final_validation()


def _default_failure_message(payload: BasePayload, error: BaseException) -> str:
    try:
        url = payload.url()
    except Exception:
        url = "<payload.url() raised>"
    return (
        f"{type(payload).__name__} request failed "
        f"(method={payload.method}, url={url}, timeout={payload.timeout}s)\n"
        f"Original error: {type(error).__name__}: {error}"
    )
