# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment-agnostic addressing for Dynamo test traffic.

A test that only sends payloads and asserts on responses needs exactly one
thing from the deployment: where to send the request. ``InferenceEndpoint``
carries that and nothing else, so the same test body works against a locally
launched process (``http://localhost:8000``), a port-forwarded Kubernetes
frontend (``http://localhost:<ephemeral>``), or a remote ingress
(``https://dynamo.example.com``).

Deployment mechanics (launching processes, applying DynamoGraphDeployment CRs,
opening port-forwards) live in the backend-specific modules --
``tests/utils/engine_process.py`` and ``tests/deploy/dgd_utils.py``.
Both hand back an ``InferenceEndpoint``; neither is visible to a
deployment-agnostic test.

This module is deliberately free of any ``dynamo`` import: the Kubernetes
deploy job runs pytest on a bare runner that installs only
``container/deps/requirements.test.txt`` and never the ``ai-dynamo`` wheel.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import Mapping, Optional, Sequence

from tests.utils.client import send_request

logger = logging.getLogger(__name__)

# Endpoint used to decide a deployment can serve inference. Chat completions is
# the narrowest probe that exercises the whole path (frontend -> router ->
# worker -> model), which /health and /v1/models do not: both can report ready
# while the model is still loading.
DEFAULT_READINESS_ENDPOINT = "/v1/chat/completions"

# Pause after the first successful probe before declaring the deployment ready.
# Carried over from wait_for_model_availability, which slept 5s at the same
# point. The first request can succeed while the rest of the graph is still
# settling (workers registering, the router filling its view), and the tests
# that follow send a single non-retried request. Removing it is a separate,
# measurable change rather than a side effect of this refactor.
DEFAULT_SETTLE_SECONDS = 5.0


class NotServingError(RuntimeError):
    """A deployment did not become able to serve inference in time."""


@dataclass(frozen=True)
class InferenceEndpoint:
    """An address a test can send inference requests to.

    Attributes:
        base_url: Scheme, host and port with no trailing slash and no path
            (for example ``http://localhost:8000``).
        model: Model name to put in request bodies. ``None`` when the caller
            supplies it per payload.
        headers: Extra headers sent with every request. Used by deployments
            that route on a header, such as the Gateway API ``Host`` header.
    """

    base_url: str
    model: Optional[str] = None
    headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("base_url must be a non-empty URL")
        if "://" not in self.base_url:
            raise ValueError(
                f"base_url must include a scheme (got {self.base_url!r}); "
                "pass e.g. http://localhost:8000"
            )
        # Normalize once so url() never produces a double slash and callers can
        # compare endpoints by value.
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    def url(self, path: str) -> str:
        """Absolute URL for an API path (leading slash optional)."""
        return f"{self.base_url}/{path.lstrip('/')}"

    def with_model(self, model: str) -> "InferenceEndpoint":
        return replace(self, model=model)

    def with_headers(self, headers: Mapping[str, str]) -> "InferenceEndpoint":
        return replace(self, headers={**self.headers, **headers})

    @classmethod
    def from_port(
        cls,
        port: int,
        *,
        host: str = "localhost",
        scheme: str = "http",
        model: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> "InferenceEndpoint":
        """Build an endpoint from a host/port pair.

        Bridges the in-container suites, which allocate integer ports per test,
        to the URL-only interface.
        """
        return cls(
            base_url=f"{scheme}://{host}:{port}",
            model=model,
            headers=dict(headers or {}),
        )


@dataclass(frozen=True)
class DeploymentEndpoints:
    """Every address a deployment exposes.

    Deployment-agnostic tests take ``frontend`` alone. Tests that assert on
    deployment shape -- per-worker metrics, routing decisions, replica counts --
    are marked ``topology_dependent`` and may also read ``workers``.
    """

    frontend: InferenceEndpoint
    workers: Sequence[InferenceEndpoint] = ()

    def worker(self, index: int) -> InferenceEndpoint:
        """Worker system endpoint by zero-based index.

        Raises:
            IndexError: with the available count, because "worker 2 of a
                1-worker deployment" is a test-configuration bug that should
                not surface as a bare connection error.
        """
        try:
            return self.workers[index]
        except IndexError:
            raise IndexError(
                f"worker index {index} out of range: deployment exposes "
                f"{len(self.workers)} worker endpoint(s)"
            ) from None


def probe_serving(
    endpoint: InferenceEndpoint,
    *,
    model: Optional[str] = None,
    api_path: str = DEFAULT_READINESS_ENDPOINT,
    timeout: float = 30.0,
) -> Optional[str]:
    """Send one minimal inference request.

    Returns:
        ``None`` when the deployment served the request, otherwise a short
        human-readable reason it did not.
    """
    resolved_model = model or endpoint.model
    if not resolved_model:
        raise ValueError(
            "probe_serving needs a model name: pass model= or use "
            "InferenceEndpoint.with_model()"
        )
    body = {
        "model": resolved_model,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 1,
        "stream": False,
    }
    try:
        response = send_request(
            url=endpoint.url(api_path),
            payload=body,
            timeout=timeout,
            method="POST",
            headers=endpoint.headers or None,
            log_level=logging.DEBUG,
        )
    except Exception as exc:  # transport errors are expected while starting up
        return f"{type(exc).__name__}: {exc}"
    if response.status_code == 200:
        return None
    return f"HTTP {response.status_code}: {response.text[:200]}"


def wait_until_serving(
    endpoint: InferenceEndpoint,
    *,
    model: Optional[str] = None,
    api_path: str = DEFAULT_READINESS_ENDPOINT,
    timeout: float = 300.0,
    poll_interval: float = 5.0,
    request_timeout: float = 30.0,
    settle_seconds: float = DEFAULT_SETTLE_SECONDS,
    log: Optional[logging.Logger] = None,
) -> None:
    """Block until the deployment serves inference, or raise.

    This is the single definition of "deployed and able to receive inference
    requests", shared by the local-process and Kubernetes backends. Backends
    remain responsible for their own liveness gates first (process/port health,
    or the DynamoGraphDeployment Ready condition); this gate then confirms the
    thing those gates cannot: that a request actually completes.

    Args:
        endpoint: Where to send the probe.
        model: Model name, defaulting to ``endpoint.model``.
        api_path: API path to probe.
        timeout: Total wall-clock budget in seconds.
        poll_interval: Delay between attempts.
        request_timeout: Per-attempt HTTP timeout.
        log: Logger for progress; defaults to this module's logger.

    Raises:
        NotServingError: if the budget is exhausted, quoting the last failure
            so a CI log shows why rather than only that it timed out.
        ValueError: if no model name is available.
    """
    log = log or logger
    # Monotonic so a wall-clock adjustment mid-run cannot extend or truncate
    # the budget.
    deadline = time.monotonic() + timeout
    attempt = 0
    reason = "no attempt completed"

    while True:
        attempt += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        reason = (
            probe_serving(
                endpoint,
                model=model,
                api_path=api_path,
                # Never let one attempt overrun the overall budget.
                timeout=min(request_timeout, remaining),
            )
            or ""
        )
        if not reason:
            log.info(
                "Deployment at %s is serving %s (attempt %d); settling %.0fs",
                endpoint.base_url,
                model or endpoint.model,
                attempt,
                settle_seconds,
            )
            if settle_seconds > 0:
                time.sleep(settle_seconds)
            return
        log.debug(
            "Not serving yet at %s (attempt %d): %s",
            endpoint.base_url,
            attempt,
            reason,
        )
        if deadline - time.monotonic() <= 0:
            break
        time.sleep(min(poll_interval, max(deadline - time.monotonic(), 0)))

    raise NotServingError(
        f"{endpoint.base_url} did not serve {model or endpoint.model!r} within "
        f"{timeout:.0f}s ({attempt} attempt(s)); last failure: {reason}"
    )
