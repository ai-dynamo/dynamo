# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU Mocker end-to-end coverage for Router queue policy.

This module covers what happens to a request *while it waits* for service:
queue admission, class deadlines, and the shedding that follows. Its companion,
``test_policy_class.py``, covers how queued work is *shared* between classes
once it is admitted.

The scenario here uses a pair of class configurations in which no field differs
except ``slo_ms``, so the deadline alone decides the outcome. Only the queued
request exercises expiry. The other request succeeds because it is dispatched
immediately, not because of its class SLO: ``slo_ms`` bounds waiting for
service, never an in-flight request.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import pytest

from tests.router.helper import wait_for_frontend_ready
from tests.router.mocker_process import MockerProcess
from tests.utils.constants import ROUTER_MODEL_NAME
from tests.utils.managed_process import DynamoFrontendProcess
from tests.utils.prometheus import sum_metric_samples

# Do not add pytest.mark.parallel: existing Mocker Router tests document races in
# process-global DistributedRuntime state under pytest-xdist.
pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.e2e,
    pytest.mark.router,
    pytest.mark.model(ROUTER_MODEL_NAME),
    pytest.mark.timeout(75),  # 3x the observed 24.71s end-to-end runtime.
]

CONFIG_DIR = Path(__file__).with_name("configs") / "policy_class"

# The two class configurations in slo_rejection.yaml differ only in `slo_ms`;
# quantum, busy threshold, and queue limit are identical.
SLO_CONFIG = "slo_rejection.yaml"
BLOCKER_POLICY_CLASS = "standard"
CANDIDATE_POLICY_CLASS = "latency"
BLOCKER_INPUT_TOKENS = 4096
CANDIDATE_INPUT_TOKENS = 32
CANDIDATE_SLO_MS = 3000  # Must match slo_rejection.yaml.
OVERLOAD_STATUS = 529  # DYN_HTTP_OVERLOAD_STATUS_CODE default.

ACTIVE_PREFILL_METRIC = "dynamo_frontend_worker_active_prefill_tokens"
PENDING_REQUESTS_METRIC = "dynamo_frontend_router_queue_pending_requests"
DEADLINE_EXPIRED_METRIC = "dynamo_frontend_router_queue_deadline_expired_total"


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HttpOutcome:
    policy_class: str
    request_index: int
    status: int
    error_body: str
    latency_s: float


def _completion_body(token_id: int, input_tokens: int) -> bytes:
    payload = {
        "model": ROUTER_MODEL_NAME,
        "prompt": [token_id] * input_tokens,
        "max_tokens": 1,
        "stream": True,
        "temperature": 0,
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


async def _post_completion(
    *,
    session: aiohttp.ClientSession,
    url: str,
    policy_class: str,
    request_index: int,
    body: bytes,
) -> HttpOutcome:
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    headers = {
        "content-type": "application/json",
        "x-dynamo-meta-policy-class": policy_class,
        "x-request-id": f"policy-queue-{policy_class}-{request_index:04d}",
    }
    async with session.post(url, data=body, headers=headers) as response:
        response_body = await response.read()
        latency_s = loop.time() - started_at
        # Error bodies are small structured JSON; keep them whole so the
        # rejection detail can be asserted rather than only reported.
        error_body = (
            response_body.decode("utf-8", errors="replace")
            if response.status != 200
            else ""
        )
        return HttpOutcome(
            policy_class=policy_class,
            request_index=request_index,
            status=response.status,
            error_body=error_body,
            latency_s=latency_s,
        )


@contextlib.contextmanager
def _policy_frontend(
    *,
    request: pytest.FixtureRequest,
    mocker: MockerProcess,
    frontend_port: int,
    config_name: str,
    case_name: str,
    readiness_policy_class: str,
) -> Iterator[None]:
    """Run one KV frontend bound to a policy-class config, ready to serve."""

    config_path = CONFIG_DIR / config_name
    extra_args = [
        "--namespace",
        mocker.namespace,
        "--discovery-backend",
        "etcd",
        "--load-aware",
        "--router-policy-config",
        str(config_path),
        "--router-min-initial-workers",
        "1",
    ]
    extra_env = {
        "DYN_REQUEST_PLANE": "nats",
        "DYN_LOG": "warn",
    }

    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        router_mode="kv",
        extra_args=extra_args,
        extra_env=extra_env,
        display_name=f"dynamo-frontend-policy-{case_name}",
    ):
        asyncio.run(
            wait_for_frontend_ready(
                frontend_url=f"http://localhost:{frontend_port}",
                expected_num_workers=1,
                timeout=60,
                engine_workers=mocker,
                store_backend="etcd",
                request_plane="nats",
                request_headers={"x-dynamo-meta-policy-class": readiness_policy_class},
            )
        )
        yield


async def _scrape(
    session: aiohttp.ClientSession,
    frontend_port: int,
    metric: str,
    labels: dict[str, str] | None = None,
) -> float:
    async with session.get(f"http://localhost:{frontend_port}/metrics") as response:
        response.raise_for_status()
        return sum_metric_samples(await response.text(), metric, labels)


async def _wait_for_metric(
    session: aiohttp.ClientSession,
    frontend_port: int,
    metric: str,
    labels: dict[str, str],
    predicate: Callable[[float], bool],
    timeout_s: float,
    what: str,
) -> float:
    """Poll one Prometheus sample until it satisfies `predicate`."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    observed = float("nan")
    while True:
        observed = await _scrape(session, frontend_port, metric, labels)
        if predicate(observed):
            return observed
        if loop.time() >= deadline:
            raise AssertionError(
                f"timed out after {timeout_s}s waiting for {what}: "
                f"{metric}{labels} last read {observed:g}"
            )
        await asyncio.sleep(0.05)


def _rejection_body(outcome: HttpOutcome) -> dict[str, Any]:
    """The parsed error body carried by a non-200 frontend response."""

    try:
        payload = json.loads(outcome.error_body)
    except json.JSONDecodeError as error:
        raise AssertionError(
            f"rejection body for the {outcome.policy_class} request is not "
            f"JSON: {outcome.error_body!r}"
        ) from error
    assert isinstance(payload, dict), (
        f"rejection body for the {outcome.policy_class} request is not a JSON "
        f"object: {payload!r}"
    )
    return payload


@dataclass(frozen=True)
class SloRejectionRun:
    blocker: HttpOutcome
    candidate: HttpOutcome
    outstanding_at_half_slo: bool
    pending_after: float
    dispatch_expiries: float
    admission_expiries: float
    deferred_wake_expiries: float


async def _run_slo_rejection(frontend_port: int) -> SloRejectionRun:
    """Hold the only worker busy, then let one short-SLO request miss its deadline."""

    url = f"http://localhost:{frontend_port}/v1/completions"
    timeout = aiohttp.ClientTimeout(total=120, connect=10)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Readiness probing runs its own requests through the router; wait for
        # their accounting to drain so the gauge below can only describe the
        # blocking request.
        await _wait_for_metric(
            session,
            frontend_port,
            ACTIVE_PREFILL_METRIC,
            {},
            predicate=lambda value: value == 0,
            timeout_s=30,
            what="readiness traffic to leave the worker idle",
        )

        # The blocking prompt is long enough, against a Mocker slowed 100x, to
        # keep the worker busy for far more than the candidate's 3s SLO, so the
        # candidate cannot begin service before its queue deadline passes.
        blocker = asyncio.create_task(
            _post_completion(
                session=session,
                url=url,
                policy_class=BLOCKER_POLICY_CLASS,
                request_index=0,
                body=_completion_body(2_000, BLOCKER_INPUT_TOKENS),
            )
        )
        candidate: asyncio.Task[HttpOutcome] | None = None
        try:
            # The router publishes this gauge from its own slot tracker when it
            # admits a request, and the same accounting decides whether a class
            # is busy. Once it is non-zero the strict
            # `prefill_busy_threshold: 0` comparison holds, so the next arrival
            # must be queued.
            await _wait_for_metric(
                session,
                frontend_port,
                ACTIVE_PREFILL_METRIC,
                {},
                predicate=lambda value: value > 0,
                timeout_s=30,
                what="the blocking request to occupy the worker",
            )

            candidate = asyncio.create_task(
                _post_completion(
                    session=session,
                    url=url,
                    policy_class=CANDIDATE_POLICY_CLASS,
                    request_index=1,
                    body=_completion_body(3_000, CANDIDATE_INPUT_TOKENS),
                )
            )
            # Pre-expiry witness that the request is waiting rather than refused
            # on arrival: halfway through its own SLO it is still outstanding,
            # while every arrival-time refusal (unknown class, queue limit)
            # answers at once. The pending-request gauge cannot serve here
            # because the router refreshes it only at scheduler lifecycle
            # boundaries, and the next one is the poll that sheds this request.
            _, outstanding = await asyncio.wait(
                {candidate}, timeout=CANDIDATE_SLO_MS / 2000
            )

            candidate_outcome = await candidate
            blocker_outcome = await blocker
        finally:
            # Never leave an in-flight request attached to a session that is
            # about to close, and retrieve every task's result: a failure above
            # would otherwise surface as an unretrieved task exception instead
            # of the real assertion.
            created = [task for task in (blocker, candidate) if task is not None]
            for task in created:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*created, return_exceptions=True)

        pending_after = await _scrape(
            session,
            frontend_port,
            PENDING_REQUESTS_METRIC,
            {"policy_class": CANDIDATE_POLICY_CLASS},
        )
        expiries = {
            stage: await _scrape(
                session,
                frontend_port,
                DEADLINE_EXPIRED_METRIC,
                {"policy_class": CANDIDATE_POLICY_CLASS, "stage": stage},
            )
            for stage in ("dispatch", "admission", "deferred_wake")
        }

    return SloRejectionRun(
        blocker=blocker_outcome,
        candidate=candidate_outcome,
        outstanding_at_half_slo=bool(outstanding),
        pending_after=pending_after,
        dispatch_expiries=expiries["dispatch"],
        admission_expiries=expiries["admission"],
        deferred_wake_expiries=expiries["deferred_wake"],
    )


@pytest.mark.usefixtures(
    "runtime_services_dynamic_ports",
    "predownload_tokenizers",
)
def test_class_slo_rejects_queued_request_at_dispatch_gate(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    dynamo_dynamic_ports,
) -> None:
    """A queued request whose class SLO passed is rejected, not dispatched late.

    One slow 4096-token request holds the only worker slot for far longer than
    3s. A second request in a 3s-SLO class therefore enters its class queue and
    cannot begin service before its queue deadline, so it is shed at the
    dispatch gate instead of being served late.

    Only that queued request exercises expiry. The blocking request arrives
    while the worker is idle and is dispatched without queueing, so it completes
    for that reason rather than because its class carries a 30s SLO; a class SLO
    never bounds a request that is already running.
    """

    monkeypatch.setenv(
        "DYN_SYSTEM_PORT",
        str(dynamo_dynamic_ports.system_ports[0]),
    )
    frontend_port = dynamo_dynamic_ports.frontend_port
    mocker_args = {
        # 100x slower than the modeled engine, so one 4096-token prompt holds
        # the worker for several seconds without any client-side sleeping.
        "speedup_ratio": 0.01,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 8192,
        "enable_prefix_caching": False,
    }

    with MockerProcess(
        request,
        mocker_args=mocker_args,
        num_mockers=1,
        store_backend="etcd",
        request_plane="nats",
        model_name=ROUTER_MODEL_NAME,
    ) as mocker:
        with _policy_frontend(
            request=request,
            mocker=mocker,
            frontend_port=frontend_port,
            config_name=SLO_CONFIG,
            case_name="slo",
            readiness_policy_class=BLOCKER_POLICY_CLASS,
        ):
            run = asyncio.run(_run_slo_rejection(frontend_port))

    logger.info(
        "SLO arm: blocker status=%d in %.2fs, candidate status=%d in %.2fs; "
        "outstanding at half SLO=%s, pending after=%g, expiries dispatch=%g "
        "admission=%g deferred_wake=%g; candidate body=%s",
        run.blocker.status,
        run.blocker.latency_s,
        run.candidate.status,
        run.candidate.latency_s,
        run.outstanding_at_half_slo,
        run.pending_after,
        run.dispatch_expiries,
        run.admission_expiries,
        run.deferred_wake_expiries,
        run.candidate.error_body,
    )

    # The blocking request never queues, so no deadline check applies to it.
    assert run.blocker.status == 200, (
        f"the blocking request failed with {run.blocker.status}: "
        f"{run.blocker.error_body!r}"
    )

    # The short-SLO request was queued behind it and shed rather than served.
    assert run.outstanding_at_half_slo, (
        f"the {CANDIDATE_POLICY_CLASS} request was answered before half its "
        "SLO elapsed, so it was refused on arrival rather than queued"
    )
    assert run.candidate.status == OVERLOAD_STATUS, (
        f"expected the queued {CANDIDATE_POLICY_CLASS} request to be rejected "
        f"with {OVERLOAD_STATUS}, got {run.candidate.status}: "
        f"{run.candidate.error_body!r}"
    )
    body = _rejection_body(run.candidate)
    assert body.get("code") == OVERLOAD_STATUS, body
    assert body.get("type") == "Overloaded", body
    details = body.get("details")
    assert isinstance(details, dict), body
    # Assert the fields this contract owns, as a subset: a later release may add
    # a field, and that should not fail a test about deadline rejection.
    assert details.get("policy_class") == CANDIDATE_POLICY_CLASS, details
    assert details.get("stage") == "dispatch", details
    assert details.get("slo_ms") == CANDIDATE_SLO_MS, details
    overdue_ms = details.get("overdue_ms")
    # How far past the deadline the shedding poll ran is timing dependent; only
    # its sign is a contract.
    assert isinstance(overdue_ms, int) and overdue_ms > 0, details
    # A rejection can only be reported once the deadline has passed, and the
    # deadline is arrival + SLO with arrival strictly after the client sent the
    # request, so the wait itself is evidence the request sat in the queue.
    assert run.candidate.latency_s >= CANDIDATE_SLO_MS / 1000, run.candidate

    # Shedding reverses the queue accounting the request applied on the way in.
    assert run.pending_after == 0, run.pending_after

    # Exactly one expiry, at the dispatch gate and nowhere else.
    assert run.dispatch_expiries == 1, run.dispatch_expiries
    assert run.admission_expiries == 0, run.admission_expiries
    assert run.deferred_wake_expiries == 0, run.deferred_wake_expiries
