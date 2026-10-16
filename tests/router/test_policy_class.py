# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU Mocker end-to-end coverage for weighted Router policy classes."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import pytest

from tests.router.e2e_harness import allocate_frontend_ports
from tests.router.helper import wait_for_frontend_ready
from tests.router.mocker_process import MockerProcess
from tests.utils.constants import ROUTER_MODEL_NAME
from tests.utils.managed_process import DynamoFrontendProcess

# Do not add pytest.mark.parallel: existing Mocker Router tests document races in
# process-global DistributedRuntime state under pytest-xdist.
pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.e2e,
    pytest.mark.router,
    pytest.mark.model(ROUTER_MODEL_NAME),
    pytest.mark.timeout(140),  # 3x the observed 45.90s end-to-end runtime.
]

POLICY_CLASSES = ("premium", "regular")
BACKEND_MAX_NUM_SEQS = 4
REQUESTS_PER_CLASS = 320
INPUT_TOKENS = 512
COMPLETION_PREFIX = REQUESTS_PER_CLASS
EARLY_PREMIUM_COMPLETIONS = REQUESTS_PER_CLASS // 4
LATE_PREMIUM_COMPLETIONS = 3 * REQUESTS_PER_CLASS // 4
CONFIG_DIR = Path(__file__).with_name("configs") / "policy_class"


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HttpOutcome:
    policy_class: str
    request_index: int
    status: int
    error_body: str
    latency_s: float


@dataclass(frozen=True)
class ClientResult:
    """HTTP outcomes in the order the client observed them complete."""

    outcomes: tuple[HttpOutcome, ...]

    def prefix_counts(self, size: int) -> Counter[str]:
        return Counter(outcome.policy_class for outcome in self.outcomes[:size])

    def counts_through_completion(
        self,
        policy_class: str,
        completion_number: int,
    ) -> Counter[str]:
        counts: Counter[str] = Counter()
        for outcome in self.outcomes:
            counts[outcome.policy_class] += 1
            if counts[policy_class] == completion_number:
                return counts
        raise AssertionError(
            f"only observed {counts[policy_class]} completions for "
            f"{policy_class!r}, expected {completion_number}"
        )


def _request_body(policy_class: str, request_index: int) -> bytes:
    # Give every request a distinct prefix so cache overlap cannot change its
    # DRR cost. All requests still have exactly the same uncached token length.
    token_id = 1000 + request_index * len(POLICY_CLASSES)
    token_id += POLICY_CLASSES.index(policy_class)
    payload = {
        "model": ROUTER_MODEL_NAME,
        "prompt": [token_id] * INPUT_TOKENS,
        "max_tokens": 1,
        "stream": True,
        "temperature": 0,
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


async def _send_one(
    *,
    session: aiohttp.ClientSession,
    start_gate: asyncio.Event,
    ready_tasks: asyncio.Queue[None],
    url: str,
    policy_class: str,
    request_index: int,
) -> HttpOutcome:
    loop = asyncio.get_running_loop()
    body = _request_body(policy_class, request_index)
    ready_tasks.put_nowait(None)
    await start_gate.wait()
    started_at = loop.time()
    headers = {
        "content-type": "application/json",
        "x-dynamo-meta-policy-class": policy_class,
        "x-request-id": f"policy-class-{policy_class}-{request_index:04d}",
    }
    async with session.post(url, data=body, headers=headers) as response:
        response_body = await response.read()
        latency_s = loop.time() - started_at
        error_body = (
            response_body[:500].decode("utf-8", errors="replace")
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


async def _send_balanced_workload(frontend_port: int) -> ClientResult:
    url = f"http://localhost:{frontend_port}/v1/completions"
    start_gate = asyncio.Event()
    ready_tasks: asyncio.Queue[None] = asyncio.Queue()
    timeout = aiohttp.ClientTimeout(total=120, connect=10)
    connector = aiohttp.TCPConnector(limit=0)

    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
    ) as session:
        tasks = []
        for request_index in range(REQUESTS_PER_CLASS):
            policy_order = (
                POLICY_CLASSES
                if request_index % 2 == 0
                else tuple(reversed(POLICY_CLASSES))
            )
            for policy_class in policy_order:
                tasks.append(
                    asyncio.create_task(
                        _send_one(
                            session=session,
                            start_gate=start_gate,
                            ready_tasks=ready_tasks,
                            url=url,
                            policy_class=policy_class,
                            request_index=request_index,
                        )
                    )
                )

        # Release only after every class-balanced client task reaches the gate.
        for _ in tasks:
            await ready_tasks.get()
        start_gate.set()
        outcomes = []
        for completed in asyncio.as_completed(tasks):
            outcomes.append(await completed)
        return ClientResult(outcomes=tuple(outcomes))


def _run_case(
    *,
    request: pytest.FixtureRequest,
    mocker: MockerProcess,
    frontend_port: int,
    config_name: str,
    case_name: str,
) -> ClientResult:
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
                request_headers={"x-dynamo-meta-policy-class": "regular"},
            )
        )

        result = asyncio.run(_send_balanced_workload(frontend_port))
        failures = [outcome for outcome in result.outcomes if outcome.status != 200]
        assert not failures, (
            f"{case_name}: {len(failures)} of {len(result.outcomes)} "
            "requests failed: "
            f"{failures[:5]}"
        )
        assert len(result.outcomes) == 2 * REQUESTS_PER_CLASS
        return result


@pytest.mark.usefixtures(
    "runtime_services_dynamic_ports",
    "predownload_tokenizers",
)
def test_weighted_policy_class_client_completion_priority(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    dynamo_dynamic_ports,
) -> None:
    """The client observes weighted premium service without regular starvation."""

    monkeypatch.setenv(
        "DYN_SYSTEM_PORT",
        str(dynamo_dynamic_ports.system_ports[0]),
    )
    frontend_ports = [
        dynamo_dynamic_ports.frontend_port,
        *allocate_frontend_ports(request, 1),
    ]
    mocker_args = {
        "speedup_ratio": 1.0,
        "max_num_seqs": BACKEND_MAX_NUM_SEQS,
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
        equal = _run_case(
            request=request,
            mocker=mocker,
            frontend_port=frontend_ports[0],
            config_name="equal_share.yaml",
            case_name="equal",
        )
        weighted = _run_case(
            request=request,
            mocker=mocker,
            frontend_port=frontend_ports[1],
            config_name="premium_4x.yaml",
            case_name="weighted",
        )

    equal_prefix = equal.prefix_counts(COMPLETION_PREFIX)
    weighted_prefix = weighted.prefix_counts(COMPLETION_PREFIX)
    equal_at_premium_drain = equal.counts_through_completion(
        "premium",
        REQUESTS_PER_CLASS,
    )
    weighted_at_premium_drain = weighted.counts_through_completion(
        "premium",
        REQUESTS_PER_CLASS,
    )
    weighted_at_premium_early = weighted.counts_through_completion(
        "premium",
        EARLY_PREMIUM_COMPLETIONS,
    )
    weighted_at_premium_late = weighted.counts_through_completion(
        "premium",
        LATE_PREMIUM_COMPLETIONS,
    )
    equal_regular_remaining = REQUESTS_PER_CLASS - equal_at_premium_drain["regular"]
    weighted_regular_remaining = (
        REQUESTS_PER_CLASS - weighted_at_premium_drain["regular"]
    )

    logger.info(
        "Client completion prefixes: equal=%d:%d, weighted=%d:%d; "
        "regular remaining when premium drained: equal=%d, weighted=%d",
        equal_prefix["premium"],
        equal_prefix["regular"],
        weighted_prefix["premium"],
        weighted_prefix["regular"],
        equal_regular_remaining,
        weighted_regular_remaining,
    )

    # The equal control guards against client connection/task-order bias.
    assert 140 <= equal_prefix["premium"] <= 180, equal_prefix
    assert equal_regular_remaining <= 20, equal_at_premium_drain

    # In the weighted arm, the first 320 client-observed completions should be
    # approximately 256:64. When all 320 premium responses have completed,
    # approximately 240 regular responses should still be outstanding.
    assert weighted_prefix["premium"] >= 240, weighted_prefix
    assert weighted_prefix["regular"] >= 48, weighted_prefix
    assert 220 <= weighted_regular_remaining <= 260, weighted_at_premium_drain

    assert weighted_prefix["premium"] - equal_prefix["premium"] >= 60, (
        equal_prefix,
        weighted_prefix,
    )
    assert weighted_regular_remaining - equal_regular_remaining >= 200, (
        equal_at_premium_drain,
        weighted_at_premium_drain,
    )

    # Regular must make progress throughout premium's drain, not only after it.
    assert weighted_at_premium_early["regular"] >= 12, weighted_at_premium_early
    assert (
        weighted_at_premium_late["regular"] - weighted_at_premium_early["regular"] >= 32
    ), (weighted_at_premium_early, weighted_at_premium_late)
