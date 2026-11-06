# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU Mocker end-to-end coverage for the kv-capacity-aware decode policy.

One prefill Mocker serves one decode Mocker with two data-parallel ranks.
Requests pinned with ``nvext.decode_worker_id`` and ``nvext.dp_rank`` load the
ranks unevenly. Unpinned requests then show, through ``nvext.worker_id``,
which decode rank each policy picks.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import pytest

from tests.router.e2e_harness import allocate_frontend_ports
from tests.router.mocker_process import DisaggMockerProcess
from tests.router.router_process import FrontendRouterProcess
from tests.utils.constants import ROUTER_MODEL_NAME

# Do not add pytest.mark.parallel: existing Mocker Router tests document races in
# process-global DistributedRuntime state under pytest-xdist.
pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.e2e,
    pytest.mark.router,
    pytest.mark.model(ROUTER_MODEL_NAME),
    pytest.mark.timeout(90),  # 3x the observed 29.39s end-to-end runtime.
]

BLOCK_SIZE = 16
SPEEDUP_RATIO = 5.0
DECODE_DP_SIZE = 2
# Each decode rank has this many KV blocks and advertises them as total_kv_blocks.
DECODE_RANK_BLOCKS = 256
# Rank 0 runs one large request and rank 1 runs two small ones. The probe would
# fill rank 0 to 270 of 256 blocks and rank 1 to 96. With W = 1000, the default
# cost D + W·N is 270 + 1000 on rank 0 and 96 + 2000 on rank 1, so the default
# picks rank 0. The kv-capacity-aware policy skips rank 0 and picks rank 1.
LARGE_BLOCKS = 190
SMALL_BLOCKS = 8
PROBE_BLOCKS = 80
# The pinned requests must still run while the probes are routed. A large
# request and its output must also fit its rank: 190 + 600 / 16 < 256.
PINNED_MAX_TOKENS = 600
PROBE_MAX_TOKENS = 4
# A prefill worker returns one token. The second chunk comes from the decode
# worker, so the router has booked the request on its decode rank by then.
DECODING_CHUNKS = 2
FULL_PROBES = 4
CONFIG_DIR = Path(__file__).with_name("configs") / "kv_capacity_policy"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Outcome:
    label: str
    status: int
    prefill_worker_id: int | None
    decode_worker_id: int | None
    decode_dp_rank: int | None
    error_body: str


@dataclass(frozen=True)
class ArmResult:
    decode_worker_id: int
    pinned: tuple[Outcome, ...]
    probe: Outcome
    full_pinned: tuple[Outcome, ...]
    full_probes: tuple[Outcome, ...]


def _payload(
    token_id: int,
    blocks: int,
    max_tokens: int,
    pin: tuple[int, int] | None = None,
) -> dict[str, Any]:
    nvext: dict[str, Any] = {"extra_fields": ["worker_id"]}
    if pin is not None:
        nvext["decode_worker_id"], nvext["dp_rank"] = pin
    # Each request repeats its own token.
    return {
        "model": ROUTER_MODEL_NAME,
        "prompt": [token_id] * (blocks * BLOCK_SIZE),
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "temperature": 0,
        "nvext": nvext,
    }


async def _send(
    session: aiohttp.ClientSession,
    url: str,
    label: str,
    payload: dict[str, Any],
    decoding: asyncio.Event | None = None,
) -> Outcome:
    async with session.post(url, json=payload) as response:
        if response.status != 200:
            body = await response.text()
            return Outcome(label, response.status, None, None, None, body[:500])
        chunks = 0
        worker_id: dict[str, Any] = {}
        async for raw_line in response.content:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if not data or data == "[DONE]":
                continue
            chunk = json.loads(data)
            worker_id = chunk.get("nvext", {}).get("worker_id") or worker_id
            if chunk.get("choices"):
                chunks += 1
                if decoding is not None and chunks >= DECODING_CHUNKS:
                    decoding.set()
        return Outcome(
            label,
            200,
            worker_id.get("prefill_worker_id"),
            worker_id.get("decode_worker_id"),
            worker_id.get("decode_dp_rank"),
            "",
        )


async def _start_pinned(
    session: aiohttp.ClientSession,
    url: str,
    requests: list[tuple[str, dict[str, Any]]],
) -> list[asyncio.Task[Outcome]]:
    """Start the pinned requests and return once each one is decoding."""
    tasks = []
    for label, payload in requests:
        decoding = asyncio.Event()
        task = asyncio.create_task(_send(session, url, label, payload, decoding))
        waiter = asyncio.create_task(decoding.wait())
        done, _ = await asyncio.wait(
            {task, waiter}, timeout=60, return_when=asyncio.FIRST_COMPLETED
        )
        waiter.cancel()
        assert waiter in done, (
            f"{label} did not start decoding: "
            f"{task.result() if task in done else 'no response within 60 s'}"
        )
        tasks.append(task)
    return tasks


async def _wait_for_decode_worker(
    session: aiohttp.ClientSession,
    url: str,
    tokens: itertools.count[int],
) -> int:
    """Wait until requests pass through prefill and decode; return the decode worker."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 60
    last: Outcome | str | None = None
    while loop.time() < deadline:
        try:
            last = await _send(session, url, "ready", _payload(next(tokens), 1, 2))
        except aiohttp.ClientError as error:
            last = repr(error)
        else:
            if (
                last.status == 200
                and last.prefill_worker_id is not None
                and last.decode_worker_id is not None
            ):
                return last.decode_worker_id
        await asyncio.sleep(0.25)
    raise AssertionError(f"frontend did not route through prefill and decode: {last}")


async def _exercise(
    url: str, tokens: itertools.count[int], *, all_full: bool
) -> ArmResult:
    timeout = aiohttp.ClientTimeout(total=60, connect=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        decode_id = await _wait_for_decode_worker(session, url, tokens)
        for rank in range(DECODE_DP_SIZE):
            check = await _send(
                session,
                url,
                f"pin@{rank}",
                _payload(next(tokens), 1, 2, (decode_id, rank)),
            )
            assert (check.status, check.decode_dp_rank) == (200, rank), check

        # Rank 0 gets one large request, rank 1 two small ones.
        pinned = await _start_pinned(
            session,
            url,
            [
                (
                    "large@0",
                    _payload(
                        next(tokens), LARGE_BLOCKS, PINNED_MAX_TOKENS, (decode_id, 0)
                    ),
                ),
                *(
                    (
                        "small@1",
                        _payload(
                            next(tokens),
                            SMALL_BLOCKS,
                            PINNED_MAX_TOKENS,
                            (decode_id, 1),
                        ),
                    )
                    for _ in range(2)
                ),
            ],
        )
        probe = await _send(
            session,
            url,
            "probe",
            _payload(next(tokens), PROBE_BLOCKS, PROBE_MAX_TOKENS),
        )
        pinned_outcomes = tuple(await asyncio.gather(*pinned))

        full_pinned: tuple[Outcome, ...] = ()
        full_probes: tuple[Outcome, ...] = ()
        if all_full:
            # A large request on each rank: the probes fit on neither.
            full_tasks = await _start_pinned(
                session,
                url,
                [
                    (
                        f"large@{rank}",
                        _payload(
                            next(tokens),
                            LARGE_BLOCKS,
                            PINNED_MAX_TOKENS,
                            (decode_id, rank),
                        ),
                    )
                    for rank in range(DECODE_DP_SIZE)
                ],
            )
            full_probes = tuple(
                await asyncio.gather(
                    *(
                        _send(
                            session,
                            url,
                            f"full-probe-{index}",
                            _payload(next(tokens), PROBE_BLOCKS, PROBE_MAX_TOKENS),
                        )
                        for index in range(FULL_PROBES)
                    )
                )
            )
            full_pinned = tuple(await asyncio.gather(*full_tasks))

    return ArmResult(
        decode_worker_id=decode_id,
        pinned=pinned_outcomes,
        probe=probe,
        full_pinned=full_pinned,
        full_probes=full_probes,
    )


def _run_arm(
    *,
    request: pytest.FixtureRequest,
    frontend_port: int,
    namespace: str,
    config_name: str,
    tokens: itertools.count[int],
    all_full: bool,
) -> ArmResult:
    with FrontendRouterProcess(
        request,
        BLOCK_SIZE,
        frontend_port,
        namespace,
        store_backend="file",
        request_plane="tcp",
        event_plane="zmq",
        extra_env={"DYN_ROUTER_POLICY_CONFIG": str(CONFIG_DIR / config_name)},
    ):
        result = asyncio.run(
            _exercise(
                f"http://localhost:{frontend_port}/v1/completions",
                tokens,
                all_full=all_full,
            )
        )
    logger.info("%s: %s", config_name, result)
    return result


def _assert_pinned(outcomes: tuple[Outcome, ...], decode_id: int) -> None:
    for outcome in outcomes:
        rank = int(outcome.label.rsplit("@", 1)[1])
        assert (outcome.status, outcome.decode_worker_id, outcome.decode_dp_rank) == (
            200,
            decode_id,
            rank,
        ), outcome


@pytest.mark.parametrize("discovery_backend", ["file"], indirect=True)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
@pytest.mark.parametrize("event_plane", ["zmq"], indirect=True)
@pytest.mark.usefixtures(
    "runtime_services_dynamic_ports",
    "predownload_tokenizers",
)
def test_kv_capacity_aware_skips_full_decode_rank(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The default cost overfills a decode rank; kv-capacity-aware does not."""

    monkeypatch.setenv("DYN_FILE_KV", str(tmp_path / "discovery"))
    namespace = f"test-kv-capacity-{uuid.uuid4().hex[:10]}"
    frontend_ports = allocate_frontend_ports(request, 2)
    # One counter for both arms, so no two prompts in the test share a prefix.
    tokens = itertools.count(1000)
    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}
    decode_args = {
        **mocker_args,
        "dp_size": DECODE_DP_SIZE,
        "num_gpu_blocks": DECODE_RANK_BLOCKS,
    }

    with (
        DisaggMockerProcess(
            request,
            namespace=namespace,
            worker_type="prefill",
            mocker_args=mocker_args,
            store_backend="file",
            request_plane="tcp",
            event_plane="zmq",
        ),
        DisaggMockerProcess(
            request,
            namespace=namespace,
            worker_type="decode",
            mocker_args=decode_args,
            store_backend="file",
            request_plane="tcp",
            event_plane="zmq",
        ),
    ):
        default = _run_arm(
            request=request,
            frontend_port=frontend_ports[0],
            namespace=namespace,
            config_name="default_w1000.yaml",
            tokens=tokens,
            all_full=False,
        )
        kvca = _run_arm(
            request=request,
            frontend_port=frontend_ports[1],
            namespace=namespace,
            config_name="kv_capacity_aware.yaml",
            tokens=tokens,
            all_full=True,
        )

    _assert_pinned(default.pinned, default.decode_worker_id)
    _assert_pinned(kvca.pinned, kvca.decode_worker_id)

    # The default cost picks the rank with fewer requests, which the probe overfills.
    assert (default.probe.status, default.probe.decode_dp_rank) == (
        200,
        0,
    ), default.probe
    # The kv-capacity-aware policy skips that rank.
    assert (kvca.probe.status, kvca.probe.decode_dp_rank) == (200, 1), kvca.probe

    # When the probes fit on no rank, every request is still placed and served.
    _assert_pinned(kvca.full_pinned, kvca.decode_worker_id)
    assert len(kvca.full_probes) == FULL_PROBES
    for probe in kvca.full_probes:
        assert probe.status == 200, probe
