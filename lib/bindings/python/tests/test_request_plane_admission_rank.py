# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend admission for Python workers, through the built extension's real
ingress and gate, on both the pull and push egress paths.

A Python endpoint is a model backend once ``register_model`` records its role,
which it does before it publishes the model card, and only a model backend's
requests pass through the backend admission gate. These endpoints start serving
before they register, as a worker that serves and registers concurrently may:
until registration, an endpoint serves exactly as before and never touches the
gate, and from then on it admits every request. Requests that arrive before the
card reports the worker's ranks are admitted under the gate's fallbacks, which
the gate's own unit tests cover.

A Python worker's ingress decodes each unary request body straight into the
object its handler receives, recording the body's ``routing.dp_rank`` in the
same pass, and the gate admits the request under that rank without reading the
object back. These tests register the worker's DP ranks through its model card
as a backend does, and send bodies over both request-plane codecs. For every
body, the handler must receive it unchanged, and the gate must admit it under
the rank the handler reads from what it received: its label in the gate's own
``request_receive_total`` counter, scraped from the process's metrics endpoint,
is that rank when the worker serves it, and ``unspecified`` otherwise.

The bodies cover missing, null and malformed ranks, unrelated nesting, repeated
keys, and non-string and binary keys. A Python dict cannot hold a repeated key,
so a mapping that encodes its entries in order, repeated keys included, stands
in for any sender that puts them on the wire; the dict the handler receives
keeps the last entry for each key, and so must the gate.

The gate also reads what each response carries, as the Rust binding parses it
under the GIL it already holds, to end a request's wait at its first generated
token or completion response, whichever comes first, on every worker role. The
scripted streams below send one response at a time and check the gate's
occupancy after each one reaches the caller.

The binding classifies a response only while the gate still reads it: each up
to the first that ends its request's wait, and none for a request that arrived
while its endpoint was no model backend.

A Python worker's gate belongs to its runtime: every endpoint the runtime
serves admits through it, and a second runtime in the same process, a worker of
its own, has its own gate, whose family its own metrics endpoint serves under
its own `worker_id`.
Classifying takes the length of a response's `token_ids` once, so these tests
send it, and an equal `neutral_ids` that nothing classifies, as lists that
count how often their length is taken: the extra lengths taken of `token_ids`
are the responses classified, whatever encoding takes of both lists alike.
"""

import asyncio
import collections.abc
import contextlib
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import aiohttp
import pytest

from dynamo.llm import (
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    WorkerType,
    register_model,
)
from dynamo.runtime import DistributedRuntime
from tests.utils.constants import DefaultPort
from tests.utils.port_utils import reserved_ports

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
]

FAMILY = "dynamo_backend_admission_"
RECEIVED = "dynamo_backend_admission_request_receive_total"
ADMITTED = "dynamo_backend_admission_request_admit_total"
REJECTED = "dynamo_backend_admission_rejection_total"
ENGINE_REQUESTS = "dynamo_backend_admission_engine_request_count"
WAITING = "dynamo_backend_admission_engine_request_waiting_count"
DP_RANK_LABEL = re.compile(r'dp_rank="([^"]*)"')
WORKER_ID_LABEL = re.compile(r'worker_id="([^"]*)"')
UNSPECIFIED = "unspecified"

# A gate admitting one engine request, waiting until its first token, and
# queueing none, under the canonical names; the aliases are cleared so none can
# shadow them.
ONE_REQUEST_LIMITS = {
    "DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT": "1",
    "DYN_BACKEND_ADMISSION_ENGINE_REQUEST_WAITING_LIMIT": "1",
    "DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT": "0",
}
LIMIT_ALIASES = (
    "DYN_ENGINE_REQUEST_LIMIT",
    "DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT",
    "DYN_DYNAMO_REQUEST_QUEUE_LIMIT",
)

# The worker serves global ranks 4 through 7. Every rank a body names anywhere,
# decoys included, is one of them, so a rank read from the wrong place shows up
# as that rank's series instead of hiding under `unspecified`.
START_RANK = 4
RANK_COUNT = 4
SERVED = range(START_RANK, START_RANK + RANK_COUNT)

TIMEOUT_S = 10

# A fork can inherit an initialized Rust runtime without its worker threads.
# Execute each case in a fresh interpreter, even when pytest itself uses --forked.
_CHILD_CASE = "DYNAMO_ADMISSION_TEST_CHILD"


@pytest.fixture(scope="module", autouse=True)
def nats_and_etcd():
    """These file-discovery/TCP tests use ZMQ events and need no shared services."""


def _run_in_fresh_process(request, tmp_path) -> bool:
    """Run the selected case in a child; return False only inside that child."""
    __tracebackhide__ = True
    nodeid = request.node.nodeid
    if os.environ.get(_CHILD_CASE) == nodeid:
        return False

    env = {
        key: value
        for key, value in os.environ.items()
        if key != "PYTEST_ADDOPTS" and not key.startswith("PYTEST_XDIST_")
    }
    env[_CHILD_CASE] = nodeid
    report = tmp_path / "child-results.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "-p",
        "no:pytest_forked",
        "-p",
        "no:cacheprovider",
        "-p",
        "no:codeblocks",
        "-n",
        "0",
        "-v",
        "--tb=short",
        "-ra",
        f"--basetemp={tmp_path / 'child'}",
        f"--junitxml={report}",
        nodeid,
    ]
    try:
        result = subprocess.run(
            command,
            cwd=request.config.rootpath,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,  # Leave time for diagnostics before the outer 120s timeout.
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"Admission child timed out: {nodeid}\n{exc.stdout!r}\n{exc.stderr!r}",
            pytrace=False,
        )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    cases = ET.parse(report).findall(".//testcase")
    assert len(cases) == 1 and not any(
        case.find(tag) is not None
        for case in cases
        for tag in ("failure", "error", "skipped")
    ), output
    return True


class Entries:
    """A mapping that encodes its entries in order, repeated keys included."""

    def __init__(self, *entries):
        self._entries = list(entries)

    def __getitem__(self, key):
        for entry_key, value in reversed(self._entries):
            if entry_key == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._entries)

    def __len__(self):
        return len(self._entries)

    def keys(self):
        return [key for key, _ in self._entries]

    def values(self):
        return [value for _, value in self._entries]


# The encoder reads any mapping through `keys()` and `values()`.
collections.abc.Mapping.register(Entries)


def as_received(body):
    """The object a handler receives for `body`: what it encodes to, decoded
    into dicts, which keep the last entry for a repeated key."""
    if isinstance(body, collections.abc.Mapping):
        return {
            key: as_received(value) for key, value in zip(body.keys(), body.values())
        }
    if isinstance(body, (list, tuple)):
        return [as_received(value) for value in body]
    return body


def handler_rank(body):
    """The rank a handler reads: `(body.get("routing") or {}).get("dp_rank")`,
    when that is an integer (not a bool) that fits a u32."""
    routing = body.get("routing") if isinstance(body, dict) else None
    rank = routing.get("dp_rank") if isinstance(routing, dict) else None
    return rank if type(rank) is int and 0 <= rank < 2**32 else None


def admission_label(body) -> str:
    """The `dp_rank` label the gate must admit a received `body` under."""
    rank = handler_rank(body)
    return str(rank) if rank in SERVED else UNSPECIFIED


def _routing(dp_rank):
    return {"routing": {"dp_rank": dp_rank}}


# (case, body, msgpack only). Bodies with binary or non-string keys cannot be
# encoded as JSON, so they run on msgpack alone.
CASES = [
    (
        "request with routing hints",
        {
            "model": "m",
            "token_ids": [1, 2],
            "routing": {"backend_instance_id": 7, "dp_rank": 5, "prefill_dp_rank": 6},
        },
        False,
    ),
    ("first served rank", _routing(4), False),
    ("last served rank", _routing(7), False),
    ("below the served ranks", _routing(3), False),
    ("past the served ranks", _routing(8), False),
    ("no routing", {"model": "m"}, False),
    ("empty body", {}, False),
    ("null routing", {"routing": None}, False),
    ("empty routing", {"routing": {}}, False),
    ("null rank", _routing(None), False),
    ("string rank", _routing("5"), False),
    ("float rank", _routing(5.0), False),
    ("bool rank", _routing(True), False),
    ("negative rank", _routing(-1), False),
    ("rank past u32", _routing(2**32 + 5), False),
    ("list rank", _routing([5]), False),
    ("list routing", {"routing": [{"dp_rank": 5}]}, False),
    ("top-level dp_rank", {"dp_rank": 5}, False),
    ("prefill rank alone", {"routing": {"prefill_dp_rank": 5}}, False),
    ("nested routing", {"nested": {"routing": {"dp_rank": 5}}}, False),
    ("nested rank", {"routing": {"nested": {"dp_rank": 5}}}, False),
    (
        "rank beside nested decoys",
        {
            "nested": {"routing": {"dp_rank": 7}},
            "routing": {"dp_rank": 6, "routing": {"dp_rank": 5}},
        },
        False,
    ),
    ("list body", [_routing(5)], False),
    ("string body", "routing", False),
    (
        "repeated routing",
        Entries(("routing", {"dp_rank": 4}), ("routing", {"dp_rank": 5})),
        False,
    ),
    (
        "repeated routing ending null",
        Entries(("routing", {"dp_rank": 5}), ("routing", None)),
        False,
    ),
    (
        "repeated routing ending without a rank",
        Entries(("routing", {"dp_rank": 5}), ("model", "m"), ("routing", {})),
        False,
    ),
    (
        "repeated rank ending null",
        {"routing": Entries(("dp_rank", 5), ("dp_rank", None))},
        False,
    ),
    (
        "repeated rank ending valid",
        {"routing": Entries(("dp_rank", None), ("dp_rank", 6))},
        False,
    ),
    (
        "repeated rank ending malformed",
        {"routing": Entries(("dp_rank", 6), ("dp_rank", "6"))},
        False,
    ),
    (
        "non-string keys",
        {1: "one", b"\x00\xff": b"\x01", "routing": {2: b"\x00", "dp_rank": 6}},
        True,
    ),
    ("binary routing key", {b"routing": {"dp_rank": 5}}, True),
    ("binary rank key", {"routing": {b"dp_rank": 5}}, True),
    (
        "binary routing key after the string one",
        Entries(("routing", {"dp_rank": 4}), (b"routing", {"dp_rank": 5})),
        True,
    ),
]


class Handlers:
    """Pull and push handlers that keep what they receive."""

    def __init__(self) -> None:
        self.received: list = []

    async def pull(self, request, context):
        self.received.append(request)
        yield {"ok": True}

    async def push(self, request, context, response_sender):
        self.received.append(request)
        response_sender.send({"ok": True})
        response_sender.close()
        return
        yield  # an async generator, as push handlers are


class Gate:
    """The gate's own metrics, scraped from the process's metrics endpoint."""

    def __init__(self, session: aiohttp.ClientSession, port: int) -> None:
        self._session = session
        self._url = f"http://127.0.0.1:{port}/metrics"

    async def by_rank(self, metric: str) -> dict[str, float]:
        """One metric's samples, summed by `dp_rank`."""
        async with self._session.get(self._url) as response:
            text = await response.text()
            assert response.status == 200, text
        counts: dict[str, float] = {}
        for line in text.splitlines():
            if line.startswith(f"{metric}{{"):
                label = DP_RANK_LABEL.search(line)
                assert label, line
                counts[label.group(1)] = counts.get(label.group(1), 0) + float(
                    line.rpartition(" ")[2]
                )
        return counts

    async def receives(self) -> dict[str, float]:
        return await self.by_rank(RECEIVED)

    async def total(self, metric: str) -> float:
        return sum((await self.by_rank(metric)).values())

    async def worker_ids(self) -> set[str]:
        """The `worker_id` of every series of the gate's family served here."""
        async with self._session.get(self._url) as response:
            text = await response.text()
            assert response.status == 200, text
        worker_ids = set()
        for line in text.splitlines():
            if line.startswith(FAMILY):
                label = WORKER_ID_LABEL.search(line)
                assert label, line
                worker_ids.add(label.group(1))
        return worker_ids

    async def occupancy(self) -> tuple[float, float]:
        """The worker's (engine requests, engine-waiting requests)."""
        return (
            sum((await self.by_rank(ENGINE_REQUESTS)).values()),
            sum((await self.by_rank(WAITING)).values()),
        )

    async def occupancy_reaches(self, expected: tuple[float, float]) -> None:
        deadline = time.monotonic() + TIMEOUT_S
        while (occupancy := await self.occupancy()) != expected:
            assert time.monotonic() < deadline, f"stuck at {occupancy}, not {expected}"
            await asyncio.sleep(0.01)

    async def ready(self) -> None:
        """Wait out the metrics server starting."""
        deadline = time.monotonic() + TIMEOUT_S
        while True:
            try:
                await self.receives()
                return
            except aiohttp.ClientConnectionError:
                assert time.monotonic() < deadline, "metrics endpoint never came up"
                await asyncio.sleep(0.05)


@contextlib.asynccontextmanager
async def _runtimes(tmp_path, codec: str, count: int):
    """`count` runtimes in this one process — each a Python worker, with a
    metrics endpoint of its own — and a session to scrape them."""
    # The system port must fit the runtime's port range, which the shared
    # allocator respects; an arbitrary ephemeral port may not.
    with reserved_ports(count, DefaultPort.SYSTEM1.value) as ports:
        os.environ["DYN_FILE_KV"] = str(tmp_path)
        os.environ["DYN_REQUEST_PLANE_CODEC"] = codec
        runtimes: list = []
        servers: list = []
        try:
            for port in ports:
                # Each runtime reads its system port as it is created.
                os.environ["DYN_SYSTEM_PORT"] = str(port)
                runtimes.append(
                    DistributedRuntime(
                        asyncio.get_running_loop(), "file", "tcp", event_plane="zmq"
                    )
                )
            async with aiohttp.ClientSession() as session:
                yield runtimes, servers, [Gate(session, port) for port in ports]
        finally:
            for server in servers:
                server.cancel()
            for server in servers:
                with contextlib.suppress(asyncio.CancelledError):
                    await server
            for runtime in runtimes:
                runtime.shutdown()


@contextlib.asynccontextmanager
async def _runtime(tmp_path, codec: str):
    """A runtime with a metrics endpoint, and a session to scrape it."""
    async with _runtimes(tmp_path, codec, 1) as (runtimes, servers, gates):
        yield runtimes[0], servers, gates[0]


def _admit_one_request_per_worker() -> None:
    """Configure every gate created from now on to admit one engine request,
    waiting until its first token, and to queue none."""
    for alias in LIMIT_ALIASES:
        os.environ.pop(alias, None)
    os.environ.update(ONE_REQUEST_LIMITS)


async def _refused_as_overloaded(client, body) -> None:
    """Send `body`, which the worker's gate must refuse as an overload."""
    with pytest.raises(Exception, match="(?i)overload"):
        stream = await client.generate(body)
        async for _ in stream:
            pass


async def _serve(runtime, servers, name, handler):
    """Serve `handler` on endpoint `name`, and return the endpoint and a client
    that has found it. It is no model backend until it registers."""
    endpoint = runtime.endpoint(f"admission-rank.backend.{name}")
    servers.append(asyncio.ensure_future(endpoint.serve_endpoint(handler)))
    client = await endpoint.client()
    await asyncio.wait_for(client.wait_for_instances(), TIMEOUT_S)
    return endpoint, client


async def _register(endpoint, name, worker_type, runtime_config=None):
    """Register model `name` on `endpoint` as a `worker_type` backend, without
    model assets. A prefill worker exposes no OpenAI surface, and a
    disaggregated worker names the peer it needs."""
    prefill = worker_type == WorkerType.Prefill
    await register_model(
        ModelInput.Tokens,
        ModelType.Prefill if prefill else ModelType.Chat,
        endpoint,
        name,
        runtime_config=runtime_config,
        worker_type=worker_type,
        needs=(
            [[WorkerType.Decode if prefill else WorkerType.Prefill]]
            if worker_type != WorkerType.Aggregated
            else None
        ),
        skip_model_assets=True,
    )


async def _admitted_under(client, handlers, gate, body) -> tuple[object, str]:
    """Send `body`, and return what the handler received and the one `dp_rank`
    the gate counted it under."""
    before = await gate.receives()
    stream = await client.generate(body)
    responses = [response.data() async for response in stream]
    assert responses == [{"ok": True}]
    received = handlers.received[-1]
    after = await gate.receives()
    counted = {
        label: after[label] - before.get(label, 0)
        for label in after
        if after[label] != before.get(label, 0)
    }
    assert len(counted) == 1 and list(counted.values()) == [1], counted
    return received, next(iter(counted))


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize("codec", ["msgpack", "json"])
async def test_the_gate_admits_python_requests_under_the_rank_their_body_names(
    request, tmp_path, codec
):
    if _run_in_fresh_process(request, tmp_path):
        return

    async with _runtime(tmp_path, codec) as (runtime, servers, gate):
        handlers = Handlers()
        endpoints = {}
        clients = {}
        for path in ("pull", "push"):
            endpoints[path], clients[path] = await _serve(
                runtime, servers, path, getattr(handlers, path)
            )
        # An endpoint that never registers is no model backend: it serves
        # exactly as before, and never touches the gate.
        _, generic = await _serve(runtime, servers, "generic", handlers.pull)
        await gate.ready()

        # Until they register, the backend endpoints serve as generic ones do.
        for path, client in clients.items():
            stream = await client.generate(_routing(5))
            assert [response.data() async for response in stream] == [{"ok": True}]
            assert handlers.received[-1] == _routing(5), path

        # Registration records each endpoint's role, and its card reports the
        # worker's ranks; the endpoints already serving admit from now on.
        runtime_config = ModelRuntimeConfig()
        runtime_config.data_parallel_start_rank = START_RANK
        runtime_config.data_parallel_size = RANK_COUNT
        for path, endpoint in endpoints.items():
            await _register(
                endpoint,
                f"admission-rank-{path}",
                WorkerType.Aggregated,
                runtime_config,
            )
        before = await gate.receives()
        assert sum(before.values()) == 0, f"admitted before registration: {before}"

        stream = await generic.generate(_routing(5))
        assert [response.data() async for response in stream] == [{"ok": True}]
        assert handlers.received[-1] == _routing(5)
        assert await gate.receives() == before, "a generic endpoint is never admitted"

        checked = 0
        for name, body, msgpack_only in CASES:
            if msgpack_only and codec != "msgpack":
                continue
            for path, client in clients.items():
                received, label = await _admitted_under(client, handlers, gate, body)
                expected = as_received(body)
                assert received == expected, f"{name} via {path}: {received!r}"
                assert label == admission_label(received), (
                    f"{name} via {path}: admitted under {label}, "
                    f"handler reads {handler_rank(received)!r}"
                )
                checked += 1
        assert checked >= 2 * 28
        # Every served rank and `unspecified` was exercised.
        counts = await gate.receives()
        assert {str(rank) for rank in SERVED} | {UNSPECIFIED} <= set(counts), counts


class Scripted:
    """Pull and push handlers that send a script of responses one at a time,
    each once the test releases it, then end once released again. A script
    entry that is an exception is raised instead of sent."""

    def __init__(self, script) -> None:
        self.script = script
        self.steps: asyncio.Queue = asyncio.Queue()

    def release(self) -> None:
        self.steps.put_nowait(None)

    async def pull(self, request, context):
        for entry in self.script:
            await self.steps.get()
            if isinstance(entry, Exception):
                raise entry
            yield entry
        await self.steps.get()

    async def push(self, request, context, response_sender):
        for entry in self.script:
            await self.steps.get()
            if isinstance(entry, Exception):
                raise entry
            response_sender.send(entry)
        await self.steps.get()
        response_sender.close()
        return
        yield  # an async generator, as push handlers are


BOOTSTRAP = {"bootstrap_host": "127.0.0.1", "bootstrap_port": 1, "bootstrap_room": 7}
USAGE = {"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3}

# (case, worker type, script, process occupancy as (engine requests, waiting)
# after each response reaches the caller). Each request holds both kinds from
# admission; its stream's end releases whatever it still holds.
SCRIPTS = [
    (
        "decode: metadata, then its first token, then more",
        WorkerType.Aggregated,
        [
            {"token_ids": [], "disaggregated_params": BOOTSTRAP},
            {"token_ids": [], "engine_data": {"kv_cache_hit": {"tokens": 1}}},
            {"index": 0, "token_ids": [11]},
            {
                "index": 0,
                "token_ids": [12],
                "finish_reason": "stop",
                "completion_usage": USAGE,
            },
        ],
        [(1, 1), (1, 1), (1, 0), (1, 0)],
    ),
    (
        "decode: a completion without a token ends the wait, held open",
        WorkerType.Decode,
        [
            {"token_ids": None, "text": "metadata"},
            {"token_ids": [], "finish_reason": "length", "completion_usage": USAGE},
        ],
        [(1, 1), (1, 0)],
    ),
    (
        "text: a role-only delta, then its generated text, held open",
        WorkerType.Aggregated,
        [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
            {"choices": [{"index": 0, "delta": {"content": "hi"}}]},
        ],
        [(1, 1), (1, 0)],
    ),
    (
        "text: a choice's finish reason completes without a token, held open",
        WorkerType.Decode,
        [
            {"choices": [], "usage": USAGE},
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": "length",
                    }
                ]
            },
        ],
        [(1, 1), (1, 0)],
    ),
    (
        "text: a later choice's generated text ends the wait, held open",
        WorkerType.Aggregated,
        [
            {
                "choices": [
                    {"index": 0, "delta": {"content": ""}},
                    {"index": 1, "delta": {"content": "hi"}},
                ]
            },
        ],
        [(1, 0)],
    ),
    (
        "decode: a payload carrying no generated output is no token",
        WorkerType.Aggregated,
        [
            {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ]
            },
            {"choices": [], "usage": USAGE},
            # A delta that is not a mapping, and content that is not a string.
            {"choices": [{"index": 0, "delta": "hi"}]},
            {"choices": [{"index": 0, "delta": {"content": ["hi"]}}]},
            {},
            "a string",
            ["a", "list"],
        ],
        [(1, 1)] * 8,
    ),
    (
        "prefill: a bootstrap announcement, then a token-less completion, held open",
        WorkerType.Prefill,
        [
            {
                "token_ids": [],
                "text": None,
                "finish_reason": None,
                "disaggregated_params": BOOTSTRAP,
            },
            {"token_ids": [], "disaggregated_params": None, "completion_usage": USAGE},
        ],
        [(1, 1), (1, 0)],
    ),
    (
        "prefill: its token ends the wait before its completion response",
        WorkerType.Prefill,
        [
            {"token_ids": [11], "disaggregated_params": "handoff"},
            {
                "token_ids": [],
                "finish_reason": "length",
                "disaggregated_params": "handoff",
            },
        ],
        [(1, 0), (1, 0)],
    ),
    (
        "prefill: usage without a handoff completes nothing; it ends at its end",
        WorkerType.Prefill,
        [
            {"token_ids": [], "disaggregated_params": BOOTSTRAP},
            {"token_ids": [], "completion_usage": USAGE},
            {"token_ids": [], "engine_data": {"kv_cache_hit": {"tokens": 1}}},
        ],
        [(1, 1), (1, 1), (1, 1)],
    ),
]


async def _next(stream):
    return (await asyncio.wait_for(stream.__anext__(), TIMEOUT_S)).data()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize("codec", ["msgpack", "json"])
@pytest.mark.parametrize("path", ["pull", "push"])
@pytest.mark.parametrize(
    "case,worker_type,script,occupancy", SCRIPTS, ids=[case for case, *_ in SCRIPTS]
)
async def test_the_response_that_ends_the_wait(
    request, tmp_path, codec, path, case, worker_type, script, occupancy
):
    if _run_in_fresh_process(request, tmp_path):
        return

    async with _runtime(tmp_path, codec) as (runtime, servers, gate):
        scripted = Scripted(script)
        endpoint, client = await _serve(runtime, servers, path, getattr(scripted, path))
        await _register(endpoint, "admission-rank-model", worker_type)
        await gate.ready()

        stream = await client.generate({"token_ids": [1, 2, 3]})
        assert await gate.occupancy() == (1, 1), "obtaining the stream is no response"
        for entry, expected in zip(script, occupancy):
            scripted.release()
            assert await _next(stream) == entry, "responses arrive unchanged"
            assert await gate.occupancy() == expected, entry
        # The stream stays open until the handler ends it.
        await asyncio.sleep(0.05)
        assert await gate.occupancy() == occupancy[-1]
        scripted.release()
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(stream.__anext__(), TIMEOUT_S)
        await gate.occupancy_reaches((0, 0))
        assert sum((await gate.by_rank(ADMITTED)).values()) == 1


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize("path", ["pull", "push"])
async def test_a_handler_failing_after_metadata_releases_both_kinds_once(
    request, tmp_path, path
):
    if _run_in_fresh_process(request, tmp_path):
        return

    async with _runtime(tmp_path, "msgpack") as (runtime, servers, gate):
        failing = Scripted(
            [
                {"token_ids": [], "disaggregated_params": BOOTSTRAP},
                RuntimeError("engine failed before its first token"),
            ]
        )
        endpoint, client = await _serve(runtime, servers, path, getattr(failing, path))
        await _register(endpoint, "admission-rank-model", WorkerType.Decode)
        await gate.ready()

        stream = await client.generate({"token_ids": [1]})
        failing.release()
        assert await _next(stream) == {
            "token_ids": [],
            "disaggregated_params": BOOTSTRAP,
        }
        assert await gate.occupancy() == (1, 1)
        failing.release()
        with pytest.raises(Exception, match="engine failed before its first token"):
            async for _ in stream:
                pass
        await gate.occupancy_reaches((0, 0))
        # Released exactly once: nothing is left to go negative.
        await asyncio.sleep(0.05)
        assert await gate.occupancy() == (0, 0)
        assert sum((await gate.by_rank(ADMITTED)).values()) == 1


class LengthCounted(list):
    """A list that counts how often its length is taken."""

    def __init__(self, items=()):
        super().__init__(items)
        self.lengths = 0

    def __len__(self):
        self.lengths += 1
        return super().__len__()


def as_sent(response):
    """`response` as a handler sends it: its `token_ids`, and a `neutral_ids`
    copy that nothing classifies, as lists that count their lengths taken."""
    return {
        **response,
        "token_ids": LengthCounted(response["token_ids"]),
        "neutral_ids": LengthCounted(response["token_ids"]),
    }


def as_delivered(response):
    """What the caller receives for `as_sent(response)`."""
    return {**response, "neutral_ids": response["token_ids"]}


def classified(sent) -> int:
    """How many of the responses `sent` the binding classified: it takes a
    response's `token_ids` length once to classify it, on top of whatever
    encoding takes of both lists alike."""
    return sum(
        response["token_ids"].lengths - response["neutral_ids"].lengths
        for response in sent
    )


async def _until(condition) -> None:
    deadline = time.monotonic() + TIMEOUT_S
    while not condition():
        assert time.monotonic() < deadline, "the handler never got there"
        await asyncio.sleep(0.01)


async def _ends(stream) -> None:
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(stream.__anext__(), TIMEOUT_S)


# (case, worker type, or None for an endpoint that never registers; its
# responses; and the index of the one that ends the wait, or None when the
# endpoint admits nothing).
CLASSIFIED = [
    (
        "decode",
        WorkerType.Decode,
        [
            {"token_ids": [], "disaggregated_params": BOOTSTRAP},
            {"token_ids": [11]},
            {"token_ids": [12]},
            {"token_ids": [13], "finish_reason": "stop", "completion_usage": USAGE},
        ],
        1,
    ),
    (
        "prefill token before its completion",
        WorkerType.Prefill,
        [
            {"token_ids": [], "disaggregated_params": BOOTSTRAP},
            {"token_ids": [11], "disaggregated_params": "handoff"},
            {
                "token_ids": [],
                "finish_reason": "length",
                "disaggregated_params": "handoff",
            },
            {"token_ids": [], "engine_data": {"kv_cache_hit": {"tokens": 1}}},
        ],
        1,
    ),
    (
        "completion first",
        WorkerType.Decode,
        [
            {"token_ids": [], "disaggregated_params": BOOTSTRAP},
            {"token_ids": [], "finish_reason": "length", "completion_usage": USAGE},
            {"token_ids": [11]},
        ],
        1,
    ),
    (
        "no model backend",
        None,
        [
            {"token_ids": []},
            {"token_ids": [11]},
            {"token_ids": [12], "finish_reason": "stop"},
        ],
        None,
    ),
]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize("path", ["pull", "push"])
@pytest.mark.parametrize(
    "case,worker_type,responses,decisive",
    CLASSIFIED,
    ids=[case for case, *_ in CLASSIFIED],
)
async def test_responses_are_classified_only_until_one_ends_the_wait(
    request, tmp_path, path, case, worker_type, responses, decisive
):
    if _run_in_fresh_process(request, tmp_path):
        return

    async with _runtime(tmp_path, "msgpack") as (runtime, servers, gate):
        sent = [as_sent(response) for response in responses]
        scripted = Scripted(sent)
        endpoint, client = await _serve(runtime, servers, path, getattr(scripted, path))
        if worker_type is not None:
            await _register(endpoint, "admission-rank-model", worker_type)
        await gate.ready()

        stream = await client.generate({"token_ids": [1]})
        for index, response in enumerate(responses):
            scripted.release()
            assert await _next(stream) == as_delivered(response)
            expected = 0 if decisive is None else min(index, decisive) + 1
            assert classified(sent) == expected, f"{case} via {path}, response {index}"
            if decisive is not None:
                assert await gate.occupancy() == (
                    (1, 1) if index < decisive else (1, 0)
                )
        scripted.release()
        await _ends(stream)
        if decisive is None:
            assert sum((await gate.receives()).values()) == 0, "never admitted"
        else:
            await gate.occupancy_reaches((0, 0))


class FailingToken:
    """A push handler whose first token cannot be encoded, as it holds a value
    no codec carries, and which carries on, as a handler may, with a valid
    token and then another. Each send waits for the test to release it."""

    def __init__(self) -> None:
        self.steps: asyncio.Queue = asyncio.Queue()
        failing = as_sent({"token_ids": [11]})
        failing["unencodable"] = object()
        self.sent = [
            as_sent({"token_ids": [], "disaggregated_params": BOOTSTRAP}),
            failing,
            as_sent({"token_ids": [12]}),
            as_sent({"token_ids": [13]}),
        ]
        self.failures: list = []

    def release(self) -> None:
        self.steps.put_nowait(None)

    async def push(self, request, context, response_sender):
        for response in self.sent:
            await self.steps.get()
            try:
                response_sender.send(response)
            except ValueError as error:
                self.failures.append(error)
        await self.steps.get()
        response_sender.close()
        return
        yield  # an async generator, as push handlers are


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_token_that_fails_to_encode_leaves_the_wait_to_the_next(
    request, tmp_path
):
    if _run_in_fresh_process(request, tmp_path):
        return

    async with _runtime(tmp_path, "msgpack") as (runtime, servers, gate):
        handler = FailingToken()
        endpoint, client = await _serve(runtime, servers, "push", handler.push)
        await _register(endpoint, "admission-rank-model", WorkerType.Decode)
        await gate.ready()

        stream = await client.generate({"token_ids": [1]})
        handler.release()
        assert await _next(stream) == as_delivered(
            {"token_ids": [], "disaggregated_params": BOOTSTRAP}
        )
        assert (classified(handler.sent), await gate.occupancy()) == (1, (1, 1))

        # Classified as a token, then never encoded: it never reaches the gate,
        # so the request still waits, and the next response is classified.
        handler.release()
        await _until(lambda: handler.failures)
        assert "failed serializing" in str(handler.failures[0])
        assert (classified(handler.sent), await gate.occupancy()) == (2, (1, 1))

        handler.release()
        assert await _next(stream) == as_delivered({"token_ids": [12]})
        assert (classified(handler.sent), await gate.occupancy()) == (3, (1, 0))

        handler.release()
        assert await _next(stream) == as_delivered({"token_ids": [13]})
        assert classified(handler.sent) == 3, "none after the token that ended it"

        handler.release()
        await _ends(stream)
        await gate.occupancy_reaches((0, 0))


class Ahead:
    """A push handler that sends every response at once, ahead of the gate,
    then holds its stream open until released."""

    def __init__(self, sent) -> None:
        self.sent = sent
        self.steps: asyncio.Queue = asyncio.Queue()

    async def push(self, request, context, response_sender):
        for response in self.sent:
            response_sender.send(response)
        await self.steps.get()
        response_sender.close()
        return
        yield  # an async generator, as push handlers are


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_push_handler_ahead_of_the_gate_keeps_the_decisive_frame(
    request, tmp_path
):
    if _run_in_fresh_process(request, tmp_path):
        return

    async with _runtime(tmp_path, "msgpack") as (runtime, servers, gate):
        responses = [
            {"token_ids": [], "disaggregated_params": BOOTSTRAP},
            {"token_ids": [11]},
            {"token_ids": [12]},
            {"token_ids": [13]},
        ]
        handler = Ahead([as_sent(response) for response in responses])
        endpoint, client = await _serve(runtime, servers, "push", handler.push)
        await _register(endpoint, "admission-rank-model", WorkerType.Decode)
        await gate.ready()

        stream = await client.generate({"token_ids": [1]})
        received = [await _next(stream) for _ in responses]
        assert received == [as_delivered(response) for response in responses]
        assert classified(handler.sent) == 2, "through the first token, none after"
        assert await gate.occupancy() == (1, 0), "its first token ended the wait"

        handler.steps.put_nowait(None)
        await _ends(stream)
        await gate.occupancy_reaches((0, 0))


class Reentering(list):
    """`token_ids` that, the first time its length is taken, runs `reenter`:
    Python that encoding the response holding it runs, as any in it can."""

    def __init__(self, items, reenter):
        self.reenter = reenter
        super().__init__(items)

    def __len__(self):
        reenter, self.reenter = self.reenter, None
        if reenter is not None:
            reenter()
        return super().__len__()


class ReenteringHandler:
    """A push handler whose first token, as it is classified, sends another
    response from inside that `send`."""

    def __init__(self) -> None:
        self.steps: asyncio.Queue = asyncio.Queue()
        self.later = as_sent({"token_ids": [12]})

    async def push(self, request, context, response_sender):
        await self.steps.get()
        token = Reentering(
            [11],
            lambda: response_sender.send({"token_ids": [], "note": "re-entered"}),
        )
        response_sender.send({"token_ids": token})
        response_sender.send(self.later)
        await self.steps.get()
        response_sender.close()
        return
        yield  # an async generator, as push handlers are


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_send_reentered_while_a_token_is_classified(request, tmp_path):
    if _run_in_fresh_process(request, tmp_path):
        return

    async with _runtime(tmp_path, "msgpack") as (runtime, servers, gate):
        handler = ReenteringHandler()
        endpoint, client = await _serve(runtime, servers, "push", handler.push)
        await _register(endpoint, "admission-rank-model", WorkerType.Decode)
        await gate.ready()

        stream = await client.generate({"token_ids": [1]})
        handler.steps.put_nowait(None)
        # The re-entrant response is accepted first, then the token whose
        # classification sent it, which ends the wait; the one after both is
        # behind it, and unclassified.
        assert await _next(stream) == {"token_ids": [], "note": "re-entered"}
        assert await _next(stream) == {"token_ids": [11]}
        assert await _next(stream) == as_delivered({"token_ids": [12]})
        assert classified([handler.later]) == 0
        assert await gate.occupancy() == (1, 0)

        handler.steps.put_nowait(None)
        await _ends(stream)
        await gate.occupancy_reaches((0, 0))


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize("path", ["pull", "push"])
async def test_a_request_keeps_the_role_its_endpoint_had_when_it_arrived(
    request, tmp_path, path
):
    if _run_in_fresh_process(request, tmp_path):
        return

    async with _runtime(tmp_path, "msgpack") as (runtime, servers, gate):
        responses = [
            {"token_ids": []},
            {"token_ids": [11]},
            {"token_ids": [12], "finish_reason": "stop"},
        ]
        sent = [as_sent(response) for response in responses]
        scripted = Scripted(sent)
        endpoint, client = await _serve(runtime, servers, path, getattr(scripted, path))
        await gate.ready()

        # It arrives while the endpoint is no model backend, which the endpoint
        # becomes mid-request: it stays unadmitted and unclassified to its end.
        stream = await client.generate({"token_ids": [1]})
        scripted.release()
        assert await _next(stream) == as_delivered(responses[0])
        await _register(endpoint, "admission-rank-model", WorkerType.Decode)
        for response in responses[1:]:
            scripted.release()
            assert await _next(stream) == as_delivered(response)
        scripted.release()
        await _ends(stream)
        assert classified(sent) == 0
        assert sum((await gate.receives()).values()) == 0
        assert await gate.occupancy() == (0, 0)

        # The next request arrives at a model backend: admitted, and classified
        # through its first token.
        sent[:] = [as_sent(response) for response in responses]
        stream = await client.generate({"token_ids": [1]})
        for index, response in enumerate(responses):
            scripted.release()
            assert await _next(stream) == as_delivered(response)
            assert classified(sent) == min(index, 1) + 1, f"after response {index}"
        scripted.release()
        await _ends(stream)
        assert sum((await gate.receives()).values()) == 1
        await gate.occupancy_reaches((0, 0))


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize("path", ["pull", "push"])
async def test_python_workers_in_one_process_admit_through_their_own_gates(
    request, tmp_path, path
):
    """Two runtimes in one process are two Python workers, each with a gate of
    one engine request: one worker's full gate never refuses the other's work,
    and each worker's metrics endpoint serves only its own gate's family."""
    if _run_in_fresh_process(request, tmp_path):
        return

    _admit_one_request_per_worker()
    async with _runtimes(tmp_path, "msgpack", 2) as (runtimes, servers, gates):
        scripted = [Scripted([{"token_ids": [11]}]) for _ in runtimes]
        clients = []
        for index, (runtime, handler) in enumerate(zip(runtimes, scripted)):
            endpoint, client = await _serve(
                runtime, servers, f"{path}-{index}", getattr(handler, path)
            )
            await _register(endpoint, f"admission-model-{index}", WorkerType.Aggregated)
            clients.append(client)
        for gate in gates:
            await gate.ready()
        worker_ids = [await gate.worker_ids() for gate in gates]
        assert [len(ids) for ids in worker_ids] == [1, 1], worker_ids
        assert worker_ids[0] != worker_ids[1], "each worker labels its own family"

        # The first worker's gate fills up.
        held = await clients[0].generate({"token_ids": [1]})
        scripted[0].release()
        assert await _next(held) == {"token_ids": [11]}
        assert await gates[0].occupancy() == (1, 0)

        # The second worker admits through its own gate all the same.
        other = await clients[1].generate({"token_ids": [1]})
        assert await gates[1].occupancy() == (1, 1)
        scripted[1].release()
        assert await _next(other) == {"token_ids": [11]}
        assert await gates[1].occupancy() == (1, 0)
        assert await gates[0].occupancy() == (1, 0)

        # Each gate is full now, and refuses only its own worker's next request.
        for full, (client, gate) in enumerate(zip(clients, gates)):
            await _refused_as_overloaded(client, {"token_ids": [1]})
            assert [await each.total(REJECTED) for each in gates] == [
                float(index <= full) for index in range(len(gates))
            ]
        for gate in gates:
            assert await gate.total(RECEIVED) == 2
            assert await gate.total(ADMITTED) == 1

        for handler, stream in zip(scripted, (held, other)):
            handler.release()
            await _ends(stream)
        for gate in gates:
            await gate.occupancy_reaches((0, 0))


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_endpoints_of_one_python_worker_share_its_gate(request, tmp_path):
    """A runtime's endpoints serve one Python worker, so they admit through its
    one gate, whichever egress path they take: one endpoint's request fills it
    for the other's, until that request ends."""
    if _run_in_fresh_process(request, tmp_path):
        return

    _admit_one_request_per_worker()
    async with _runtime(tmp_path, "msgpack") as (runtime, servers, gate):
        scripted = {path: Scripted([{"token_ids": [11]}]) for path in ("pull", "push")}
        endpoints = {}
        clients = {}
        for path, handler in scripted.items():
            endpoints[path], clients[path] = await _serve(
                runtime, servers, path, getattr(handler, path)
            )
            await _register(
                endpoints[path], f"admission-model-{path}", WorkerType.Aggregated
            )
        await gate.ready()
        assert len(await gate.worker_ids()) == 1

        held = await clients["pull"].generate({"token_ids": [1]})
        scripted["pull"].release()
        assert await _next(held) == {"token_ids": [11]}
        assert await gate.occupancy() == (1, 0)

        # The push endpoint's request finds the worker's one gate full.
        await _refused_as_overloaded(clients["push"], {"token_ids": [1]})
        assert await gate.total(REJECTED) == 1

        # Once the pull request ends, the push endpoint admits. The refusal
        # leased the worker out of the refused client's own routing for a
        # moment, so a fresh client of the endpoint sends the request.
        scripted["pull"].release()
        await _ends(held)
        await gate.occupancy_reaches((0, 0))
        fresh = await endpoints["push"].client()
        await asyncio.wait_for(fresh.wait_for_instances(), TIMEOUT_S)
        stream = await fresh.generate({"token_ids": [1]})
        assert await gate.occupancy() == (1, 1)
        scripted["push"].release()
        assert await _next(stream) == {"token_ids": [11]}
        scripted["push"].release()
        await _ends(stream)
        await gate.occupancy_reaches((0, 0))
        assert await gate.total(RECEIVED) == 3
        assert await gate.total(ADMITTED) == 2
