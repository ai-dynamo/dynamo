# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Relay source selection and HTTP state contract, without external services."""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from tests.utils.constants import DynamoPortRange
from tests.utils.port_utils import allocate_port, deallocate_port

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.asyncio,
    # Keep the process-global native runtime out of the parent of later forked tests.
    pytest.mark.forked,
    pytest.mark.timeout(30),
]


def write_sources(path: Path, namespaces: list[str], **fields) -> None:
    document = {
        "version": 1,
        "sources": [{"namespace": name} for name in namespaces],
        **fields,
    }
    staged = path.with_suffix(".next")
    staged.write_text(json.dumps(document))
    staged.replace(path)


@pytest.fixture
def sources_file(tmp_path):
    path = tmp_path / "sources.json"
    write_sources(path, [])
    return path


@pytest.fixture
def system_port(request, monkeypatch):
    port = allocate_port(DynamoPortRange.ROUTER) if request.param else -1
    monkeypatch.setenv("DYN_SYSTEM_PORT", str(port))
    monkeypatch.delenv("POD_UID", raising=False)
    try:
        yield port
    finally:
        if port != -1:
            deallocate_port(port)


@pytest_asyncio.fixture
async def runtime(system_port):
    pytest.importorskip("dynamo._core", reason="requires the native Dynamo extension")
    from dynamo.runtime import DistributedRuntime

    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "mem", "tcp", event_plane="zmq"
    )
    try:
        yield runtime
    finally:
        runtime.shutdown()


@pytest.fixture
def relay_factory(runtime):
    from dynamo.llm import KvDcRelay

    @asynccontextmanager
    async def start(*, bind="127.0.0.1:0", **kwargs):
        relay = KvDcRelay(
            runtime.endpoint("test.relay.control"),
            "test-dc",
            bind=bind,
            **kwargs,
        )
        await relay.start()
        try:
            yield relay
        finally:
            await relay.shutdown()

    return start


async def wait_for_sources(relay, predicate):
    last = None
    for _ in range(150):
        last = (await relay.health())["sources"]
        if predicate(last):
            return last
        await asyncio.sleep(0.1)
    pytest.fail(f"Relay sources did not converge: {last}")


@pytest.mark.parametrize("system_port", [True], indirect=True)
@pytest.mark.parametrize(
    "mode,bind",
    [("discovery", None), ("from-file", "127.0.0.1:0")],
    ids=["discovery-wan-disabled", "from-file-wan-enabled"],
)
async def test_http_state(
    relay_factory, sources_file, system_port, monkeypatch, mode, bind
):
    monkeypatch.setenv("POD_UID", "test-pod")
    options = {}
    if mode == "from-file":
        write_sources(sources_file, [], connectionRevision="revision-a")
        options = {
            "sources_file": str(sources_file),
            "connection_revision": "revision-a",
        }

    async with relay_factory(bind=bind, **options) as relay:
        async with httpx.AsyncClient(timeout=1) as client:
            url = f"http://127.0.0.1:{system_port}/engine/state"
            for _ in range(100):
                try:
                    response = await client.get(url)
                except httpx.ConnectError:
                    await asyncio.sleep(0.1)
                    continue
                response.raise_for_status()
                state = response.json()
                if state["ready"]:
                    break
                await asyncio.sleep(0.1)
            else:
                pytest.fail("Relay HTTP state did not become ready")

        assert state["mode"] == mode
        assert state["podUID"] == "test-pod"
        assert state["connectionRevision"] == options.get("connection_revision")
        health = await relay.health()
        assert health["wan_enabled"] is (bind is not None)
        assert state["sources"] == health["sources"]
        assert state["sources"]["count"] == 0


@pytest.mark.parametrize("system_port", [False], indirect=True)
async def test_file_sources_reload_on_same_relay(relay_factory, sources_file):
    async with relay_factory(sources_file=str(sources_file)) as relay:
        state = (await relay.health())["sources"]
        assert state["count"] == 0
        assert state["appliedRevision"] is not None
        assert state["desiredRevision"] == state["appliedRevision"]
        assert state["lastError"] is None
        previous = state["appliedRevision"]
        for namespaces in (["a", "b"], ["b"], []):
            write_sources(sources_file, namespaces)
            state = await wait_for_sources(
                relay,
                lambda state: state["appliedRevision"] != previous
                and state["count"] == len(namespaces),
            )
            assert state["desiredRevision"] == state["appliedRevision"]
            assert state["lastError"] is None
            previous = state["appliedRevision"]


@pytest.mark.parametrize("system_port", [False], indirect=True)
async def test_invalid_update_retains_sources_and_recovers(relay_factory, sources_file):
    write_sources(sources_file, ["a"])
    async with relay_factory(sources_file=str(sources_file)) as relay:
        initial = (await relay.health())["sources"]
        sources_file.write_text('{"private-invalid-file":')

        rejected = await wait_for_sources(relay, lambda state: state["lastError"])
        assert rejected["appliedRevision"] == initial["appliedRevision"]
        assert rejected["desiredRevision"] is None
        assert rejected["count"] == 1
        assert "private-invalid-file" not in rejected["lastError"]

        write_sources(sources_file, ["b", "c"])
        recovered = await wait_for_sources(
            relay, lambda state: state["count"] == 2 and state["lastError"] is None
        )
        assert recovered["appliedRevision"] != initial["appliedRevision"]
        assert recovered["desiredRevision"] == recovered["appliedRevision"]


@pytest.mark.parametrize("system_port", [False], indirect=True)
@pytest.mark.parametrize(
    ("document", "message"),
    [
        pytest.param(None, "sources file is unavailable", id="missing"),
        pytest.param('{"private-invalid-file":', "invalid sources JSON", id="invalid"),
        pytest.param(
            '{"version":1,"connectionRevision":"other","sources":[]}',
            "sources connection revision does not match",
            id="mismatched-connection",
        ),
    ],
)
async def test_invalid_initial_sources_fail_startup(
    relay_factory, sources_file, document, message
):
    if document is None:
        sources_file.unlink()
    else:
        sources_file.write_text(document)

    with pytest.raises(Exception, match=message) as error:
        async with relay_factory(sources_file=str(sources_file)):
            pytest.fail("Relay accepted invalid initial sources")
    assert "private-invalid-file" not in str(error.value)
