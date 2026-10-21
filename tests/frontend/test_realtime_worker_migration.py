# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-loss regression through the real frontend, discovery and TCP transport."""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from contextlib import ExitStack

import aiohttp
import pytest

from tests.frontend.test_realtime_python_bridge import _recv_json
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.timeout(120),
]


@pytest.fixture(params=[1])
def migration_frontend(
    request, monkeypatch, dynamo_dynamic_ports, tmp_path, realtime_budget_mib
):
    model = "migration-asr"
    endpoint = f"migration_{uuid.uuid4().hex}.asr.generate"
    port = dynamo_dynamic_ports.frontend_port
    monkeypatch.setenv("DYN_FILE_KV", str(tmp_path / "discovery"))
    monkeypatch.delenv("NATS_SERVER", raising=False)
    with ExitStack() as stack:
        stack.enter_context(
            DynamoFrontendProcess(
                request,
                frontend_port=port,
                migration_limit=request.param,
                extra_args=[
                    "--migration-realtime-budget-mib",
                    str(realtime_budget_mib),
                    "--discovery-backend",
                    "file",
                    "--request-plane",
                    "tcp",
                    "--event-plane",
                    "zmq",
                ],
                terminate_all_matching_process_names=False,
            )
        )
        workers = {}
        for worker_id in ("a", "b"):
            workers[worker_id] = stack.enter_context(
                ManagedProcess(
                    command=[
                        "python3",
                        "-m",
                        "tests.frontend.realtime_asr_mock_worker",
                        "--worker-id",
                        worker_id,
                        "--model",
                        model,
                        "--endpoint",
                        endpoint,
                        "--audit-dir",
                        str(tmp_path),
                        *(["--split-deltas"] if worker_id == "b" else []),
                    ],
                    health_check_funcs=[(tmp_path / f"{worker_id}.ready").exists],
                    timeout=30,
                    terminate_all_matching_process_names=False,
                    log_dir=str(tmp_path / worker_id),
                )
            )
        yield port, model, workers, tmp_path


@pytest.fixture
def realtime_budget_mib():
    return 512


def session_update(model):
    return {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {"model": model},
                    "turn_detection": None,
                }
            },
        },
    }


async def open_session(client, port, model):
    ws = await client.ws_connect(f"http://127.0.0.1:{port}/v1/realtime")
    assert (await _recv_json(ws, 5))["type"] == "session.created"
    await ws.send_json(session_update(model))
    updated = await _recv_json(ws, 5)
    assert updated["type"] == "session.updated", updated
    assert updated["session"]["audio"]["input"]["turn_detection"] is None
    return ws, updated["event_id"].split(":")[0]


async def append(ws, text):
    event = {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(text.encode()).decode(),
    }
    await ws.send_json(event)
    return event


async def ready_session(client, deployment):
    port, model, workers, _ = deployment
    seen = set()
    async with asyncio.timeout(15):
        while seen != set(workers):
            ws, worker_id = await open_session(client, port, model)
            seen.add(worker_id)
            await ws.close()
    return await open_session(client, port, model)


async def worker_loss_round_trip(deployment):
    _, _, workers, audit_dir = deployment
    async with aiohttp.ClientSession() as client:
        ws, worker_id = await ready_session(client, deployment)
        async with ws:
            chunks = ["héllo ", *(f"{index:02} " for index in range(96))]
            expected_audio = [await append(ws, text) for text in chunks]
            prefix = "".join(chunks)
            deltas = []
            item_id = None
            while "".join(deltas) != prefix:
                event = await _recv_json(ws, 5)
                assert event["type"].endswith(".delta"), event
                if item_id is None:
                    item_id = event["item_id"]
                assert event["item_id"] == item_id
                deltas.append(event["delta"])

            workers[worker_id].proc.kill()
            # All 97 appends must be replayed through the 64-slot request queue.
            expected_audio.append(await append(ws, "tail"))
            await ws.send_json({"type": "input_audio_buffer.commit"})

            committed = 0
            while True:
                event = await _recv_json(ws, 10)
                assert event["item_id"] == item_id, event
                if event["type"].endswith(".delta"):
                    deltas.append(event["delta"])
                elif event["type"] == "input_audio_buffer.committed":
                    committed += 1
                elif event["type"].endswith(".completed"):
                    expected_text = prefix + "tail"
                    assert event["transcript"] == expected_text
                    assert "".join(deltas) == expected_text
                    break
                else:
                    pytest.fail(f"unexpected recovery event: {event}")
            assert committed == 1

            replacement = "b" if worker_id == "a" else "a"
            replay = [
                [json.loads(line) for line in path.read_text().splitlines()]
                for path in audit_dir.glob(f"{replacement}-*.jsonl")
            ]
            replay = [events for events in replay if len(events) > 1]
            assert len(replay) == 1, replay
            original = [
                [json.loads(line) for line in path.read_text().splitlines()]
                for path in audit_dir.glob(f"{worker_id}-*.jsonl")
            ]
            original = [events for events in original if len(events) > 1]
            assert len(original) == 1
            assert replay[0][0] == original[0][0]
            assert [e["audio"] for e in replay[0] if e["type"].endswith(".append")] == [
                e["audio"] for e in expected_audio
            ]
            assert replay[0][-1]["type"] == "input_audio_buffer.commit"

            await append(ws, "next turn")
            await ws.send_json({"type": "input_audio_buffer.commit"})
            while True:
                event = await _recv_json(ws, 5)
                assert event["item_id"] != item_id, event
                if event["type"].endswith(".completed"):
                    assert event["transcript"] == "next turn"
                    break


def test_realtime_worker_loss_preserves_session(migration_frontend):
    asyncio.run(worker_loss_round_trip(migration_frontend))


async def finish_turns(ws, expected):
    finals = []
    events = []
    while len(finals) < len(expected):
        event = await _recv_json(ws, 10)
        assert event["type"] not in {
            "error",
            "session.created",
            "session.updated",
        }, event
        events.append(event)
        if event["type"].endswith(".completed"):
            finals.append(event["transcript"])
    assert finals == expected
    return events


async def failover_at_boundary(deployment, boundary):
    _, _, workers, audit_dir = deployment
    async with aiohttp.ClientSession() as client:
        ws, active = await ready_session(client, deployment)
        async with ws:
            replacement = "b" if active == "a" else "a"
            if boundary != "idle":
                (audit_dir / f"{active}.hold_final").touch()
                await append(ws, "first")
                await ws.send_json({"type": "input_audio_buffer.commit"})
                while (await _recv_json(ws, 5))[
                    "type"
                ] != "input_audio_buffer.committed":
                    pass
                if boundary == "queued":
                    await append(ws, "second")
                    await ws.send_json({"type": "input_audio_buffer.commit"})
            workers[active].proc.kill()
            if boundary == "idle":
                await append(ws, "after idle")
                await ws.send_json({"type": "input_audio_buffer.commit"})
            expected = {
                "idle": ["after idle"],
                "committed": ["first"],
                "queued": ["first", "second"],
            }[boundary]
            events = await finish_turns(ws, expected)
            assert sum(e["type"] == "input_audio_buffer.committed" for e in events) == (
                0 if boundary == "committed" else 1
            )
            # A successful turn must not reset the socket's migration budget.
            workers[replacement].proc.kill()
            event = await _recv_json(ws, 5)
            assert event["type"] == "error", event
            assert event["error"]["message"] == "Internal server error"
            assert (
                await asyncio.wait_for(ws.receive(), 5)
            ).type == aiohttp.WSMsgType.CLOSE


@pytest.mark.parametrize("boundary", ["idle", "committed", "queued"])
def test_worker_loss_at_turn_boundaries(migration_frontend, boundary):
    asyncio.run(failover_at_boundary(migration_frontend, boundary))


async def unavailable_replacement(deployment, failure):
    _, _, workers, audit_dir = deployment
    async with aiohttp.ClientSession() as client:
        ws, active = await ready_session(client, deployment)
        async with ws:
            replacement = "b" if active == "a" else "a"
            if failure == "absent":
                workers[replacement].proc.kill()
            elif failure == "hung":
                (audit_dir / f"{replacement}.hang_setup").touch()
            elif failure == "updated":
                await ws.send_json(session_update(deployment[1]))
                assert (await _recv_json(ws, 5))["type"] == "session.updated"
            workers[active].proc.kill()
            event = await _recv_json(ws, 8)
            assert event["type"] == "error", event
            assert (
                await asyncio.wait_for(ws.receive(), 5)
            ).type == aiohttp.WSMsgType.CLOSE


@pytest.mark.parametrize("failure", ["absent", "hung", "updated"])
def test_unrecoverable_session_closes(migration_frontend, failure):
    asyncio.run(unavailable_replacement(migration_frontend, failure))


@pytest.mark.parametrize("migration_frontend", [0], indirect=True)
def test_migration_is_disabled_by_default(migration_frontend):
    asyncio.run(unavailable_replacement(migration_frontend, "disabled"))


@pytest.mark.parametrize("realtime_budget_mib", [0])
def test_zero_realtime_budget_disables_recovery(migration_frontend):
    asyncio.run(unavailable_replacement(migration_frontend, "disabled"))


async def disconnect_during_replay(deployment):
    _, _, workers, audit_dir = deployment
    async with aiohttp.ClientSession() as client:
        ws, active = await ready_session(client, deployment)
        replacement = "b" if active == "a" else "a"
        async with ws:
            await append(ws, "unfinished")
            assert (await _recv_json(ws, 5))["type"].endswith(".delta")
            (audit_dir / f"{replacement}.hang_replay").touch()
            workers[active].proc.kill()
            async with asyncio.timeout(5):
                while True:
                    replay = [
                        path
                        for path in audit_dir.glob(f"{replacement}-*.jsonl")
                        if len(path.read_text().splitlines()) > 1
                    ]
                    if replay:
                        break
                    await asyncio.sleep(0.01)
            await ws.close()
            async with asyncio.timeout(5):
                while not replay[0].with_suffix(".closed").exists():
                    await asyncio.sleep(0.01)


def test_client_disconnect_stops_replacement(migration_frontend):
    asyncio.run(disconnect_during_replay(migration_frontend))


async def application_error_then_valid_turn(deployment):
    _, _, _, audit_dir = deployment
    async with aiohttp.ClientSession() as client:
        ws, active = await ready_session(client, deployment)
        async with ws:
            await append(ws, "partial")
            partial = ""
            while partial != "partial":
                event = await _recv_json(ws, 5)
                assert event["type"].endswith(".delta"), event
                partial += event["delta"]
            canceled_item = event["item_id"]
            abort = audit_dir / f"{active}.abort_turn"
            abort.touch()
            await append(ws, "overflow")
            error = await _recv_json(ws, 5)
            assert error["type"] == "error", error
            assert error["error"]["code"] == "invalid_audio", error
            abort.unlink()
            await append(ws, "next turn")
            await ws.send_json({"type": "input_audio_buffer.commit"})
            events = await finish_turns(ws, ["next turn"])
            assert all(event["item_id"] != canceled_item for event in events)
            assert len({event["item_id"] for event in events}) == 1
            assert (
                "".join(
                    event["delta"]
                    for event in events
                    if event["type"].endswith(".delta")
                )
                == "next turn"
            )


@pytest.mark.parametrize("migration_frontend", [0, 1], indirect=True)
def test_recoverable_audio_error_preserves_session(migration_frontend):
    asyncio.run(application_error_then_valid_turn(migration_frontend))
