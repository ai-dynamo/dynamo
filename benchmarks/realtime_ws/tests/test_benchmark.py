# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol-level tests with local HTTP/WS servers; no engine or GPU."""

import asyncio
import base64
import json
import wave
from contextlib import asynccontextmanager

import pytest
from aiohttp import web

from benchmarks.realtime_ws.benchmark import (
    RequestState,
    parse_args,
    prepare_audio,
    run,
    run_one,
    summarize,
)

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.core,
    pytest.mark.timeout(10),
]


@asynccontextmanager
async def server(handler):
    app = web.Application()
    app.router.add_route("*", "/test", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        yield f"http://127.0.0.1:{runner.addresses[0][1]}/test"
    finally:
        await runner.cleanup()


def options(tmp_path, profile, url, *extra):
    return parse_args(
        [
            "--profile",
            profile,
            "--url",
            url,
            "--model",
            "mock",
            "--output-dir",
            str(tmp_path / "run"),
            "--warmups",
            "0",
            "--requests",
            "1",
            "--timeout",
            "2",
            *extra,
        ]
    )


def audio_file(tmp_path):
    path = tmp_path / "input.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x02\x00" * 2400)
    return path


async def test_asr_records_incremental_output_and_measured_only_summary(tmp_path):
    async def handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "session.created"})
        async for message in ws:
            event = json.loads(message.data)
            if event["type"] == "input_audio_buffer.append":
                await ws.send_json({"type": "transcription.delta", "delta": "hello"})
            if event["type"] == "input_audio_buffer.commit" and event.get("final"):
                await ws.send_json({"type": "transcription.done", "text": "hello"})
        return ws

    path = audio_file(tmp_path)
    async with server(handler) as url:
        args = options(
            tmp_path,
            "native-asr",
            url,
            "--audio",
            str(path),
            "--warmups",
            "1",
            "--requests",
            "3",
            "--concurrency",
            "2",
        )
        summary = await run(args)
    assert summary["requests"] == summary["successes"] == 3
    assert summary["warmup_failures"] == 0
    assert summary["text_before_commit_requests"] == 3
    rows = [
        json.loads(line)
        for line in (args.output_dir / "requests.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 4
    assert all(row["transcript"] == "hello" for row in rows)
    assert all(row["first_text_after_commit_s"] < 0 for row in rows)


async def test_tts_ws_waits_for_session_done_and_uses_reported_rate(tmp_path):
    async def handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for message in ws:
            if json.loads(message.data)["type"] == "input.done":
                await ws.send_json(
                    {"type": "audio.start", "format": "pcm", "sample_rate": 16000}
                )
                await ws.send_bytes(b"\x01\x00" * 160)
                await ws.send_json({"type": "audio.done"})
                await ws.send_json({"type": "session.done"})
        return ws

    async with server(handler) as url:
        state = await run_one(options(tmp_path, "native-tts-ws", url), [], "test")
    result = state.result()
    assert result["success"], result["error"]
    assert result["terminal_event"] == "session.done"
    assert result["audio_duration_s"] == pytest.approx(0.01)
    assert result["first_audio_from_first_input_s"] >= 0


async def test_tts_http_headers_do_not_count_as_audio(tmp_path):
    async def handler(request):
        assert (await request.json())["response_format"] == "pcm"
        response = web.StreamResponse(
            headers={"Content-Type": "application/octet-stream"}
        )
        await response.prepare(request)
        await asyncio.sleep(0.02)
        await response.write(b"\x01\x00" * 240)
        await response.write_eof()
        return response

    async with server(handler) as url:
        state = await run_one(options(tmp_path, "tts-http", url), [], "test")
    assert state.result()["success"]
    headers = next(
        event
        for event in state.events
        if event["event"]["type"] == "benchmark.http.headers"
    )
    assert state.first_audio - state.started > headers["elapsed_s"]
    assert state.first_audio - state.started >= 0.02
    assert state.result()["audio_duration_s"] == pytest.approx(0.01)


@pytest.mark.parametrize("mode", ["error", "close", "no_audio", "odd_audio"])
async def test_failures_are_not_reported_as_success(tmp_path, mode):
    async def handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for message in ws:
            if json.loads(message.data)["type"] == "input.done":
                if mode == "error":
                    await ws.send_json({"type": "error", "message": "failed"})
                elif mode == "close":
                    await ws.close()
                else:
                    if mode == "odd_audio":
                        await ws.send_bytes(b"\x01")
                    await ws.send_json({"type": "session.done"})
        return ws

    async with server(handler) as url:
        state = await run_one(options(tmp_path, "native-tts-ws", url), [], "test")
    assert not state.result()["success"]
    assert state.error


def test_audio_event_redaction_and_native_omni_completion():
    state = RequestState("test", "native-omni", started=0, first_input=1, committed=2)
    encoded = base64.b64encode(b"\x00\x01" * 100).decode()
    assert not state.accept_event(
        {"type": "response.output_audio.delta", "delta": encoded}, 3
    )
    assert not state.accept_event({"type": "response.output_audio.done"}, 4)
    assert state.accept_event(
        {"type": "response.done", "response": {"status": "completed"}}, 5
    )
    assert encoded not in json.dumps(state.events)
    assert state.result()["first_audio_after_commit_s"] == 1


def test_whole_file_resampling_is_deterministic(tmp_path):
    path = audio_file(tmp_path)
    first, first_info = prepare_audio(path, 16000)
    second, second_info = prepare_audio(path, 16000)
    assert first == second
    assert first_info == second_info
    assert first_info["duration_s"] == pytest.approx(0.1)
    assert first_info["prepared_pcm_sha256"] != first_info["source_sha256"]


@pytest.mark.parametrize(
    "profile,final_text",
    [
        ("dynamo-asr", ""),
        ("dynamo-asr", " \t\n"),
        ("dynamo-omni", ""),
        ("dynamo-omni", " \t\n"),
    ],
)
async def test_empty_final_text_is_failure_even_after_asr_deltas(
    tmp_path, profile, final_text
):
    async def handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for message in ws:
            if json.loads(message.data)["type"] == "input_audio_buffer.commit":
                if profile.endswith("asr"):
                    await ws.send_json(
                        {
                            "type": "conversation.item.input_audio_transcription.delta",
                            "delta": "partial",
                        }
                    )
                    await ws.send_json(
                        {
                            "type": "conversation.item.input_audio_transcription.completed",
                            "transcript": final_text,
                        }
                    )
                else:
                    await ws.send_json(
                        {"type": "response.output_text.delta", "delta": final_text}
                    )
                    await ws.send_json(
                        {"type": "response.done", "response": {"status": "completed"}}
                    )
        return ws

    async with server(handler) as url:
        args = options(
            tmp_path,
            profile,
            url,
            "--audio",
            str(audio_file(tmp_path)),
            "--modalities",
            "text",
        )
        state = await run_one(args, [], "test")
    assert not state.result()["success"]
    assert state.result()["error_category"] == "empty_text"
    assert state.result()["transcript"] == final_text


async def test_token_limit_truncation_retains_partial_metrics_and_distinct_category(
    tmp_path,
):
    async def handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for message in ws:
            if json.loads(message.data)["type"] == "response.create":
                await ws.send_json(
                    {
                        "type": "response.output_audio.delta",
                        "delta": base64.b64encode(b"\x01\x00" * 240).decode(),
                    }
                )
                await ws.send_json(
                    {
                        "type": "response.done",
                        "response": {
                            "status": "incomplete",
                            "status_details": {
                                "type": "incomplete",
                                "reason": "max_output_tokens",
                            },
                        },
                    }
                )
        return ws

    async with server(handler) as url:
        args = options(
            tmp_path, "native-omni", url, "--audio", str(audio_file(tmp_path))
        )
        state = await run_one(args, [], "test")
    result = state.result()
    assert not result["success"]
    assert result["truncated"]
    assert result["error_category"] == "truncated"
    assert result["terminal_status"] == "incomplete"
    assert result["done_after_commit_s"] >= result["first_audio_after_commit_s"] >= 0
    assert result["audio_duration_s"] == pytest.approx(0.01)
    summary = summarize([result], 1)
    assert summary["successes"] == 0
    assert summary["truncated_requests"] == 1
    assert summary["error_categories"] == {"truncated": 1}
    assert summary["metrics"]["first_audio_after_commit_s"] is None
