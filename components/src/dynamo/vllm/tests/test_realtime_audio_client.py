# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import base64
import importlib.util
import sys
import wave
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

pytestmark = [
    pytest.mark.integration,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(5),
]


@pytest.fixture(scope="module")
def audio_client():
    path = (
        Path(__file__).resolve().parents[5]
        / "examples/backends/vllm/launch/realtime_audio_client.py"
    )
    spec = importlib.util.spec_from_file_location("realtime_audio_client_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "session_type,input_sample_rate,expected_rate",
    [
        pytest.param("realtime", None, 16_000, id="omni-python-default"),
        pytest.param("realtime", 24_000, 24_000, id="omni-sidecar-24khz"),
        pytest.param("transcription", None, 24_000, id="asr-default"),
    ],
)
async def test_example_sends_declared_pcm_rate(
    audio_client, tmp_path, session_type, input_sample_rate, expected_rate
):
    source = tmp_path / "audio.wav"
    with wave.open(str(source), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(bytes(3_200 * 2))  # 0.2 seconds of mono PCM16.

    received = []

    async def receive_audio(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        while True:
            event = await ws.receive_json(timeout=2)
            received.append(event)
            if event["type"] == "input_audio_buffer.commit":
                break
        if session_type == "transcription":
            await ws.send_json(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "test audio",
                }
            )
        else:
            await ws.send_json(
                {"type": "response.done", "response": {"status": "completed"}}
            )
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/v1/realtime", receive_audio)
    async with TestServer(app) as server:
        args = argparse.Namespace(
            url=str(server.make_url("/v1/realtime").with_scheme("ws")),
            model="test-model",
            session_type=session_type,
            input_audio=str(source),
            input_audio_url="https://unused.invalid/audio.wav",
            input_sample_rate=input_sample_rate,
            language="en",
            output_modalities=None,
            output_dir=str(tmp_path / "output"),
            output_sample_rate=24_000,
            chunk_ms=50,
            timeout=2,
            require_incremental=False,
        )
        assert await asyncio.wait_for(audio_client.run(args), timeout=3) == 0

    assert [event["type"] for event in received] == [
        "session.update",
        *["input_audio_buffer.append"] * 4,
        "input_audio_buffer.commit",
    ]
    session = received[0]["session"]
    assert session["type"] == session_type
    if session_type == "realtime" and input_sample_rate is None:
        assert "audio" not in session
    else:
        assert session["audio"]["input"]["format"] == {
            "type": "audio/pcm",
            "rate": expected_rate,
        }

    chunks = [
        base64.b64decode(event["audio"], validate=True) for event in received[1:-1]
    ]
    assert [len(chunk) for chunk in chunks] == [expected_rate * 2 // 20] * 4
    assert sum(map(len, chunks)) / (expected_rate * 2) == pytest.approx(0.2)


def test_example_rejects_16khz_transcription(audio_client, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "realtime_audio_client.py",
            "--model",
            "test-model",
            "--session-type",
            "transcription",
            "--input-sample-rate",
            "16000",
        ],
    )
    with pytest.raises(SystemExit) as error:
        audio_client.main()
    assert error.value.code == 2
    assert (
        "transcription sessions require --input-sample-rate 24000"
        in capsys.readouterr().err
    )
