# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream synthesized speech through Dynamo's realtime transcription API."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import sys
import time
from array import array
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from aiohttp import ClientWebSocketResponse

SAMPLE_RATE = 24_000


def _websocket_url(base_url: str) -> str:
    parts = urlsplit(base_url)
    scheme = "wss" if parts.scheme == "https" else "ws"
    return urlunsplit((scheme, parts.netloc, "/v1/realtime", "", ""))


def _pcm_rms(pcm: bytes) -> float:
    if not pcm or len(pcm) % 2:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


async def _synthesize(
    session: aiohttp.ClientSession, args: argparse.Namespace
) -> tuple[bytes, float]:
    started = time.perf_counter()
    first_chunk_at: float | None = None
    audio = bytearray()
    async with session.post(
        f"{args.base_url.rstrip('/')}/v1/audio/speech",
        json={
            "model": args.tts_model,
            "input": args.text,
            "voice": args.voice,
            "response_format": "pcm",
        },
    ) as response:
        response.raise_for_status()
        async for chunk in response.content.iter_any():
            if chunk:
                first_chunk_at = first_chunk_at or time.perf_counter()
                audio.extend(chunk)
    if first_chunk_at is None:
        raise RuntimeError("TTS returned no audio")
    return bytes(audio), first_chunk_at - started


async def _transcribe(
    session: aiohttp.ClientSession, args: argparse.Namespace, pcm: bytes
) -> tuple[str, float, float]:
    async with session.ws_connect(
        _websocket_url(args.base_url), max_msg_size=64 * 1024 * 1024
    ) as websocket:
        await websocket.send_json(
            {
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                            "transcription": {
                                "model": args.asr_model,
                                "language": args.language,
                            },
                            "noise_reduction": None,
                            "turn_detection": None,
                        }
                    },
                },
            }
        )

        started = time.perf_counter()

        async def receive_transcript() -> tuple[str, float, float]:
            first_transcript_at: float | None = None
            while True:
                message = await asyncio.wait_for(websocket.receive(), args.timeout)
                if message.type is not aiohttp.WSMsgType.TEXT:
                    raise RuntimeError(f"realtime connection closed: {message.type}")
                event = json.loads(message.data)
                event_type = event.get("type")
                if event_type == "conversation.item.input_audio_transcription.delta":
                    first_transcript_at = first_transcript_at or time.perf_counter()
                elif event_type == (
                    "conversation.item.input_audio_transcription.completed"
                ):
                    first_transcript_at = first_transcript_at or time.perf_counter()
                    completed_at = time.perf_counter()
                    return (
                        event.get("transcript", ""),
                        first_transcript_at - started,
                        completed_at - started,
                    )
                elif event_type == "error":
                    raise RuntimeError(event.get("error", event))

        receive_task = asyncio.create_task(receive_transcript())
        try:
            for offset in range(0, len(pcm), args.chunk_bytes):
                chunk = pcm[offset : offset + args.chunk_bytes]
                await websocket.send_json(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode(),
                    }
                )
                await asyncio.sleep(len(chunk) / (SAMPLE_RATE * 2))
            await websocket.send_json({"type": "input_audio_buffer.commit"})
            return await receive_task
        finally:
            receive_task.cancel()
            await asyncio.gather(receive_task, return_exceptions=True)


async def _receive_event(websocket: ClientWebSocketResponse, timeout: float) -> dict:
    message = await asyncio.wait_for(websocket.receive(), timeout)
    if message.type is not aiohttp.WSMsgType.TEXT:
        raise RuntimeError(f"realtime connection closed: {message.type}")
    event = json.loads(message.data)
    if event.get("type") == "error":
        raise RuntimeError(event.get("error", event))
    return event


async def _wait_for_event(
    websocket: ClientWebSocketResponse, event_type: str, timeout: float
) -> dict:
    while True:
        event = await _receive_event(websocket, timeout)
        if event.get("type") == event_type:
            return event


async def _open_realtime_llm(
    session: aiohttp.ClientSession, args: argparse.Namespace
) -> ClientWebSocketResponse:
    websocket = await session.ws_connect(_websocket_url(args.base_url))
    try:
        await _wait_for_event(websocket, "session.created", args.timeout)
        await websocket.send_json(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": args.llm_model,
                    "instructions": args.llm_instructions,
                    "max_output_tokens": args.max_output_tokens,
                    "output_modalities": ["text"],
                },
            }
        )
        await _wait_for_event(websocket, "session.updated", args.timeout)
    except BaseException:
        await websocket.close()
        raise
    return websocket


async def _complete_realtime(
    websocket: ClientWebSocketResponse,
    args: argparse.Namespace,
    transcript: str,
) -> tuple[str, float, float]:
    started = time.perf_counter()
    await websocket.send_json(
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": transcript}],
            },
        }
    )
    await websocket.send_json({"type": "response.create"})

    first_token_at: float | None = None
    output: list[str] = []
    while True:
        event = await _receive_event(websocket, args.timeout)
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            delta = event.get("delta", "")
            if delta:
                first_token_at = first_token_at or time.perf_counter()
                output.append(delta)
        elif event_type == "response.done":
            completed_at = time.perf_counter()
            status = event.get("response", {}).get("status")
            if status in {"cancelled", "failed"}:
                raise RuntimeError(event["response"])
            break
    if first_token_at is None:
        raise RuntimeError("realtime LLM returned no text")
    return "".join(output), first_token_at - started, completed_at - started


async def _complete_chat(
    session: aiohttp.ClientSession, args: argparse.Namespace, transcript: str
) -> tuple[str, float, float]:
    started = time.perf_counter()
    first_token_at: float | None = None
    output: list[str] = []
    async with session.post(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        json={
            "model": args.llm_model,
            "messages": [
                {"role": "system", "content": args.llm_instructions},
                {"role": "user", "content": transcript},
            ],
            "max_completion_tokens": args.max_output_tokens,
            "stream": True,
        },
    ) as response:
        response.raise_for_status()
        while raw_line := await response.content.readline():
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            payload = json.loads(line.removeprefix("data: "))
            if payload.get("error"):
                raise RuntimeError(payload["error"])
            for choice in payload.get("choices", []):
                delta = choice.get("delta", {}).get("content")
                if delta:
                    first_token_at = first_token_at or time.perf_counter()
                    output.append(delta)
    completed_at = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("chat LLM returned no text")
    return "".join(output), first_token_at - started, completed_at - started


async def run(args: argparse.Namespace) -> None:
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        realtime_llm = (
            await _open_realtime_llm(session, args)
            if args.llm_transport == "realtime"
            else None
        )
        try:
            pcm, tts_ttfb = await _synthesize(session, args)
            rms = _pcm_rms(pcm)
            if rms < args.min_rms:
                raise RuntimeError(f"TTS audio is silent or invalid (RMS={rms:.1f})")
            transcript, asr_first_transcript, asr_completed = await _transcribe(
                session, args, pcm
            )
            if not transcript.strip():
                raise RuntimeError("ASR returned an empty transcript")
            llm = None
            if realtime_llm is not None:
                llm = await _complete_realtime(realtime_llm, args, transcript)
            elif args.llm_transport == "chat":
                llm = await _complete_chat(session, args, transcript)
        finally:
            if realtime_llm is not None:
                await realtime_llm.close()

    result = {
        "audio_bytes": len(pcm),
        "audio_rms": round(rms, 1),
        "tts_ttfb_ms": round(tts_ttfb * 1000, 1),
        "asr_first_transcript_ms": round(asr_first_transcript * 1000, 1),
        "asr_completed_ms": round(asr_completed * 1000, 1),
        "transcript": transcript,
    }
    if llm is not None:
        response_text, llm_ttft, llm_total = llm
        result.update(
            {
                "llm_transport": args.llm_transport,
                "llm_ttft_from_asr_final_ms": round(llm_ttft * 1000, 1),
                "llm_total_from_asr_final_ms": round(llm_total * 1000, 1),
                "asr_start_to_llm_first_token_ms": round(
                    (asr_completed + llm_ttft) * 1000, 1
                ),
                "response_text": response_text,
            }
        )
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--tts-model", default="nvidia/magpie-tts-multilingual")
    parser.add_argument("--asr-model", default="nemotron-asr-streaming")
    parser.add_argument("--voice", default="Magpie-Multilingual.EN-US.Aria")
    parser.add_argument("--language", default="en")
    parser.add_argument("--text", default="Dynamo speech streaming is ready.")
    parser.add_argument(
        "--llm-transport",
        choices=("none", "chat", "realtime"),
        default="none",
        help="Optionally continue the completed transcript through the LLM",
    )
    parser.add_argument("--llm-model", default="nvidia/nemotron-3-nano")
    parser.add_argument(
        "--llm-instructions", default="Reply with one concise sentence."
    )
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--chunk-bytes", type=int, default=4_800)
    parser.add_argument("--min-rms", type=float, default=100.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
