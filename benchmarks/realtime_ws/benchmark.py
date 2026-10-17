# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure realtime audio paths with a shared closed-loop workload.

This is an endpoint benchmark, not an engine benchmark: every measured request
opens a new connection. Audio preparation and output-file writes are outside the
timed request. See README.md for protocol and comparison limitations.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import platform
import statistics
import sys
import time
import wave
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import scipy
from scipy.signal import resample_poly

PROFILES = (
    "dynamo-asr",
    "native-asr",
    "dynamo-omni",
    "native-omni",
    "tts-http",
    "native-tts-ws",
)
INPUT_RATES = {
    "dynamo-asr": 24000,
    "native-asr": 16000,
    "dynamo-omni": 16000,
    "native-omni": 24000,
}
TEXT_DELTA_EVENTS = frozenset(
    {
        "transcription.delta",
        "conversation.item.input_audio_transcription.delta",
        "response.output_audio_transcript.delta",
        "response.audio_transcript.delta",
        "response.output_text.delta",
        "response.text.delta",
    }
)
AUDIO_DELTA_EVENTS = frozenset(
    {
        "response.output_audio.delta",
        "response.audio.delta",
    }
)
ASR_DONE_EVENTS = frozenset(
    {
        "transcription.done",
        "conversation.item.input_audio_transcription.completed",
    }
)


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare_audio(path: Path, rate: int) -> tuple[bytes, dict[str, Any]]:
    """Resample once, before timing, using a deterministic polyphase filter."""
    source = path.read_bytes()
    with wave.open(str(path), "rb") as wav:
        source_rate, channels = wav.getframerate(), wav.getnchannels()
        if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError("Input must be an uncompressed PCM16 WAV")
        raw_pcm = wav.readframes(wav.getnframes())
    if channels == 1 and source_rate == rate:
        pcm = raw_pcm
    else:
        samples = np.frombuffer(raw_pcm, dtype="<i2").astype(np.float64)
        samples = samples.reshape(-1, channels).mean(axis=1)
        if source_rate != rate:
            divisor = math.gcd(source_rate, rate)
            samples = resample_poly(samples, rate // divisor, source_rate // divisor)
        pcm = np.clip(np.rint(samples), -32768, 32767).astype("<i2").tobytes()
    if not pcm:
        raise ValueError("Input audio is empty")
    return pcm, {
        "source_path": str(path.resolve()),
        "source_sha256": sha256(source),
        "source_rate": source_rate,
        "source_channels": channels,
        "prepared_pcm_sha256": sha256(pcm),
        "prepared_rate": rate,
        "prepared_bytes": len(pcm),
        "duration_s": len(pcm) / (2 * rate),
        "resampler": "scipy.signal.resample_poly; float64; round; clip; s16le",
    }


def redact_event(event: dict[str, Any]) -> dict[str, Any]:
    """Keep event metadata and text, never base64 audio payloads."""
    is_audio_delta = event.get("type") in AUDIO_DELTA_EVENTS

    def redact(value: Any, key: str = "") -> Any:
        if isinstance(value, str) and (
            key in {"audio", "audio_base64"} or (key == "delta" and is_audio_delta)
        ):
            return {"base64_chars": len(value)}
        if isinstance(value, dict):
            return {name: redact(item, name) for name, item in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    return redact(event)


@dataclass
class RequestState:
    request_id: str
    profile: str
    started: float = field(default_factory=time.perf_counter)
    connected: float | None = None
    first_input: float | None = None
    committed: float | None = None
    first_text: float | None = None
    first_audio: float | None = None
    done: float | None = None
    finished: float | None = None
    output_rate: int = 24000
    output_channels: int = 1
    error: str | None = None
    error_category: str | None = None
    terminal_event: str | None = None
    terminal_status: str | None = None
    status_details: dict[str, Any] | None = None
    truncated: bool = False
    final_text: str | None = None
    text_parts: list[str] = field(default_factory=list)
    audio: bytearray = field(default_factory=bytearray)
    events: list[dict[str, Any]] = field(default_factory=list)
    audio_times: list[float] = field(default_factory=list)

    def record(self, event: dict[str, Any], now: float | None = None) -> None:
        stamp = time.perf_counter() if now is None else now
        self.events.append(
            {"elapsed_s": stamp - self.started, "event": redact_event(event)}
        )

    def add_audio(self, chunk: bytes, now: float) -> None:
        if chunk:
            self.audio.extend(chunk)
            if len(self.audio) >= 2 * self.output_channels:
                if self.first_audio is None:
                    self.first_audio = now
                self.audio_times.append(now)

    def accept_event(self, event: dict[str, Any], now: float) -> bool:
        self.record(event, now)
        kind = event.get("type")
        if kind in {"error", "conversation.item.input_audio_transcription.failed"}:
            self.error_category = "server_error"
            raise RuntimeError(json.dumps(redact_event(event)))
        if kind == "audio.start":
            if event.get("format", "pcm") != "pcm":
                raise ValueError("TTS WS returned non-PCM audio")
            self.output_rate = int(event.get("sample_rate", self.output_rate))
            self.output_channels = int(event.get("channels", 1))
        if kind in TEXT_DELTA_EVENTS:
            delta = event.get("delta", "")
            if delta:
                self.text_parts.append(delta)
                if delta.strip() and self.first_text is None:
                    self.first_text = now
        if kind in AUDIO_DELTA_EVENTS:
            self.add_audio(base64.b64decode(event.get("delta", ""), validate=True), now)
        terminal = (
            (self.profile.endswith("asr") and kind in ASR_DONE_EVENTS)
            or (self.profile.endswith("omni") and kind == "response.done")
            or (self.profile == "native-tts-ws" and kind == "session.done")
        )
        if kind in ASR_DONE_EVENTS:
            self.final_text = event.get("text", event.get("transcript", ""))
            if self.final_text.strip() and self.first_text is None:
                self.first_text = now
        if terminal:
            # Preserve partial-output timings even if the terminal status is
            # incomplete/failed; only completed requests enter summary metrics.
            self.done, self.terminal_event = now, kind
        if kind == "response.done":
            response = event.get("response", {})
            self.terminal_status = response.get("status")
            self.status_details = response.get("status_details")
            if self.terminal_status not in {None, "completed"}:
                reason = (self.status_details or {}).get("reason")
                self.truncated = (
                    self.terminal_status == "incomplete"
                    and reason == "max_output_tokens"
                )
                self.error_category = (
                    "truncated"
                    if self.truncated
                    else f"response_{self.terminal_status}"
                )
                raise RuntimeError(
                    f"Response ended with status {self.terminal_status}: {self.status_details}"
                )
        return terminal

    def transcript(self) -> str:
        return (
            self.final_text if self.final_text is not None else "".join(self.text_parts)
        )

    def result(self) -> dict[str, Any]:
        def difference(end: float | None, begin: float | None) -> float | None:
            return end - begin if end is not None and begin is not None else None

        duration = len(self.audio) / (2 * self.output_channels * self.output_rate)
        generation = difference(self.done, self.committed)
        result = {
            "request_id": self.request_id,
            "profile": self.profile,
            "success": self.error is None and self.done is not None,
            "error": self.error,
            "error_category": self.error_category,
            "terminal_event": self.terminal_event,
            "terminal_status": self.terminal_status,
            "status_details": self.status_details,
            "truncated": self.truncated,
            "connect_s": difference(self.connected, self.started),
            "setup_s": difference(self.first_input, self.started),
            "input_s": difference(self.committed, self.first_input),
            "request_s": difference(self.done, self.started),
            "attempt_s": difference(self.finished, self.started),
            "done_from_first_input_s": difference(self.done, self.first_input),
            "done_after_commit_s": generation,
            "first_text_from_first_input_s": difference(
                self.first_text, self.first_input
            ),
            "first_text_after_commit_s": difference(self.first_text, self.committed),
            "first_audio_from_first_input_s": difference(
                self.first_audio, self.first_input
            ),
            "first_audio_after_commit_s": difference(self.first_audio, self.committed),
            "text_before_commit": self.first_text is not None
            and self.committed is not None
            and self.first_text < self.committed,
            "audio_before_commit": self.first_audio is not None
            and self.committed is not None
            and self.first_audio < self.committed,
            "transcript": self.transcript(),
            "text_deltas": len(self.text_parts),
            "audio_chunks": len(self.audio_times),
            "audio_bytes": len(self.audio),
            "audio_sha256": sha256(self.audio),
            "audio_rate": self.output_rate,
            "audio_channels": self.output_channels,
            "audio_duration_s": duration,
            "generation_rtf": generation / duration
            if generation is not None and duration
            else None,
            "max_audio_gap_s": max(
                (b - a for a, b in zip(self.audio_times, self.audio_times[1:])),
                default=None,
            ),
        }
        if len(self.audio) >= 2:
            samples = np.frombuffer(
                self.audio[: len(self.audio) // 2 * 2], dtype="<i2"
            ).astype(np.float64)
            result["audio_rms_pcm16"] = float(np.sqrt(np.mean(samples * samples)))
            result["audio_nonzero_samples"] = int(np.count_nonzero(samples))
        return result


def session_events(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.profile == "native-asr":
        return [
            {"type": "session.update", "model": args.model},
            {"type": "input_audio_buffer.commit", "final": False},
        ]
    if args.profile == "dynamo-asr":
        config = {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": args.input_rate},
                    "transcription": {"model": args.model, "language": args.language},
                    "noise_reduction": None,
                    "turn_detection": None,
                }
            },
        }
    elif args.profile.endswith("omni"):
        config = {
            "type": "realtime",
            "model": args.model,
            "output_modalities": args.modalities.split(","),
        }
        if args.profile == "native-omni":
            config["audio"] = {
                "input": {
                    "format": {"type": "audio/pcm", "rate": args.input_rate},
                    "turn_detection": None,
                },
                "output": {"format": {"type": "audio/pcm", "rate": args.output_rate}},
            }
    elif args.profile == "native-tts-ws":
        config = {
            "type": "session.config",
            "model": args.model,
            "voice": args.voice,
            "language": args.language,
            "response_format": "pcm",
            "stream_audio": True,
            "split_granularity": "none",
        }
        config.update(args.request_extra)
        return [config]
    else:
        return []
    config.update(args.session_extra)
    return [{"type": "session.update", "session": config}]


async def send_input(
    ws: aiohttp.ClientWebSocketResponse,
    args: argparse.Namespace,
    state: RequestState,
    chunks: list[tuple[str, float]],
) -> None:
    for event in session_events(args):
        await ws.send_json(event)
    state.first_input = time.perf_counter()
    state.record({"type": "benchmark.input.start"}, state.first_input)
    if args.profile == "native-tts-ws":
        await ws.send_json({"type": "input.text", "text": args.text})
        state.committed = time.perf_counter()
        await ws.send_json({"type": "input.done"})
    else:
        elapsed_audio = 0.0
        for payload, duration in chunks:
            await ws.send_str(payload)
            elapsed_audio += duration
            if args.pace:
                target = state.first_input + elapsed_audio / args.pace
                await asyncio.sleep(max(0.0, target - time.perf_counter()))
        state.committed = time.perf_counter()
        event = {"type": "input_audio_buffer.commit"}
        if args.profile == "native-asr":
            event["final"] = True
        await ws.send_json(event)
        if args.profile == "native-omni":
            await ws.send_json({"type": "response.create"})
    state.record({"type": "benchmark.input.commit"}, state.committed)


async def request_ws(
    args: argparse.Namespace, state: RequestState, chunks: list[tuple[str, float]]
) -> None:
    # A fresh ClientSession ensures every request includes connection setup.
    async with (
        aiohttp.ClientSession() as client,
        client.ws_connect(args.url, max_msg_size=64 * 1024 * 1024) as ws,
    ):
        state.connected = time.perf_counter()
        sender = asyncio.create_task(send_input(ws, args, state, chunks))
        try:
            async for message in ws:
                now = time.perf_counter()
                if message.type == aiohttp.WSMsgType.BINARY:
                    if args.profile != "native-tts-ws":
                        raise ValueError("Unexpected binary WS frame")
                    state.add_audio(message.data, now)
                    state.record(
                        {"type": "benchmark.audio.binary", "bytes": len(message.data)},
                        now,
                    )
                elif message.type == aiohttp.WSMsgType.TEXT:
                    if state.accept_event(json.loads(message.data), now):
                        await sender
                        break
                elif message.type == aiohttp.WSMsgType.ERROR:
                    raise RuntimeError(f"WebSocket receive failed: {ws.exception()}")
            if state.done is None:
                raise RuntimeError("WebSocket closed before terminal event")
        finally:
            if not sender.done():
                sender.cancel()
            with suppress(asyncio.CancelledError):
                await sender


async def request_http(args: argparse.Namespace, state: RequestState) -> None:
    payload = {
        "model": args.model,
        "input": args.text,
        "voice": args.voice,
        "language": args.language,
        "response_format": "pcm",
        "stream": True,
    }
    payload.update(args.request_extra)
    if payload["response_format"] != "pcm":
        raise ValueError("tts-http requires response_format=pcm to measure real audio")
    async with aiohttp.ClientSession() as client:
        state.first_input = state.committed = time.perf_counter()
        state.record({"type": "benchmark.input.commit"}, state.committed)
        async with client.post(args.url, json=payload) as response:
            state.record(
                {
                    "type": "benchmark.http.headers",
                    "status": response.status,
                    "content_type": response.headers.get("Content-Type"),
                }
            )
            if response.status != 200:
                raise RuntimeError(
                    f"HTTP {response.status}: {(await response.text())[:4000]}"
                )
            async for chunk in response.content.iter_any():
                now = time.perf_counter()
                state.add_audio(chunk, now)
                state.record({"type": "benchmark.audio.http", "bytes": len(chunk)}, now)
        if not state.audio:
            raise RuntimeError("HTTP response had no audio")
        if state.audio.startswith(b"RIFF"):
            raise ValueError("Server returned WAV despite PCM request; invalid timing")
        state.done = time.perf_counter()
        state.terminal_event = "http.eof"


async def run_one(
    args: argparse.Namespace, chunks: list[tuple[str, float]], request_id: str
) -> RequestState:
    state = RequestState(request_id, args.profile, output_rate=args.output_rate)
    try:
        async with asyncio.timeout(args.timeout):
            if args.profile == "tts-http":
                await request_http(args, state)
            else:
                await request_ws(args, state, chunks)
        if state.audio and len(state.audio) % (2 * state.output_channels):
            raise ValueError("Output audio ends with an incomplete PCM16 frame")
        requires_text = args.profile.endswith("asr") or (
            args.profile.endswith("omni") and args.modalities.split(",") == ["text"]
        )
        if requires_text and not state.transcript().strip():
            state.error_category = "empty_text"
            raise ValueError(
                "Request terminated with no non-whitespace transcript/text"
            )
        if (
            "tts" in args.profile
            or (args.profile.endswith("omni") and "audio" in args.modalities)
        ) and not state.audio:
            state.error_category = "empty_audio"
            raise ValueError("Audio-generating request terminated with no audio")
    except (
        aiohttp.ClientError,
        TimeoutError,
        RuntimeError,
        ValueError,
        OSError,
    ) as exc:
        # Request failures are observations, persisted and counted as benchmark
        # failures. The CLI exits nonzero; programming errors still propagate.
        state.error = f"{type(exc).__name__}: {exc}"
        if state.error_category is None:
            state.error_category = (
                "timeout" if isinstance(exc, TimeoutError) else "request_error"
            )
    state.finished = time.perf_counter()
    return state


def distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": max(values),
    }


def summarize(results: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
    successful = [result for result in results if result["success"]]
    metric_keys = (
        "connect_s",
        "setup_s",
        "input_s",
        "request_s",
        "done_from_first_input_s",
        "done_after_commit_s",
        "first_text_from_first_input_s",
        "first_text_after_commit_s",
        "first_audio_from_first_input_s",
        "first_audio_after_commit_s",
        "audio_duration_s",
        "generation_rtf",
        "max_audio_gap_s",
    )
    return {
        "requests": len(results),
        "successes": len(successful),
        "failures": len(results) - len(successful),
        "truncated_requests": sum(result["truncated"] for result in results),
        "error_categories": dict(
            Counter(r["error_category"] for r in results if not r["success"])
        ),
        "wall_s": wall_s,
        "successful_requests_per_s": len(successful) / wall_s if wall_s else None,
        "output_audio_seconds_per_s": sum(r["audio_duration_s"] for r in successful)
        / wall_s
        if wall_s
        else None,
        "text_before_commit_requests": sum(r["text_before_commit"] for r in successful),
        "audio_before_commit_requests": sum(
            r["audio_before_commit"] for r in successful
        ),
        "metrics": {
            key: distribution([r[key] for r in successful if r[key] is not None])
            for key in metric_keys
        },
        "errors": [
            {
                "request_id": r["request_id"],
                "error": r["error"],
                "error_category": r["error_category"],
                "terminal_status": r["terminal_status"],
                "status_details": r["status_details"],
                "truncated": r["truncated"],
            }
            for r in results
            if not r["success"]
        ],
    }


def persist(state: RequestState, output_dir: Path, phase: str) -> dict[str, Any]:
    result = {"phase": phase, **state.result()}
    if state.audio and len(state.audio) % (2 * state.output_channels) == 0:
        filename = f"{state.request_id}.wav"
        with wave.open(str(output_dir / filename), "wb") as wav:
            wav.setnchannels(state.output_channels)
            wav.setsampwidth(2)
            wav.setframerate(state.output_rate)
            wav.writeframes(state.audio)
        result["audio_file"] = filename
    elif state.audio:
        filename = f"{state.request_id}.invalid.pcm"
        (output_dir / filename).write_bytes(state.audio)
        result["audio_file"] = filename
    with (output_dir / "events.jsonl").open("a") as stream:
        for event in state.events:
            stream.write(
                json.dumps({"request_id": state.request_id, "phase": phase, **event})
                + "\n"
            )
    with (output_dir / "requests.jsonl").open("a") as stream:
        stream.write(json.dumps(result, allow_nan=False) + "\n")
    return result


async def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    audio_info = None
    chunks = []
    if args.audio:
        pcm, audio_info = prepare_audio(args.audio, args.input_rate)
        chunk_bytes = max(1, round(args.input_rate * args.chunk_ms / 1000)) * 2
        for offset in range(0, len(pcm), chunk_bytes):
            chunk = pcm[offset : offset + chunk_bytes]
            chunks.append(
                (
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode(),
                        }
                    ),
                    len(chunk) / (2 * args.input_rate),
                )
            )
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "input_audio": audio_info,
            "text_sha256": sha256(args.text.encode()),
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": {
                "aiohttp": aiohttp.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "load_policy": "closed-loop; one fresh connection per request; no retries",
            "time_origin": "perf_counter; first_input before send; commit before final send",
            "percentiles": "numpy linear interpolation; descriptive only for small N",
        }
    )
    write_json(args.output_dir / "config.json", config)
    warmups = []
    for index in range(args.warmups):
        state = await run_one(args, chunks, f"warmup-{index:04d}")
        warmups.append(persist(state, args.output_dir, "warmup"))
        print(
            json.dumps(
                {
                    "request_id": state.request_id,
                    "success": state.error is None,
                    "error": state.error,
                }
            ),
            flush=True,
        )
    measured_states = []
    next_index = 0

    async def worker() -> None:
        nonlocal next_index
        while next_index < args.requests:
            index = next_index
            next_index += 1
            state = await run_one(args, chunks, f"request-{index:04d}")
            measured_states.append(state)

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    wall_s = time.perf_counter() - started
    # File I/O must not block other in-flight requests or contaminate throughput.
    results = [persist(state, args.output_dir, "measured") for state in measured_states]
    summary = summarize(results, wall_s)
    summary["warmup_failures"] = sum(not result["success"] for result in warmups)
    summary["input_audio"] = audio_info
    summary["comparison_label"] = args.label
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--url", required=True, help="Full WS or HTTP endpoint URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument(
        "--text",
        default="The quick brown fox jumps over the lazy dog. Today we are testing speech with Dynamo.",
    )
    parser.add_argument("--voice", default="vivian")
    parser.add_argument("--language", help="Defaults to en for ASR and English for TTS")
    parser.add_argument("--input-rate", type=int)
    parser.add_argument("--output-rate", type=int, default=24000)
    parser.add_argument("--modalities", default="audio")
    parser.add_argument("--chunk-ms", type=float, default=100)
    parser.add_argument(
        "--pace",
        type=float,
        default=1,
        help="Audio speed: 1=realtime, 0=unpaced, 2=twice realtime",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--request-extra", type=json.loads, default={})
    parser.add_argument("--session-extra", type=json.loads, default={})
    parser.add_argument("--label", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.profile in INPUT_RATES and args.audio is None:
        parser.error("--audio is required for ASR and Omni profiles")
    args.input_rate = args.input_rate or INPUT_RATES.get(args.profile, 24000)
    args.language = args.language or (
        "en" if args.profile.endswith("asr") else "English"
    )
    if args.requests < 1 or args.concurrency < 1 or args.warmups < 0:
        parser.error("requests/concurrency must be positive and warmups nonnegative")
    if (
        args.pace < 0
        or args.chunk_ms <= 0
        or args.timeout <= 0
        or min(args.input_rate, args.output_rate) <= 0
    ):
        parser.error("Invalid pace, chunk duration, timeout, or sample rate")
    if not isinstance(args.request_extra, dict) or not isinstance(
        args.session_extra, dict
    ):
        parser.error("request-extra/session-extra must be JSON objects")
    return args


if __name__ == "__main__":
    summary_result = asyncio.run(run(parse_args()))
    raise SystemExit(
        1 if summary_result["failures"] or summary_result["warmup_failures"] else 0
    )
