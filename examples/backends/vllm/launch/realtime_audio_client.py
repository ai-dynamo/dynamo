# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Send file audio to Dynamo's OpenAI-compatible ``/v1/realtime`` endpoint.

The client supports two session types:

* ``realtime`` sends 16 kHz PCM16 to a vLLM-Omni worker and saves returned
  ``response.output_audio.delta`` events as WAV files.
* ``transcription`` sends 24 kHz PCM16 to a standard vLLM ASR worker and prints
  the returned transcription delta and completed events.

Any audio file readable by ``soundfile`` works. Without ``--input-audio``, the
client downloads a small speech sample from the vLLM-Omni repository.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import tempfile
import urllib.request
import wave

import aiohttp
import numpy as np

REALTIME_INPUT_SAMPLE_RATE = 16_000
TRANSCRIPTION_INPUT_SAMPLE_RATE = 24_000
DEFAULT_AUDIO_URL = (
    "https://raw.githubusercontent.com/vllm-project/vllm-omni/main/"
    "tests/assets/qwen3_tts/clone_2.wav"
)


def _read_audio(path: str) -> tuple[np.ndarray, int]:
    """Read audio as float32 plus its sample rate."""
    try:
        import soundfile as sf

        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        return audio, sample_rate
    except ImportError:
        with wave.open(path, "rb") as wav_file:
            if wav_file.getsampwidth() != 2:
                raise SystemExit(
                    "soundfile is not installed and input is not 16-bit PCM WAV; "
                    "install soundfile or convert the input to PCM16"
                )
            sample_rate = wav_file.getframerate()
            frames = np.frombuffer(
                wav_file.readframes(wav_file.getnframes()), dtype="<i2"
            )
            audio = frames.astype(np.float32) / 32768.0
            return audio.reshape(-1, wav_file.getnchannels()), sample_rate


def _fetch_default_audio(url: str) -> str:
    """Download the default speech sample once and return its path."""
    if not url.startswith("https://"):
        raise SystemExit("--input-audio-url must use HTTPS")
    destination = os.path.join(
        tempfile.gettempdir(), "dynamo_realtime_" + os.path.basename(url)
    )
    if not os.path.exists(destination):
        print(f"[client] no --input-audio given; fetching sample from {url}")
        urllib.request.urlretrieve(  # noqa: S310 - user-selectable HTTPS audio URL
            url, destination
        )
    else:
        print(f"[client] using cached sample {destination}")
    return destination


def _load_pcm16(path: str | None, default_url: str, sample_rate: int) -> bytes:
    """Load and resample an audio file as mono PCM16 bytes."""
    if path is None:
        path = _fetch_default_audio(default_url)
    elif not os.path.isfile(path):
        raise SystemExit(
            f"--input-audio file not found: {path!r}. Pass a valid audio file "
            "or omit --input-audio to fetch the sample clip."
        )

    audio, source_rate = _read_audio(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if source_rate != sample_rate:
        duration = audio.shape[0] / source_rate
        target_length = int(duration * sample_rate)
        source_points = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
        target_points = np.linspace(0.0, duration, num=target_length, endpoint=False)
        audio = np.interp(target_points, source_points, audio).astype(np.float32)

    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    print(
        f"[client] loaded {path}: {len(pcm16)} samples "
        f"({len(pcm16) / sample_rate:.2f}s) @ {sample_rate} Hz mono"
    )
    return pcm16.tobytes()


def _write_wav(path: str, pcm16_bytes: bytes, sample_rate: int) -> None:
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16_bytes)


def _input_sample_rate(session_type: str) -> int:
    if session_type == "transcription":
        return TRANSCRIPTION_INPUT_SAMPLE_RATE
    return REALTIME_INPUT_SAMPLE_RATE


def _session_block(args: argparse.Namespace) -> dict:
    if args.session_type == "transcription":
        return {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": TRANSCRIPTION_INPUT_SAMPLE_RATE,
                    },
                    "transcription": {
                        "model": args.model,
                        "language": args.language,
                    },
                    "noise_reduction": None,
                    "turn_detection": None,
                }
            },
        }

    return {
        "type": "realtime",
        "model": args.model,
        "output_modalities": args.output_modalities or ["audio"],
    }


async def run(args: argparse.Namespace) -> int:
    input_sample_rate = _input_sample_rate(args.session_type)
    pcm16 = _load_pcm16(args.input_audio, args.input_audio_url, input_sample_rate)
    chunk_bytes = max(input_sample_rate * 2 * args.chunk_ms // 1000, 2)

    if args.session_type == "realtime":
        os.makedirs(args.output_dir, exist_ok=True)

    audio_out = bytearray()
    audio_delta_count = 0
    transcript_parts: list[str] = []
    transcript = ""
    response_id = None
    status = None

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(args.url, max_msg_size=64 * 1024 * 1024) as ws:
            print(f"[client] connected to {args.url}")
            await ws.send_str(
                json.dumps({"type": "session.update", "session": _session_block(args)})
            )

            for offset in range(0, len(pcm16), chunk_bytes):
                chunk = pcm16[offset : offset + chunk_bytes]
                await ws.send_str(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode(),
                        }
                    )
                )
                print(
                    f"[client] sent {min(offset + len(chunk), len(pcm16))} "
                    "bytes of audio"
                )
                await asyncio.sleep(args.chunk_ms / 1000)

            await ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))
            print("[client] committed audio")

            while True:
                message = await asyncio.wait_for(ws.receive(), timeout=args.timeout)
                if message.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    print(
                        f"[client] socket closed: {message.data!r} "
                        f"{message.extra!r}"
                    )
                    break
                if message.type is not aiohttp.WSMsgType.TEXT:
                    continue

                event = json.loads(message.data)
                event_type = event.get("type")
                if event_type == "response.output_audio.delta":
                    delta = base64.b64decode(event.get("delta", ""))
                    if delta:
                        audio_delta_count += 1
                        audio_out.extend(delta)
                        chunk_path = os.path.join(
                            args.output_dir, f"chunk_{audio_delta_count:04d}.wav"
                        )
                        _write_wav(chunk_path, delta, args.output_sample_rate)
                        print(
                            f"<- {event_type} ({len(delta)} bytes) -> "
                            f"{os.path.basename(chunk_path)}"
                        )
                elif event_type in (
                    "response.output_audio_transcript.delta",
                    "conversation.item.input_audio_transcription.delta",
                ):
                    transcript_parts.append(event.get("delta", ""))
                    print(f"<- {event_type}: {event.get('delta')!r}")
                elif event_type == (
                    "conversation.item.input_audio_transcription.completed"
                ):
                    transcript = event.get("transcript", "")
                    status = "completed"
                    print(f"<- {event_type}: {transcript!r}")
                    break
                elif event_type == "response.created":
                    response_id = event["response"]["id"]
                    print(f"<- response.created (id={response_id})")
                elif event_type == "response.done":
                    status = event["response"]["status"]
                    print(f"<- response.done (status={status})")
                    break
                elif event_type == "error":
                    print(f"<- ERROR: {json.dumps(event.get('error'), indent=2)}")
                    break
                else:
                    print(f"<- {event_type}")

    transcript = transcript or "".join(transcript_parts)
    print("\n[client] === summary ===")
    print(f"  session     : {args.session_type}")
    print(f"  status      : {status}")
    print(f"  transcript  : {transcript!r}")

    if args.session_type == "realtime":
        print(f"  response_id : {response_id}")
        print(
            f"  audio       : {audio_delta_count} delta(s) joined -> "
            f"{len(audio_out)} bytes ({len(audio_out) // 2} samples)"
        )
        if audio_out:
            output_path = os.path.join(args.output_dir, "response.wav")
            _write_wav(output_path, bytes(audio_out), args.output_sample_rate)
            print(f"  saved audio : {output_path} @ {args.output_sample_rate} Hz")
        else:
            print("  saved audio : none")

    return 0 if status == "completed" else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://localhost:8000/v1/realtime")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument(
        "--session-type",
        choices=["realtime", "transcription"],
        default="realtime",
        help="realtime returns model responses; transcription returns ASR text",
    )
    parser.add_argument(
        "--input-audio", default=None, help="any soundfile-readable audio"
    )
    parser.add_argument(
        "--input-audio-url",
        default=DEFAULT_AUDIO_URL,
        help="sample audio URL fetched when --input-audio is omitted",
    )
    parser.add_argument("--language", default="en", help="transcription input language")
    parser.add_argument(
        "--output-modalities",
        nargs="+",
        choices=["audio", "text"],
        default=None,
        help="realtime session output modalities (default: audio)",
    )
    parser.add_argument(
        "--output-dir",
        default="realtime_output",
        help="directory for realtime response WAV files",
    )
    parser.add_argument(
        "--output-sample-rate",
        type=int,
        default=24_000,
        help="sample rate for realtime response WAV files",
    )
    parser.add_argument(
        "--chunk-ms", type=int, default=100, help="audio append chunk size"
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="per-frame receive timeout"
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
