# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client for the Dynamo vLLM-Omni realtime endpoint (``/v1/realtime``).

Streams an audio file to a launched ``dynamo.frontend`` over a WebSocket using
OpenAI Realtime client events, prints every server event as it arrives, and
saves the synthesized audio.

The wire contract is the Dynamo/OpenAI-spec realtime envelope emitted by
``RealtimeOmniHandler`` (``response.output_audio.delta`` etc.) — not vLLM-Omni's
native ``response.audio.delta`` names.

Any audio file readable by ``soundfile`` works; it is downmixed to mono and
linearly resampled to 16 kHz PCM16 (the realtime input format). When
``--input-audio`` is omitted, a sample clip is fetched from the vLLM-Omni
GitHub repo (``tests/assets/qwen3_tts/clone_2.wav``).

python -m dynamo.frontend --http-port 8000
python -m dynamo.vllm.omni --realtime --model Qwen/Qwen3-Omni-30B-A3B-Instruct

Usage (omit --input-audio to fetch the sample clip from GitHub):
  python realtime_omni_client.py \
      --url ws://localhost:8000/v1/realtime \
      --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
      --output-wav response.wav

  # or point at your own audio file:
  python realtime_omni_client.py --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
      --input-audio /path/to/your.wav
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

INPUT_SAMPLE_RATE = 16000  # realtime input format: 16 kHz mono PCM16

# Sample input fetched when --input-audio is omitted: a Qwen3-TTS voice-clone
# reference clip from the vLLM-Omni GitHub repo.
DEFAULT_AUDIO_URL = (
    "https://raw.githubusercontent.com/vllm-project/vllm-omni/main/"
    "tests/assets/qwen3_tts/clone_2.wav"
)


def _read_audio(path: str) -> tuple[np.ndarray, int]:
    """Read audio as float32 + sample rate via soundfile, or stdlib wave (PCM16)."""
    try:
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        return audio, sr
    except ImportError:
        with wave.open(path, "rb") as wf:
            if wf.getsampwidth() != 2:
                raise SystemExit(
                    "soundfile not installed and input is not 16-bit PCM WAV; "
                    "install soundfile or pre-convert to PCM16 "
                    "(ffmpeg -i in.wav -ac 1 -ar 16000 -sample_fmt s16 out.wav)"
                )
            sr = wf.getframerate()
            frames = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
            audio = frames.astype(np.float32) / 32768.0
            return audio.reshape(-1, wf.getnchannels()), sr


def _fetch_default_audio(url: str) -> str:
    """Download the sample clip from GitHub to a temp file; return its path."""
    dest = os.path.join(tempfile.gettempdir(), "realtime_omni_" + os.path.basename(url))
    if not os.path.exists(dest):
        print(f"[client] no --input-audio given; fetching sample from {url}")
        urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed https GitHub URL
    else:
        print(f"[client] using cached sample {dest}")
    return dest


def _load_pcm16_16k(path: str | None, default_url: str) -> bytes:
    """Load an audio file as 16 kHz mono PCM16 bytes, fetching a sample if needed."""
    if path is None:
        path = _fetch_default_audio(default_url)
    elif not os.path.isfile(path):
        raise SystemExit(
            f"--input-audio file not found: {path!r}. Pass a real audio file, "
            "or omit --input-audio to fetch the vLLM-Omni sample clip."
        )

    audio, sr = _read_audio(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix to mono
    if sr != INPUT_SAMPLE_RATE:
        # Linear resample to 16 kHz (adequate for a demo client).
        duration = audio.shape[0] / sr
        tgt_len = int(duration * INPUT_SAMPLE_RATE)
        xp = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
        x = np.linspace(0.0, duration, num=tgt_len, endpoint=False)
        audio = np.interp(x, xp, audio).astype(np.float32)
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2")
    print(
        f"[client] loaded {path}: {len(pcm16)} samples "
        f"({len(pcm16) / INPUT_SAMPLE_RATE:.2f}s) @ {INPUT_SAMPLE_RATE} Hz mono"
    )
    return pcm16.tobytes()


def _write_wav(path: str, pcm16_bytes: bytes, sample_rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16_bytes)


async def run(args: argparse.Namespace) -> int:
    pcm16 = _load_pcm16_16k(args.input_audio, args.input_audio_url)
    chunk_bytes = max(INPUT_SAMPLE_RATE * 2 // 1000 * args.chunk_ms, 2)

    audio_out = bytearray()
    transcript = []
    response_id = None
    status = None

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(args.url, max_msg_size=64 * 1024 * 1024) as ws:
            print(f"[client] connected to {args.url}")

            # 1) Select the model (the frontend picks the engine on session.update).
            await ws.send_str(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {"type": "realtime", "model": args.model},
                    }
                )
            )

            # 2) Stream the audio in chunks, then commit to trigger generation.
            for i in range(0, len(pcm16), chunk_bytes):
                await ws.send_str(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(
                                pcm16[i : i + chunk_bytes]
                            ).decode(),
                        }
                    )
                )
            await ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))
            print(f"[client] sent {len(pcm16)} bytes of audio + commit\n")

            # 3) Print every server event until the response completes.
            while True:
                msg = await asyncio.wait_for(ws.receive(), timeout=args.timeout)
                if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    print(f"[client] socket closed: {msg.data!r} {msg.extra!r}")
                    break
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    continue
                event = json.loads(msg.data)
                etype = event.get("type")

                if etype == "response.output_audio.delta":
                    delta = base64.b64decode(event.get("delta", ""))
                    audio_out.extend(delta)
                    print(f"<- response.output_audio.delta ({len(delta)} bytes)")
                elif etype == "response.output_audio_transcript.delta":
                    transcript.append(event.get("delta", ""))
                    print(f"<- transcript.delta: {event.get('delta')!r}")
                elif etype == "response.created":
                    response_id = event["response"]["id"]
                    print(f"<- response.created (id={response_id})")
                elif etype == "response.done":
                    status = event["response"]["status"]
                    print(f"<- response.done (status={status})")
                    break
                elif etype == "error":
                    print(f"<- ERROR: {json.dumps(event.get('error'), indent=2)}")
                    break
                else:
                    print(f"<- {etype}")

    print("\n[client] === summary ===")
    print(f"  response_id : {response_id}")
    print(f"  status      : {status}")
    print(f"  transcript  : {''.join(transcript)!r}")
    print(f"  audio bytes : {len(audio_out)} ({len(audio_out) / 2} samples)")
    if audio_out:
        _write_wav(args.output_wav, bytes(audio_out), args.output_sample_rate)
        print(f"  saved audio : {args.output_wav} @ {args.output_sample_rate} Hz")
    return 0 if status == "completed" else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://localhost:8000/v1/realtime")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument(
        "--input-audio", default=None, help="any soundfile-readable audio"
    )
    parser.add_argument(
        "--input-audio-url",
        default=DEFAULT_AUDIO_URL,
        help="sample audio URL fetched when --input-audio is omitted",
    )
    parser.add_argument("--output-wav", default="realtime_response.wav")
    parser.add_argument(
        "--output-sample-rate",
        type=int,
        default=24000,
        help="sample rate to save the response audio at (Omni talker default 24kHz)",
    )
    parser.add_argument(
        "--chunk-ms", type=int, default=100, help="append chunk size in ms"
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="per-frame recv timeout"
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
