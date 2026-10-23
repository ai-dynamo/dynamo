# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replay caller audio at microphone speed and record the simultaneous reply."""

import argparse
import asyncio
import base64
import json
import math
import re
import time
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def number_words(value):
    small = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
    tens = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()
    if value < 0:
        return "minus " + number_words(-value)
    if value < 20:
        return small[value]
    if value < 100:
        return tens[value // 10] + (" " + small[value % 10] if value % 10 else "")
    for scale, name in ((1000000, "million"), (1000, "thousand"), (100, "hundred")):
        if value >= scale:
            return (
                number_words(value // scale)
                + " "
                + name
                + (" " + number_words(value % scale) if value % scale else "")
            )


def tool_summary(events, require=False):
    calls, reports = {}, []
    for index, event in enumerate(events):
        if event["type"] == "response.function_call_arguments.done":
            call_id = event["call_id"]
            assert call_id not in calls, "Duplicate tool call ID"
            calls[call_id] = event
        elif (
            event["type"] == "conversation.item.added"
            and event["item"]["type"] == "function_call_output"
        ):
            item = event["item"]
            call = calls[item["call_id"]]
            args, output = json.loads(call["arguments"]), json.loads(item["output"])
            assert call["name"] == "generate_random_number", call
            if "error" not in output:
                assert (
                    type(output["result"]) is int
                    and args["min"] <= output["result"] <= args["max"]
                ), (args, output)
            following = events[index + 1 :]
            # Attribute speech before the next call to this result.
            next_call = next(
                (
                    i
                    for i, e in enumerate(following)
                    if e["type"] == "response.function_call_arguments.done"
                ),
                len(following),
            )
            following = following[:next_call]
            reply = "".join(
                e["delta"]
                for e in following
                if e["type"]
                in (
                    "response.output_audio_transcript.delta",
                    "response.output_text.delta",
                )
            )
            words = re.sub(r"\band\b", " ", reply.lower().replace("negative", "minus"))
            words = " ".join(re.findall(r"[a-z]+|-?\d+", words))
            value = output.get("result")
            grounded = type(value) is int and any(
                re.search(r"(?<!\w)" + re.escape(form) + r"(?!\w)", words)
                for form in (str(value), number_words(value))
            )
            reports.append(
                {
                    "call_id": item["call_id"],
                    "name": call["name"],
                    "arguments": args,
                    "output": output,
                    "reply_after_result": reply,
                    "spoken_result_matches": grounded,
                    "audio_after_result": any(
                        e["type"] == "response.output_audio.delta" for e in following
                    ),
                }
            )
    if require:
        assert reports, "No completed native tool call traversed the frontend"
        assert any(
            r["spoken_result_matches"] and r["audio_after_result"] for r in reports
        ), reports
    return reports


async def replay(args):
    args.output.mkdir(parents=True, exist_ok=True)
    audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=True)
    # The vendor dialogue fixture stores caller and agent on separate channels.
    caller = audio[:, 0]
    factor = math.gcd(sample_rate, 16000)
    caller = resample_poly(caller, 16000 // factor, sample_rate // factor)
    if args.seconds:
        caller = caller[: int(args.seconds * 16000)]
    caller = np.concatenate(
        (caller, np.zeros(int(args.tail * 16000), dtype=np.float32))
    )
    caller = np.pad(caller, (0, (-len(caller)) % 1280))
    pcm = (np.clip(caller, -1, 1) * 32767).astype("<i2").tobytes()
    packets, text, times, events = [], [], [], []
    audio_during_input = 0
    input_finished = False
    started = time.perf_counter()
    configured = asyncio.Event()
    drained = asyncio.Event()
    output_rate = 22050

    async with aiohttp.ClientSession() as session:
        url = (
            args.url.rstrip("/")
            .replace("http://", "ws://")
            .replace("https://", "wss://")
            + "/v1/realtime"
        )
        async with session.ws_connect(url, max_msg_size=4 * 1024 * 1024) as ws:
            backend = None
            input_acks = None

            async def receive():
                nonlocal audio_during_input, backend, input_acks
                async for message in ws:
                    if message.type != aiohttp.WSMsgType.TEXT:
                        continue
                    event = json.loads(message.data)
                    now = time.perf_counter() - started
                    events.append({"received_s": now, **event})
                    kind = event["type"]
                    if kind == "session.created":
                        await ws.send_json(
                            {
                                "type": "session.update",
                                "event_id": "configure",
                                "session": {
                                    "type": "realtime",
                                    "model": "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",
                                    "output_modalities": ["audio"],
                                },
                            }
                        )
                    elif kind == "session.updated":
                        assert (
                            event["session"]["model"]
                            == "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"
                        )
                        configured.set()
                    elif kind == "response.output_audio.delta":
                        packets.append(base64.b64decode(event["delta"]))
                        times.append(now)
                        audio_during_input += int(not input_finished)
                    elif kind in (
                        "response.output_text.delta",
                        "response.output_audio_transcript.delta",
                    ):
                        text.append(event["delta"])
                    elif kind == "response.done":
                        metadata = event["response"]["metadata"]
                        backend = metadata["dynamo_worker_id"]
                        input_acks = int(metadata["input_acks"])
                    elif kind == "input_audio_buffer.committed":
                        drained.set()
                    elif kind == "error":
                        raise RuntimeError(event)
                if not drained.is_set():
                    raise RuntimeError(
                        "WebSocket closed before the input-drained receipt"
                    )

            receiver = asyncio.create_task(receive())

            async def wait_for(event, timeout):
                waiting = asyncio.create_task(event.wait())
                try:
                    done, _ = await asyncio.wait(
                        (waiting, receiver),
                        timeout=timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if receiver in done:
                        receiver.result()
                    if not event.is_set():
                        raise TimeoutError("Expected realtime event did not arrive")
                finally:
                    waiting.cancel()
                    await asyncio.gather(waiting, return_exceptions=True)

            try:
                await wait_for(configured, 120)
                sending_started = time.perf_counter()
                for seq, offset in enumerate(range(0, len(pcm), 2560)):
                    if receiver.done():
                        receiver.result()
                    await ws.send_json(
                        {
                            "type": "input_audio_buffer.append",
                            "event_id": f"frame-{seq}",
                            "audio": base64.b64encode(
                                pcm[offset : offset + 2560]
                            ).decode(),
                        }
                    )
                    await asyncio.sleep(
                        max(0, sending_started + (seq + 1) * 0.08 - time.perf_counter())
                    )
                input_finished = True
                await ws.send_json(
                    {"type": "input_audio_buffer.commit", "event_id": "input-end"}
                )
                await wait_for(drained, 120)
                await ws.close()
                await asyncio.wait_for(receiver, 30)
            finally:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
    reply = np.frombuffer(b"".join(packets), dtype="<i2")
    assert len(reply), "No output audio"
    input_frames = len(caller) // 1280
    expected_samples = input_frames * output_rate * 80 // 1000
    assert len(reply) == expected_samples, (len(reply), expected_samples)
    assert backend, "No Dynamo worker identity returned through the realtime engine"
    assert input_acks == input_frames, (input_acks, input_frames)
    rms = float(np.sqrt(np.mean((reply.astype(np.float32) / 32768) ** 2)))
    assert rms > 0.0001, f"Reply contains only silence: RMS={rms}"
    assert "".join(text).strip(), "No spoken text returned"
    assert audio_during_input > 0, "No output arrived while input was streaming"
    sf.write(args.output / "reply.wav", reply, output_rate, subtype="PCM_16")
    (args.output / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )
    intervals = np.diff(times)
    summary = {
        "url": args.url,
        "dynamo_backend": backend,
        "input_seconds": len(caller) / 16000,
        "input_frames": input_frames,
        "input_acks": input_acks,
        "output_seconds": len(reply) / output_rate,
        "output_sample_rate": output_rate,
        "audio_chunks": len(packets),
        "audio_chunks_during_input": audio_during_input,
        "first_audio_s": times[0],
        "audio_interval_p50_ms": float(np.median(intervals) * 1000)
        if len(intervals)
        else 0,
        "audio_interval_p95_ms": float(np.percentile(intervals, 95) * 1000)
        if len(intervals)
        else 0,
        "audio_rms": rms,
        "text": "".join(text),
        "tools": tool_summary(events, args.require_tool),
        "elapsed_s": time.perf_counter() - started,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18898")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument("--tail", type=float, default=4)
    parser.add_argument("--require-tool", action="store_true")
    asyncio.run(replay(parser.parse_args()))
