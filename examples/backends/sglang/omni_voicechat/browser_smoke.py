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

"""Exercise the actual microphone, AudioWorklet, WebSocket and playback UI."""

import argparse
import asyncio
import base64
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from playwright.async_api import async_playwright
from smoke import tool_summary


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    audio, rate = sf.read(args.audio, dtype="float32", always_2d=True)
    microphone = args.output / "microphone.wav"
    sf.write(
        microphone,
        np.concatenate((audio[:, 0], np.zeros(rate * 6))),
        rate,
        subtype="PCM_16",
    )
    errors, messages, packets, text = [], [], [], []
    sockets, sent, events = [], [], []

    def frame(payload):
        event = json.loads(payload)
        messages.append(event["type"])
        events.append(
            event
            if event["type"] != "response.output_audio.delta"
            else {"type": event["type"]}
        )
        if event["type"] == "response.output_audio.delta":
            packets.append(base64.b64decode(event["delta"]))
        if event["type"] in (
            "response.output_text.delta",
            "response.output_audio_transcript.delta",
        ):
            text.append(event["delta"])
        if event["type"] == "error":
            errors.append(event)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={microphone.resolve()}%noloop",
                "--autoplay-policy=no-user-gesture-required",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
            ]
        )
        page = await browser.new_page(
            viewport={"width": 1440, "height": 1000}, permissions=["microphone"]
        )
        page.on("pageerror", lambda error: errors.append(str(error)))

        def connected(ws):
            sockets.append(ws.url)
            ws.on("framereceived", frame)
            ws.on("framesent", lambda payload: sent.append(json.loads(payload)["type"]))

        page.on("websocket", connected)
        await page.goto(args.url)
        await page.wait_for_function(
            "document.querySelector('#statusText').textContent !== 'checking…'"
        )
        # Observe the actual AudioWorklet-to-speaker path, beyond received bytes.
        await page.evaluate(
            """() => {
          window.audioMeters = {};
          const original = AnalyserNode.prototype.getFloatTimeDomainData;
          AnalyserNode.prototype.getFloatTimeDomainData = function(samples) {
            original.call(this, samples);
            const rate = this.context.sampleRate;
            const meter = window.audioMeters[rate] ||= {windows: 0, invalid: 0, maxRms: 0};
            let sum = 0;
            for (const sample of samples) {
              if (Number.isFinite(sample)) sum += sample * sample;
              else meter.invalid += 1;
            }
            meter.windows += 1;
            meter.maxRms = Math.max(meter.maxRms, Math.sqrt(sum / samples.length));
          };
        }"""
        )
        await page.screenshot(path=str(args.output / "idle.png"), full_page=True)
        await page.locator("#orb").click()
        await page.wait_for_function(
            "document.querySelector('#statusPill').dataset.state === 'live'",
            timeout=120000,
        )
        await asyncio.sleep(args.seconds)
        await page.screenshot(
            path=str(args.output / "conversation.png"), full_page=True
        )
        completed_rows = await page.locator("#toolPanel [data-complete=true]").count()
        meters = await page.evaluate("window.audioMeters")
        await page.locator("#endBtn").click()
        await page.wait_for_timeout(3000)
        await browser.close()
    (args.output / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )
    summary = {
        "websocket_urls": sockets,
        "sent_counts": {kind: sent.count(kind) for kind in sorted(set(sent))},
        "audio_chunks": len(packets),
        "text": "".join(text),
        "event_counts": {kind: messages.count(kind) for kind in sorted(set(messages))},
        "errors": errors,
        "audio_meters": meters,
        "tools": tool_summary(events, args.require_tool),
        "completed_tool_rows": completed_rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.require_tool:
        assert completed_rows > 0, "No tool result displayed in the page"
    assert not errors, errors
    assert packets, "Browser received no audio"
    assert "".join(text).strip(), "Browser received no spoken text"
    assert "input_audio_buffer.append" in sent, "Browser microphone sent no audio"
    assert sockets and all(
        url.endswith(":18898/v1/realtime") for url in sockets
    ), sockets
    playback = meters.get("22050", {})
    assert playback.get("windows", 0) > 0, "No speaker-path samples observed"
    assert playback.get("invalid", 0) == 0, playback
    assert playback.get("maxRms", 0) > 0.0001, "Browser playback was silent"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18897")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=46)
    parser.add_argument("--require-tool", action="store_true")
    asyncio.run(main(parser.parse_args()))
