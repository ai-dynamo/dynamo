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

"""Serve playground assets only; browsers connect directly to dynamo.frontend."""

import json
import os
from pathlib import Path

from aiohttp import ClientError, ClientSession, ClientTimeout, web
from demo_tools import TOOLS

MODEL = "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"


def adapt_asset(path, body):
    if path == "/":
        return body.replace(
            b"<title>Duplex Live</title>", b"<title>Dynamo + Nemotron VoiceChat</title>"
        ).replace(b"full duplex \xc2\xb7 sglang-omni", b"Dynamo + SGLang-Omni")
    if path.endswith("/config.js"):
        port = int(os.environ.get("FRONTEND_PUBLIC_PORT", "18898"))
        body += (
            f"""
window.DEMO_WS_URL = `${{location.protocol === "https:" ? "wss" : "ws"}}://${{location.hostname}}:{port}/v1/realtime`;
window.DEMO_CAPABILITIES_URL = "/v1/realtime/capabilities";
window.DEMO_TOOLS = {json.dumps(TOOLS)};
"""
        ).encode()
    if path.endswith("/session.js"):
        # The adapter validates these fixed native formats before session.updated.
        replacements = {
            "const session = { output_modalities: this.outputModalities };": 'const session = { type: "realtime", model: "'
            + MODEL
            + '", output_modalities: ["audio"], tools: window.DEMO_TOOLS, tool_choice: "auto" };',
            "this.granted = event.session.sglang.granted;": 'this.granted = { interaction: "native", native_full_duplex: true, native_unit_ms: 80, input_modalities: ["audio"], output_modalities: ["audio"], input_audio_format: {type: "audio/pcm", rate: 16000}, output_audio_format: {type: "audio/pcm", rate: 22050} };',
            'this.send("sglang.input_audio.end");': 'this.send("input_audio_buffer.commit");',
            'case "sglang.input_audio.drained":': 'case "input_audio_buffer.committed":',
        }
        for old, new in replacements.items():
            if body.count(old.encode()) != 1:
                raise ValueError(f"Pinned playground source changed: {old}")
            body = body.replace(old.encode(), new.encode())
    if path.endswith("/app.js"):
        body = b'import { observeToolEvent } from "./tools.js";\n' + body
        old_handler = b"await activeSession.handle(JSON.parse(message.data));"
        if body.count(old_handler) != 1:
            raise ValueError("Pinned playground event handler changed")
        body = body.replace(
            old_handler,
            b"const event = JSON.parse(message.data); observeToolEvent(event); await activeSession.handle(event);",
        )
        old = b"transport: (event) => activeSocket.send(JSON.stringify(event)),"
        if body.count(old) != 1:
            raise ValueError("Pinned playground transport changed")
        body = body.replace(
            old,
            b'transport: (event) => event.type === "session.close" ? activeSocket.close() : activeSocket.send(JSON.stringify(event)),',
        )
    return body


async def application():
    app = web.Application()
    native = os.environ.get("OMNI_URL", "http://voicechat-backend:8097")
    frontend = os.environ.get("FRONTEND_URL", "http://127.0.0.1:8080")

    async def clients(app):
        async with ClientSession(timeout=ClientTimeout(total=10)) as session:
            app[CLIENT] = session
            yield

    async def status(request):
        try:
            async with app[CLIENT].get(frontend + "/v1/models") as response:
                models = await response.json()
                ready = any(model["id"] == MODEL for model in models.get("data", []))
        except (ClientError, TimeoutError):
            ready = False
        return web.json_response(
            {"ready": ready, "model": MODEL, "frontend": "dynamo.frontend"},
            status=200 if ready else 503,
        )

    async def asset(request):
        async with app[CLIENT].get(native + request.path) as response:
            body = adapt_asset(request.path, await response.read())
            return web.Response(
                body=body,
                status=response.status,
                headers={
                    "Content-Type": response.headers.get(
                        "Content-Type", "application/octet-stream"
                    ),
                    "Cache-Control": "no-store",
                },
            )

    async def tools_script(request):
        return web.Response(
            body=Path(__file__).with_name("tools.js").read_bytes(),
            content_type="application/javascript",
            headers={"Cache-Control": "no-store"},
        )

    async def tools_catalog(request):
        return web.json_response({"tools": TOOLS, "execution": "server"})

    app.cleanup_ctx.append(clients)
    app.router.add_get("/health", status)
    app.router.add_get("/api/status", status)
    app.router.add_get("/api/tools", tools_catalog)
    app.router.add_get("/static/orb/tools.js", tools_script)
    for path in ("/", "/api/presets", "/v1/realtime/capabilities", "/static/{path:.*}"):
        app.router.add_get(path, asset)
    return app


CLIENT = web.AppKey("client", ClientSession)

if __name__ == "__main__":
    web.run_app(application(), host="0.0.0.0", port=8082)
