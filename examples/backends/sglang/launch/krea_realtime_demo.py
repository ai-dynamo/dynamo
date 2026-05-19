#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Demo web server for the Dynamo Krea realtime video worker.

Adapted from https://gist.github.com/GuanLuo/ff3b622a7ea43dad45d73c34f7550bf9 .
The original gist had three tabs (MJPEG Live, MP4 Playback, MP4 Stream); only
the MP4 Stream path works against the realtime worker, so the other two have
been removed.

Five UI fields propagate end-to-end to the realtime worker:

    prompt              → CreateVideoRequest.prompt
                          → RealtimeVideoGenerationsRequest.prompt
    size                → CreateVideoRequest.size
                          → RealtimeVideoGenerationsRequest.size
    fps                 → CreateVideoRequest.nvext.fps
                          → RealtimeVideoGenerationsRequest.fps
    seed                → CreateVideoRequest.nvext.seed
                          → RealtimeVideoGenerationsRequest.seed
                            (defaults to 1024 if omitted — see
                            `_build_realtime_request` in
                            realtime_video_handler.py)
    num_inference_steps → CreateVideoRequest.nvext.num_inference_steps
                          → RealtimeVideoGenerationsRequest.num_inference_steps

Routes:
  GET  /            → HTML UI
  POST /api/video   → POST /v1/videos with stream:true (SSE) → first MP4 chunk
  GET  /api/next    → returns the next buffered SSE chunk
  GET  /api/debug   → current request body / response chunks

Usage:
  python krea_realtime_demo.py [--dynamo URL] [--host HOST] [--port PORT]

Requires: aiohttp (pip install aiohttp).
"""

import argparse
import asyncio
import copy
import json
import logging

from aiohttp import ClientSession, ClientTimeout, web

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Dynamo Krea Realtime Video</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #0f0f0f; color: #e0e0e0;
           padding: 28px; max-width: 860px; }
    h1   { font-size: 1.05rem; font-weight: 600; color: #fff; margin-bottom: 20px; }

    .form-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 14px; }
    label { display: flex; flex-direction: column; gap: 4px; font-size: 0.78rem; color: #777; }
    label.span2 { grid-column: span 2; }
    label.span3 { grid-column: span 3; }
    input, textarea { background: #1a1a1a; color: #e0e0e0; border: 1px solid #2e2e2e;
                      border-radius: 4px; padding: 7px 9px; font-size: 0.85rem; outline: none; }
    input:focus, textarea:focus { border-color: #60a5fa; }
    textarea { resize: vertical; min-height: 52px; }

    .actions { display: flex; gap: 10px; align-items: center; margin-bottom: 14px; }
    button { padding: 7px 18px; border: none; border-radius: 4px; cursor: pointer;
             font-size: 0.83rem; transition: opacity .15s; }
    button:disabled { opacity: .35; cursor: default; }
    .btn-primary { background: #2563eb; color: #fff; }
    .btn-primary:hover:not(:disabled) { background: #1d4ed8; }
    .btn-stop    { background: #374151; color: #d1d5db; }
    .btn-stop:hover:not(:disabled) { background: #4b5563; }
    .status { font-size: 0.78rem; color: #5b6370; }

    .display { border: 1px solid #1a1a1a; border-radius: 6px; background: #000;
               min-height: 300px; display: flex; align-items: center; justify-content: center;
               overflow: hidden; }
    .display img, .display video { max-width: 100%; max-height: 70vh; }
    .placeholder { color: #2a2a2a; font-size: 0.85rem; }
    .spinner { width: 22px; height: 22px; border: 2px solid #1f1f1f;
               border-top-color: #60a5fa; border-radius: 50%;
               animation: spin .7s linear infinite; display: none; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .debug-panel { margin-top: 14px; display: flex; flex-direction: column; gap: 8px; }
    .debug-label { font-size: 0.7rem; color: #4b5563; text-transform: uppercase;
                   letter-spacing: 0.06em; margin-bottom: 3px; }
    .debug-pre { font-family: 'Menlo', 'Consolas', monospace; font-size: 0.72rem;
                 background: #070707; border: 1px solid #1c1c1c; border-radius: 4px;
                 padding: 8px 10px; color: #6ee7b7; white-space: pre-wrap;
                 word-break: break-all; max-height: 160px; overflow-y: auto;
                 min-height: 36px; }
  </style>
</head>
<body>
  <h1>Dynamo Krea Realtime Video</h1>

  <div class="panel">
    <div class="form-grid">
      <label>Model
        <input id="ms-model" value="krea-realtime-video-diffusers" />
      </label>
      <label>Size
        <input id="ms-size" value="832x480" placeholder="e.g. 832x480" />
      </label>
      <label>Seconds
        <input id="ms-seconds" type="number" min="1" value="3" />
      </label>
      <label class="span3">Prompt
        <textarea id="ms-prompt">a cat playing on a sunny beach</textarea>
      </label>
      <label>FPS
        <input id="ms-fps" type="number" min="1" value="12" />
      </label>
      <label>Seed
        <input id="ms-seed" type="number" value="1024" />
      </label>
      <label>Inference steps
        <input id="ms-steps" type="number" min="1" value="4" />
      </label>
    </div>
    <div class="actions">
      <button id="ms-gen" class="btn-primary">&#9654; Generate</button>
      <button id="ms-next" class="btn-stop" disabled>&#9654;&#9654; Next Chunk</button>
      <div    id="ms-spinner" class="spinner"></div>
      <span   id="ms-status" class="status">Idle</span>
    </div>
    <div class="display" id="ms-display">
      <span class="placeholder" id="ms-placeholder">Video chunks will appear here</span>
      <video id="ms-video" controls style="display:none"></video>
    </div>
    <div class="debug-panel">
      <div><div class="debug-label">Request body</div><pre id="ms-req-log" class="debug-pre">—</pre></div>
      <div><div class="debug-label">Response chunks</div><pre id="ms-res-log" class="debug-pre">—</pre></div>
    </div>
  </div>

  <script>
    function detectMime(bytes) {
      if (bytes[0] === 0xFF && bytes[1] === 0xD8) return 'image/jpeg';
      if (bytes.length > 8 &&
          bytes[4] === 0x66 && bytes[5] === 0x74 &&
          bytes[6] === 0x79 && bytes[7] === 0x70) return 'video/mp4';
      return 'application/octet-stream';
    }

    // Build the nvext object that propagates to the realtime worker.
    // Mapping in dynamo.sglang/protocol.py VideoNvExt + the realtime handler's
    // _build_realtime_request: fps, seed, num_inference_steps all flow through
    // to RealtimeVideoGenerationsRequest.
    function buildNvext(fps, seed, steps) {
      const nv = {};
      if (fps)   nv.fps = parseInt(fps);
      if (seed !== '' && seed !== null && seed !== undefined) nv.seed = parseInt(seed);
      if (steps) nv.num_inference_steps = parseInt(steps);
      return nv;
    }

    function readMsForm() {
      const model  = document.getElementById('ms-model').value.trim();
      const prompt = document.getElementById('ms-prompt').value.trim();
      if (!model || !prompt) { alert('Model and Prompt are required.'); return null; }
      const body = { model, prompt, output_format: "mp4", stream: true };
      const size    = document.getElementById('ms-size').value.trim();
      const seconds = document.getElementById('ms-seconds').value;
      if (size)    body.size    = size;
      if (seconds) body.seconds = parseInt(seconds);
      const nv = buildNvext(
        document.getElementById('ms-fps').value,
        document.getElementById('ms-seed').value,
        document.getElementById('ms-steps').value,
      );
      if (Object.keys(nv).length) body.nvext = nv;
      return body;
    }

    // ── MP4 Stream (SSE — Krea realtime path) ───────────────────────────────
    const msGen     = document.getElementById('ms-gen');
    const msNext    = document.getElementById('ms-next');
    const msSpinner = document.getElementById('ms-spinner');
    const msStatus  = document.getElementById('ms-status');
    const msVideo   = document.getElementById('ms-video');
    const msPh      = document.getElementById('ms-placeholder');
    let msPrevUrl   = null;

    async function playClip(json, videoEl, prevUrlRef) {
      let url, mime;
      if (json.b64_json) {
        const bin   = atob(json.b64_json);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        mime = detectMime(bytes);
        if (prevUrlRef.value) URL.revokeObjectURL(prevUrlRef.value);
        url = URL.createObjectURL(new Blob([bytes], { type: mime }));
        prevUrlRef.value = url;
      } else if (json.url) {
        url = json.url;
        mime = 'video/mp4';
      } else {
        throw new Error('Response has no b64_json or url');
      }
      videoEl.src = url;
      videoEl.style.display = 'block';
      videoEl.play().catch(() => {});
    }

    const msPrevRef = { value: null };

    msGen.addEventListener('click', async () => {
      const body = readMsForm();
      if (!body) return;
      msGen.disabled = true;
      msSpinner.style.display = 'block';
      msStatus.textContent = 'Generating…';
      msPh.style.display = 'none';
      msVideo.style.display = 'none';
      try {
        const resp = await fetch('/api/video', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const j = await resp.json();
        if (!resp.ok) throw new Error(j.error || 'Dynamo error ' + resp.status);
        await playClip(j, msVideo, msPrevRef);
        msStatus.textContent = 'Playing chunk 1.';
        msNext.disabled = false;
      } catch (e) {
        msStatus.textContent = 'Error: ' + e.message;
      } finally {
        msGen.disabled = false;
        msSpinner.style.display = 'none';
      }
    });

    async function fetchNextStreamClip() {
      msNext.disabled = true;
      try {
        const resp = await fetch('/api/next');
        if (resp.status === 404) {
          msStatus.textContent = 'Stream complete.';
          return;
        }
        const j = await resp.json();
        if (!resp.ok) throw new Error(j.error || 'Error ' + resp.status);
        await playClip(j, msVideo, msPrevRef);
        msStatus.textContent = 'Playing next chunk.';
        msNext.disabled = false;
      } catch (e) {
        msStatus.textContent = 'Error: ' + e.message;
      }
    }

    msNext.addEventListener('click', fetchNextStreamClip);
    msVideo.addEventListener('ended', () => { if (!msNext.disabled) fetchNextStreamClip(); });

    // ── Debug panel polling ─────────────────────────────────────────────────
    const reqLog = document.getElementById('ms-req-log');
    const resLog = document.getElementById('ms-res-log');
    let _lastReq = null, _lastChunks = null;
    setInterval(async () => {
      try {
        const d = await fetch('/api/debug').then(r => r.json());
        if (d.request !== _lastReq) {
          reqLog.textContent = d.request || '—';
          _lastReq = d.request;
        }
        if (d.chunks !== _lastChunks) {
          resLog.textContent = d.chunks || '—';
          resLog.scrollTop = resLog.scrollHeight;
          _lastChunks = d.chunks;
        }
      } catch (_) {}
    }, 100);
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def handle_index(request: web.Request) -> web.Response:
    return web.Response(text=_HTML, content_type="text/html")


async def handle_video(request: web.Request) -> web.Response:
    """POST /api/video → Dynamo POST /v1/videos.

    Pass-through proxy. Body is sent verbatim — prompt/size/seconds/nvext fields
    propagate as-is to the realtime worker via Dynamo's CreateVideoRequest.

    When body["stream"] is true the upstream is SSE: we consume the event
    stream into request.app["clips"] and return the first chunk immediately.
    Subsequent chunks are fetched via /api/next.
    """
    body = await request.json()
    streaming = bool(body.get("stream"))

    debug = request.app["debug"]
    debug["request"] = json.dumps(body, indent=2)
    debug["chunks"] = ""
    request.app["clips"] = []
    request.app["next_clip"] = 1

    dynamo_url = request.app["dynamo_url"]
    timeout = ClientTimeout(total=300, connect=10)

    try:
        if streaming:
            first_clip_ready = asyncio.Event()

            async def consume_sse() -> None:
                try:
                    async with ClientSession(timeout=timeout) as session:
                        async with session.post(
                            f"{dynamo_url}/v1/videos", json=body
                        ) as upstream:
                            buf = ""
                            async for chunk in upstream.content.iter_any():
                                text = chunk.decode("utf-8", errors="replace")
                                buf += text
                                lines = buf.split("\n")
                                buf = lines[-1]
                                for line in lines[:-1]:
                                    line = line.rstrip("\r")
                                    if not line.startswith("data: "):
                                        continue
                                    data_str = line[6:].strip()
                                    if data_str == "[DONE]":
                                        continue
                                    try:
                                        event = json.loads(data_str)
                                        if isinstance(event.get("data"), list):
                                            for item in event["data"]:
                                                request.app["clips"].append(item.copy())
                                                if "b64_json" in item:
                                                    item[
                                                        "b64_json"
                                                    ] = f"<video_data_{len(request.app['clips'])-1}>"
                                                if "url" in item:
                                                    item[
                                                        "url"
                                                    ] = f"<video_url_{len(request.app['clips'])-1}>"
                                        elif "b64_json" in event or "url" in event:
                                            request.app["clips"].append(event.copy())
                                            if "b64_json" in event:
                                                event[
                                                    "b64_json"
                                                ] = f"<video_data_{len(request.app['clips'])-1}>"
                                            if "url" in event:
                                                event[
                                                    "url"
                                                ] = f"<video_url_{len(request.app['clips'])-1}>"
                                        debug["chunks"] += json.dumps(event, indent=2)
                                    except Exception:
                                        pass
                                    if (
                                        request.app["clips"]
                                        and not first_clip_ready.is_set()
                                    ):
                                        first_clip_ready.set()
                except Exception as exc:
                    logger.error("SSE consume error: %s", exc)
                finally:
                    if not first_clip_ready.is_set():
                        first_clip_ready.set()

            asyncio.create_task(consume_sse())
            await first_clip_ready.wait()

            if not request.app["clips"]:
                raise web.HTTPBadGateway(reason="SSE stream ended without a video clip")
            return web.json_response(request.app["clips"][0])

        async with ClientSession(timeout=timeout) as session:
            async with session.post(f"{dynamo_url}/v1/videos", json=body) as upstream:
                if upstream.status != 200:
                    text = await upstream.text()
                    raise web.HTTPBadGateway(
                        reason=f"Dynamo error {upstream.status}: {text}"
                    )
                json_body = await upstream.json()
                if isinstance(json_body.get("data"), list):
                    for item in json_body["data"]:
                        request.app["clips"].append(item)
                sanitized = copy.deepcopy(json_body)
                if isinstance(sanitized.get("data"), list):
                    for i, item in enumerate(sanitized["data"]):
                        if "b64_json" in item:
                            item["b64_json"] = f"<video_data_{i}>"
                        if "url" in item:
                            item["url"] = f"<video_url_{i}>"
                debug["chunks"] = json.dumps(sanitized, indent=2)
                return web.json_response(
                    request.app["clips"][0], status=upstream.status
                )
    except web.HTTPException:
        raise
    except Exception as exc:
        logger.error("Video proxy error: %s", exc)
        raise web.HTTPBadGateway(reason=str(exc))


async def handle_next(request: web.Request) -> web.Response:
    """GET /api/next → return the next buffered video clip (same shape as /api/video)."""
    clips = request.app["clips"]
    if len(clips) == 0 or request.app["next_clip"] >= len(clips):
        return web.json_response({"error": "No clips available"}, status=404)
    clip = clips[request.app["next_clip"]]
    request.app["next_clip"] += 1
    return web.json_response(clip, status=200)


async def handle_debug(request: web.Request) -> web.Response:
    return web.json_response(request.app["debug"])


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def make_app(dynamo_url: str) -> web.Application:
    app = web.Application()
    app["dynamo_url"] = dynamo_url.rstrip("/")
    app["debug"] = {"request": "", "chunks": ""}
    app["clips"] = []
    app["next_clip"] = 1
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/debug", handle_debug)
    app.router.add_get("/api/next", handle_next)
    app.router.add_post("/api/video", handle_video)
    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo web server for the Dynamo Krea realtime video worker"
    )
    parser.add_argument(
        "--dynamo", default="http://localhost:8000", help="Dynamo HTTP server URL"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8888, help="Listen port")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Proxying to Dynamo at %s", args.dynamo)
    logger.info("Open http://localhost:%d in your browser", args.port)
    web.run_app(make_app(args.dynamo), host=args.host, port=args.port)
