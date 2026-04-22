#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Proxy web server for Dynamo video demos.

Two UI tabs:
  MJPEG Live   — GET /api/stream?model=...&prompt=... proxies to Dynamo POST /v1/videos/stream.
                 Browser uses native img.src = '/api/stream?...' for MJPEG display.
  MP4 Playback — calls POST /v1/videos (non-streaming), plays back b64_json or url payload.

Routes:
  GET  /            → HTML UI
  GET  /api/stream  → query-params → POST /v1/videos/stream (multipart/x-mixed-replace)
  POST /api/video   → proxy to Dynamo POST /v1/videos (non-streaming JSON)

Usage:
  python server.py [--dynamo URL] [--host HOST] [--port PORT]
"""

import argparse
import asyncio
import copy
import json
import logging
import re

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
  <title>Dynamo Video</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #0f0f0f; color: #e0e0e0;
           padding: 28px; max-width: 860px; }
    h1   { font-size: 1.05rem; font-weight: 600; color: #fff; margin-bottom: 20px; }

    /* tabs */
    .tabs { display: flex; border-bottom: 1px solid #222; margin-bottom: 22px; }
    .tab-btn { background: none; border: none; border-bottom: 2px solid transparent;
               padding: 8px 18px; cursor: pointer; font-size: 0.83rem; color: #666;
               margin-bottom: -1px; transition: color .15s, border-color .15s; }
    .tab-btn:hover  { color: #aaa; }
    .tab-btn.active { color: #60a5fa; border-bottom-color: #60a5fa; }
    .tab { display: none; }
    .tab.active { display: block; }

    /* form */
    .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px; }
    label { display: flex; flex-direction: column; gap: 4px; font-size: 0.78rem; color: #777; }
    label.span2 { grid-column: span 2; }
    input, textarea { background: #1a1a1a; color: #e0e0e0; border: 1px solid #2e2e2e;
                      border-radius: 4px; padding: 7px 9px; font-size: 0.85rem; outline: none; }
    input:focus, textarea:focus { border-color: #60a5fa; }
    textarea { resize: vertical; min-height: 52px; }

    /* actions */
    .actions { display: flex; gap: 10px; align-items: center; margin-bottom: 14px; }
    button { padding: 7px 18px; border: none; border-radius: 4px; cursor: pointer;
             font-size: 0.83rem; transition: opacity .15s; }
    button:disabled { opacity: .35; cursor: default; }
    .btn-primary { background: #2563eb; color: #fff; }
    .btn-primary:hover:not(:disabled) { background: #1d4ed8; }
    .btn-stop    { background: #374151; color: #d1d5db; }
    .btn-stop:hover:not(:disabled) { background: #4b5563; }
    .status { font-size: 0.78rem; color: #5b6370; }

    /* display */
    .display { border: 1px solid #1a1a1a; border-radius: 6px; background: #000;
               min-height: 300px; display: flex; align-items: center; justify-content: center;
               overflow: hidden; }
    .display img, .display video { max-width: 100%; max-height: 70vh; }
    .placeholder { color: #2a2a2a; font-size: 0.85rem; }
    .spinner { width: 22px; height: 22px; border: 2px solid #1f1f1f;
               border-top-color: #60a5fa; border-radius: 50%;
               animation: spin .7s linear infinite; display: none; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .mock-warn { display: none; font-size: 0.75rem; color: #f59e0b;
                 background: #1c1407; border: 1px solid #78350f;
                 border-radius: 4px; padding: 6px 10px; margin-bottom: 10px; }
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
  <h1>Dynamo Video</h1>

  <div class="tabs">
    <button class="tab-btn active" data-tab="mjpeg">MJPEG Live</button>
    <button class="tab-btn"       data-tab="mp4">MP4 Playback</button>
  </div>

  <!-- ── MJPEG Live ── -->
  <div id="tab-mjpeg" class="tab active">
    <div id="mj-mock-warn" class="mock-warn">
      Mock model: always streams the same pre-recorded video — prompt, size, fps, and seconds are ignored.
    </div>
    <div class="form-grid">
      <label>Model
        <input id="mj-model" value="video-stream-model" />
      </label>
      <label>Size
        <input id="mj-size" placeholder="e.g. 832x480" />
      </label>
      <label class="span2">Prompt
        <textarea id="mj-prompt">a cat playing on a sunny beach</textarea>
      </label>
      <label>Seconds
        <input id="mj-seconds" type="number" min="1" placeholder="e.g. 5" />
      </label>
      <label>FPS
        <input id="mj-fps" type="number" min="1" placeholder="e.g. 25" />
      </label>
    </div>
    <div class="actions">
      <button id="mj-start" class="btn-primary">&#9654; Start</button>
      <button id="mj-stop"  class="btn-stop" disabled>&#9632; Stop</button>
      <span   id="mj-status" class="status">Idle</span>
    </div>
    <div class="display" id="mj-display">
      <span class="placeholder" id="mj-placeholder">Stream will appear here</span>
      <img id="mj-frame" alt="" style="display:none" />
    </div>
    <div class="debug-panel">
      <div><div class="debug-label">Request body</div><pre id="mj-req-log" class="debug-pre">—</pre></div>
      <div><div class="debug-label">Response chunks</div><pre id="mj-res-log" class="debug-pre">—</pre></div>
    </div>
  </div>

  <!-- ── MP4 Playback ── -->
  <div id="tab-mp4" class="tab">
    <div id="mp-mock-warn" class="mock-warn">
      Mock model: always returns the same pre-recorded video — prompt, size, and seconds are ignored.
    </div>
    <div class="form-grid">
      <label>Model
        <input id="mp-model" value="video-stream-model" />
      </label>
      <label>Size
        <input id="mp-size" placeholder="e.g. 832x480" />
      </label>
      <label class="span2">Prompt
        <textarea id="mp-prompt">a cat playing on a sunny beach</textarea>
      </label>
      <label>Seconds
        <input id="mp-seconds" type="number" min="1" placeholder="e.g. 5" />
      </label>
    </div>
    <div class="actions">
      <button id="mp-gen" class="btn-primary">&#9654; Generate</button>
      <button id="mp-next" class="btn-stop" disabled>&#9654;&#9654; Next Clip</button>
      <div    id="mp-spinner" class="spinner"></div>
      <span   id="mp-status" class="status">Idle</span>
    </div>
    <div class="display" id="mp-display">
      <span class="placeholder" id="mp-placeholder">Video will appear here</span>
      <img   id="mp-img"   alt="" style="display:none" />
      <video id="mp-video" controls style="display:none"></video>
    </div>
    <div class="debug-panel">
      <div><div class="debug-label">Request body</div><pre id="mp-req-log" class="debug-pre">—</pre></div>
      <div><div class="debug-label">Response chunks</div><pre id="mp-res-log" class="debug-pre">—</pre></div>
    </div>
  </div>

  <script>
    // ── Mock-model warning ───────────────────────────────────────────────────
    const MOCK_MODEL = 'video-stream-model';
    function bindMockWarn(inputId, warnId) {
      const inp  = document.getElementById(inputId);
      const warn = document.getElementById(warnId);
      function update() { warn.style.display = inp.value.trim() === MOCK_MODEL ? 'block' : 'none'; }
      inp.addEventListener('input', update);
      update();
    }
    bindMockWarn('mj-model', 'mj-mock-warn');
    bindMockWarn('mp-model', 'mp-mock-warn');

    // ── Tab switching ────────────────────────────────────────────────────────
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
      });
    });

    // ── Helpers ──────────────────────────────────────────────────────────────

    // Detect MIME type from magic bytes.
    function detectMime(bytes) {
      if (bytes[0] === 0xFF && bytes[1] === 0xD8) return 'image/jpeg';
      // MP4 ftyp box: 4-byte size + "ftyp" at offset 4
      if (bytes.length > 8 &&
          bytes[4] === 0x66 && bytes[5] === 0x74 &&
          bytes[6] === 0x79 && bytes[7] === 0x70) return 'video/mp4';
      return 'application/octet-stream';
    }

    // ── MJPEG Live ───────────────────────────────────────────────────────────
    const mjStart  = document.getElementById('mj-start');
    const mjStop   = document.getElementById('mj-stop');
    const mjStatus = document.getElementById('mj-status');
    const mjFrame  = document.getElementById('mj-frame');
    const mjPh     = document.getElementById('mj-placeholder');

    mjStart.addEventListener('click', () => {
      const model  = document.getElementById('mj-model').value.trim();
      const prompt = document.getElementById('mj-prompt').value.trim();
      if (!model || !prompt) { alert('Model and Prompt are required.'); return; }

      const params = new URLSearchParams({ model, prompt });
      const size    = document.getElementById('mj-size').value.trim();
      const seconds = document.getElementById('mj-seconds').value;
      const fps     = document.getElementById('mj-fps').value;
      if (size)    params.set('size', size);
      if (seconds) params.set('seconds', seconds);
      if (fps)     params.set('fps', fps);

      // Native MJPEG: browser streams multipart/x-mixed-replace from the GET endpoint.
      mjFrame.src = '/api/stream?' + params.toString();
      mjFrame.style.display = 'block';
      mjPh.style.display    = 'none';
      mjStart.disabled      = true;
      mjStop.disabled       = false;
      mjStatus.textContent  = 'Streaming…';
    });

    mjFrame.addEventListener('error', () => {
      mjStatus.textContent = 'Stream ended or error.';
      mjStart.disabled     = false;
      mjStop.disabled      = true;
    });

    mjStop.addEventListener('click', () => {
      mjFrame.src           = '';
      mjFrame.style.display = 'none';
      mjPh.style.display    = 'block';
      mjStart.disabled      = false;
      mjStop.disabled       = true;
      mjStatus.textContent  = 'Stopped.';
    });

    // ── MP4 Playback ─────────────────────────────────────────────────────────
    const mpGen     = document.getElementById('mp-gen');
    const mpNext    = document.getElementById('mp-next');
    const mpSpinner = document.getElementById('mp-spinner');
    const mpStatus  = document.getElementById('mp-status');
    const mpImg     = document.getElementById('mp-img');
    const mpVideo   = document.getElementById('mp-video');
    const mpPh      = document.getElementById('mp-placeholder');
    let   mpPrevUrl = null;

    async function playVideoResponse(json) {
      let url, mime;
      if (json.b64_json) {
        const bin   = atob(json.b64_json);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        mime = detectMime(bytes);
        if (mpPrevUrl) URL.revokeObjectURL(mpPrevUrl);
        url = URL.createObjectURL(new Blob([bytes], { type: mime }));
        mpPrevUrl = url;
      } else if (json.url) {
        url  = json.url;
        mime = url.endsWith('.jpg') || url.endsWith('.jpeg') ? 'image/jpeg' : 'video/mp4';
      } else {
        throw new Error('Response has no b64_json or url field');
      }

      mpPh.style.display = 'none';
      if (mime === 'image/jpeg') {
        mpImg.src           = url;
        mpImg.style.display = 'block';
        mpStatus.textContent = 'Done (JPEG frame — worker is in JPEG mode).';
      } else {
        mpVideo.src           = url;
        mpVideo.style.display = 'block';
        mpVideo.play().catch(() => {});
        mpStatus.textContent  = 'Playing.';
      }
    }

    mpGen.addEventListener('click', async () => {
      const model  = document.getElementById('mp-model').value.trim();
      const prompt = document.getElementById('mp-prompt').value.trim();
      if (!model || !prompt) { alert('Model and Prompt are required.'); return; }

      const body    = { model, prompt, output_format: "mp4" };
      const size    = document.getElementById('mp-size').value.trim();
      const seconds = document.getElementById('mp-seconds').value;
      if (size)    body.size    = size;
      if (seconds) body.seconds = parseInt(seconds);

      mpGen.disabled           = true;
      mpSpinner.style.display  = 'block';
      mpStatus.textContent     = 'Generating…';
      mpImg.style.display      = 'none';
      mpVideo.style.display    = 'none';
      mpPh.style.display       = 'block';
      mpPh.textContent         = 'Generating…';

      try {
        const resp = await fetch('/api/video', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify(body),
        });
        const json = await resp.json();
        if (!resp.ok) throw new Error(json.error || 'Dynamo error ' + resp.status);
        await playVideoResponse(json);
        mpNext.disabled = false;
      } catch (e) {
        mpStatus.textContent = 'Error: ' + e.message;
        mpPh.textContent     = 'Failed.';
      } finally {
        mpGen.disabled          = false;
        mpSpinner.style.display = 'none';
      }
    });

    async function fetchAndPlayNext() {
      mpNext.disabled         = true;
      mpGen.disabled          = true;
      mpSpinner.style.display = 'block';
      mpStatus.textContent    = 'Fetching next clip…';
      try {
        const resp = await fetch('/api/next');
        if (resp.status === 404 || resp.status === 501) {
          mpStatus.textContent = 'No more clips.';
          mpNext.disabled = true;
          return;
        }
        const json = await resp.json();
        if (!resp.ok) throw new Error(json.error || 'Error ' + resp.status);
        await playVideoResponse(json);
        mpNext.disabled = false;
      } catch (e) {
        mpStatus.textContent = 'Error: ' + e.message;
      } finally {
        mpGen.disabled          = false;
        mpSpinner.style.display = 'none';
      }
    }

    mpNext.addEventListener('click', fetchAndPlayNext);

    mpVideo.addEventListener('ended', () => {
      if (!mpNext.disabled) fetchAndPlayNext();
    });

    // ── Debug panel polling ──────────────────────────────────────────────────
    const mjReqLog = document.getElementById('mj-req-log');
    const mjResLog = document.getElementById('mj-res-log');
    const mpReqLog = document.getElementById('mp-req-log');
    const mpResLog = document.getElementById('mp-res-log');
    let _lastReq = null, _lastChunks = null;

    setInterval(async () => {
      try {
        const d = await fetch('/api/debug').then(r => r.json());
        if (d.request !== _lastReq) {
          const txt = d.request || '—';
          mjReqLog.textContent = txt;
          mpReqLog.textContent = txt;
          _lastReq = d.request;
        }
        if (d.chunks !== _lastChunks) {
          const txt = d.chunks || '—';
          mjResLog.textContent = txt;
          mpResLog.textContent = txt;
          mjResLog.scrollTop = mjResLog.scrollHeight;
          mpResLog.scrollTop = mpResLog.scrollHeight;
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


async def handle_stream(request: web.Request) -> web.StreamResponse:
    """GET /api/stream?model=...&prompt=... → Dynamo POST /v1/videos/stream (MJPEG)."""
    q = request.rel_url.query
    model = q.get("model", "")
    prompt = q.get("prompt", "")
    if not model or not prompt:
        raise web.HTTPBadRequest(reason="model and prompt are required")

    body: dict = {"model": model, "prompt": prompt}
    if q.get("size"):
        body["size"] = q["size"]
    if q.get("seconds"):
        body["seconds"] = int(q["seconds"])
    nvext: dict = {}
    if q.get("fps"):
        nvext["fps"] = int(q["fps"])
    if nvext:
        body["nvext"] = nvext

    debug = request.app["debug"]
    debug["request"] = json.dumps(body, indent=2)
    debug["chunks"] = ""

    dynamo_url = request.app["dynamo_url"]
    timeout = ClientTimeout(total=None, connect=10)

    try:
        async with ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{dynamo_url}/v1/videos/stream", json=body
            ) as upstream:
                if upstream.status != 200:
                    text = await upstream.text()
                    raise web.HTTPBadGateway(
                        reason=f"Dynamo error {upstream.status}: {text}"
                    )

                ct = upstream.headers.get(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                m = re.search(r"boundary=(\S+)", ct)
                boundary = (m.group(1) if m else "frame").strip('"')
                bound_b = f"--{boundary}\r\n".encode()
                head_end = b"\r\n\r\n"
                next_bound_b = f"--{boundary}".encode()

                response = web.StreamResponse(headers={"Content-Type": ct})
                await response.prepare(request)

                buf = b""
                frame_idx = 0
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                    buf += chunk
                    if len(buf) > 2_097_152:
                        buf = buf[-1_048_576:]
                    while True:
                        bpos = buf.find(bound_b)
                        if bpos == -1:
                            break
                        hstart = bpos + len(bound_b)
                        hend = buf.find(head_end, hstart)
                        if hend == -1:
                            break
                        nextb = buf.find(next_bound_b, hend + len(head_end))
                        if nextb == -1:
                            break
                        headers_text = buf[hstart:hend].decode(
                            "utf-8", errors="replace"
                        )
                        data_size = nextb - (hend + len(head_end)) - 2
                        ct_match = re.search(
                            r"Content-Type:\s*([^\r\n]+)", headers_text, re.I
                        )
                        entry = {
                            "frame": frame_idx,
                            "content_type": (
                                ct_match.group(1).strip() if ct_match else "image/jpeg"
                            ),
                            "data": f"<frame_data_{frame_idx}>",
                            "size_bytes": max(0, data_size),
                        }
                        debug["chunks"] += json.dumps(entry) + "\n"
                        frame_idx += 1
                        buf = buf[nextb:]
                    debug["chunks"] += "==== end of response chunk ====\n"

                    # Add client-side delay to show streaming frames
                    # Otherwise the frames will be updated as soon as
                    # the next frame is received which is too fast to see.
                    await asyncio.sleep(0.2)
    except web.HTTPException:
        raise
    except Exception as exc:
        logger.error("Stream proxy error: %s", exc)
        raise web.HTTPBadGateway(reason=str(exc))

    return response


async def handle_video(request: web.Request) -> web.Response:
    """POST /api/video → Dynamo POST /v1/videos (non-streaming JSON)."""
    body = await request.json()

    debug = request.app["debug"]
    debug["request"] = json.dumps(body, indent=2)
    debug["chunks"] = ""
    request.app["clips"] = []
    request.app["next_clip"] = 1

    dynamo_url = request.app["dynamo_url"]
    timeout = ClientTimeout(total=300, connect=10)

    try:
        # [gluo TODO] parse SSE
        async with ClientSession(timeout=timeout) as session:
            async with session.post(f"{dynamo_url}/v1/videos", json=body) as upstream:
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
    except Exception as exc:
        logger.error("Video proxy error: %s", exc)
        raise web.HTTPBadGateway(reason=str(exc))


async def handle_next(request: web.Request) -> web.Response:
    """GET /api/next → return the next video clip (same response shape as /api/video)."""
    clips = request.app["clips"]
    if len(clips) == 0 or request.app["next_clip"] >= len(clips):
        return web.json_response({"error": "No clips available"}, status=404)
    clip = clips[request.app["next_clip"]]
    request.app["next_clip"] += 1
    return web.json_response(clip, status=200)


async def handle_debug(request: web.Request) -> web.Response:
    """GET /api/debug → current request body and response chunks strings."""
    return web.json_response(request.app["debug"])


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def make_app(dynamo_url: str) -> web.Application:
    app = web.Application()
    app["dynamo_url"] = dynamo_url.rstrip("/")
    app["debug"] = {"request": "", "chunks": ""}
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/debug", handle_debug)
    app.router.add_get("/api/next", handle_next)
    app.router.add_get("/api/stream", handle_stream)
    app.router.add_post("/api/video", handle_video)
    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Proxy web server for Dynamo video demos"
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
