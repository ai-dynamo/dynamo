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
  </div>

  <!-- ── MP4 Playback ── -->
  <div id="tab-mp4" class="tab">
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
      <div    id="mp-spinner" class="spinner"></div>
      <span   id="mp-status" class="status">Idle</span>
    </div>
    <div class="display" id="mp-display">
      <span class="placeholder" id="mp-placeholder">Video will appear here</span>
      <img   id="mp-img"   alt="" style="display:none" />
      <video id="mp-video" controls style="display:none"></video>
    </div>
  </div>

  <script>
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
    const mpSpinner = document.getElementById('mp-spinner');
    const mpStatus  = document.getElementById('mp-status');
    const mpImg     = document.getElementById('mp-img');
    const mpVideo   = document.getElementById('mp-video');
    const mpPh      = document.getElementById('mp-placeholder');
    let   mpPrevUrl = null;

    mpGen.addEventListener('click', async () => {
      const model  = document.getElementById('mp-model').value.trim();
      const prompt = document.getElementById('mp-prompt').value.trim();
      if (!model || !prompt) { alert('Model and Prompt are required.'); return; }

      const body    = { model, prompt };
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

        const item = json.data && json.data[0];
        if (!item) throw new Error('Response contains no data');

        let url, mime;
        if (item.b64_json) {
          const bin   = atob(item.b64_json);
          const bytes = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
          mime = detectMime(bytes);
          if (mpPrevUrl) URL.revokeObjectURL(mpPrevUrl);
          url = URL.createObjectURL(new Blob([bytes], { type: mime }));
          mpPrevUrl = url;
        } else if (item.url) {
          url  = item.url;
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
      } catch (e) {
        mpStatus.textContent = 'Error: ' + e.message;
        mpPh.textContent     = 'Failed.';
      } finally {
        mpGen.disabled          = false;
        mpSpinner.style.display = 'none';
      }
    });
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
    """GET /api/stream?model=...&prompt=... → Dynamo POST /v1/videos/stream (MJPEG).

    Maps query parameters to the Dynamo JSON body so the browser can use a
    plain img.src = '/api/stream?...' for native MJPEG display.
    """
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

                response = web.StreamResponse(
                    headers={
                        "Content-Type": upstream.headers.get(
                            "Content-Type",
                            "multipart/x-mixed-replace; boundary=frame",
                        )
                    }
                )
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                    await asyncio.sleep(0.1)  # Verify

    except web.HTTPException:
        raise
    except Exception as exc:
        logger.error("Stream proxy error: %s", exc)
        raise web.HTTPBadGateway(reason=str(exc))

    return response


async def handle_video(request: web.Request) -> web.Response:
    """POST /api/video → Dynamo POST /v1/videos (non-streaming JSON)."""
    body = await request.json()
    dynamo_url = request.app["dynamo_url"]
    timeout = ClientTimeout(total=300, connect=10)

    try:
        async with ClientSession(timeout=timeout) as session:
            async with session.post(f"{dynamo_url}/v1/videos", json=body) as upstream:
                json_body = await upstream.json()
                return web.json_response(json_body, status=upstream.status)
    except Exception as exc:
        logger.error("Video proxy error: %s", exc)
        raise web.HTTPBadGateway(reason=str(exc))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def make_app(dynamo_url: str) -> web.Application:
    app = web.Application()
    app["dynamo_url"] = dynamo_url.rstrip("/")
    app.router.add_get("/", handle_index)
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
