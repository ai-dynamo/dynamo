#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Proxy web server for Dynamo video streaming.

Serves the HTML UI and proxies requests to the Dynamo HTTP server.
Because the browser loads the page from this server (same origin), no CORS
preflight is triggered when the page calls /v1/videos/stream.

Two UI tabs are provided:
  - Live Stream: uses <img src="/stream?..."> for native MJPEG display.
  - MP4 Download: calls /mp4 which streams multiple MP4 clips as SSE; the
    browser plays them in sequence, auto-advancing as each clip arrives.

The /stream endpoint is a GET wrapper around the Dynamo POST endpoint so the
browser can use a plain <img src="/stream?..."> for native MJPEG display.

The /mp4 endpoint generates `count` video clips sequentially, encoding each
via ffmpeg and emitting it as a base64 SSE `data:` event. The browser decodes
each event to a Blob URL and queues it for playback.

Usage:
  python server.py [--dynamo URL] [--host HOST] [--port PORT]

Options:
  --dynamo   Dynamo HTTP server base URL (default: http://localhost:8000)
  --host     Bind address (default: 0.0.0.0)
  --port     Listen port (default: 8888)
"""

import argparse
import asyncio
import base64
import logging

from aiohttp import ClientSession, ClientTimeout, web

logger = logging.getLogger(__name__)

# ── Embedded HTML ─────────────────────────────────────────────────────────────

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Dynamo Video</title>
  <style>
    body { font-family: sans-serif; max-width: 900px; margin: 40px auto; padding: 0 16px; background: #111; color: #eee; }
    h1 { font-size: 1.2rem; margin-bottom: 1rem; }
    .tabs { display: flex; gap: 2px; margin-bottom: 16px; border-bottom: 1px solid #333; }
    .tab-btn { background: #222; color: #888; border: 1px solid #333; border-bottom: none; border-radius: 4px 4px 0 0; padding: 8px 18px; cursor: pointer; font-size: 0.9rem; margin-bottom: -1px; }
    .tab-btn.active { background: #111; color: #eee; border-color: #555; border-bottom-color: #111; }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }
    .controls label { display: flex; flex-direction: column; gap: 4px; font-size: 0.85rem; color: #aaa; }
    .controls input, .controls textarea { background: #222; color: #eee; border: 1px solid #444; border-radius: 4px; padding: 6px 8px; font-size: 0.9rem; }
    .controls textarea { resize: vertical; min-height: 60px; }
    .row { display: flex; gap: 10px; margin-bottom: 14px; align-items: center; }
    button { padding: 8px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem; }
    .btn-primary { background: #0a84ff; color: #fff; }
    .btn-stop { background: #555; color: #eee; }
    button:disabled { opacity: 0.4; cursor: default; }
    .status { font-size: 0.8rem; color: #888; }
    .display-box { width: 100%; border: 1px solid #333; border-radius: 6px; background: #000; min-height: 300px; display: flex; flex-direction: column; align-items: center; justify-content: center; overflow: hidden; }
    .display-box img, .display-box video { max-width: 100%; max-height: 70vh; }
    .placeholder { color: #444; font-size: 0.9rem; }
    .download-link { display: none; color: #0a84ff; text-decoration: none; margin-top: 10px; font-size: 0.9rem; }
    .spinner { width: 28px; height: 28px; border: 3px solid #333; border-top-color: #0a84ff; border-radius: 50%; animation: spin 0.8s linear infinite; display: none; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <h1>Dynamo Video</h1>

  <div class="tabs">
    <button class="tab-btn active" data-tab="stream">&#9654; Live Stream</button>
    <button class="tab-btn" data-tab="mp4">&#9654; MP4 Playback</button>
  </div>

  <!-- ── Stream Tab ── -->
  <div id="tab-stream" class="tab-panel active">
    <div class="controls">
      <label>Model<input id="s-model" type="text" value="video-stream-model" /></label>
      <label>Size (optional)<input id="s-size" type="text" placeholder="e.g. 832x480" /></label>
      <label style="grid-column: span 2;">Prompt<textarea id="s-prompt">a cat playing on a sunny beach</textarea></label>
      <label>Seconds (optional)<input id="s-seconds" type="number" min="1" placeholder="e.g. 5" /></label>
      <label>FPS (optional)<input id="s-fps" type="number" min="1" placeholder="e.g. 25" /></label>
    </div>
    <div class="row">
      <button id="s-start" class="btn-primary">&#9654; Start</button>
      <button id="s-stop" class="btn-stop" disabled>&#9632; Stop</button>
      <span id="s-status" class="status">Idle</span>
    </div>
    <div class="display-box">
      <span id="s-placeholder" class="placeholder">Stream will appear here</span>
      <!-- The browser handles MJPEG natively via a same-origin GET request. -->
      <img id="s-frame" alt="video frame" style="display:none" />
    </div>
  </div>

  <!-- ── MP4 Tab ── -->
  <div id="tab-mp4" class="tab-panel">
    <div class="controls">
      <label>Model<input id="m-model" type="text" value="video-stream-model" /></label>
      <label>Size (optional)<input id="m-size" type="text" placeholder="e.g. 832x480" /></label>
      <label style="grid-column: span 2;">Prompt<textarea id="m-prompt">a cat playing on a sunny beach</textarea></label>
      <label>Seconds (optional)<input id="m-seconds" type="number" min="1" placeholder="e.g. 5" /></label>
      <label>FPS (optional)<input id="m-fps" type="number" min="1" placeholder="e.g. 25" /></label>
      <label>Count<input id="m-count" type="number" min="1" value="3" /></label>
    </div>
    <div class="row">
      <button id="m-gen" class="btn-primary">&#9654; Generate</button>
      <button id="m-cancel" class="btn-stop" style="display:none">&#x2715; Cancel</button>
      <div id="m-spinner" class="spinner"></div>
      <span id="m-status" class="status">Idle</span>
    </div>
    <div class="display-box">
      <span id="m-placeholder" class="placeholder">Videos will play here as they are generated</span>
      <video id="m-video" controls style="display:none"></video>
      <a id="m-download" class="download-link" download="video.mp4">&#8681; Download current clip</a>
    </div>
  </div>

  <script>
    // ── Tab switching ──
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
      });
    });

    // ── Stream Tab ──
    const sStart       = document.getElementById('s-start');
    const sStop        = document.getElementById('s-stop');
    const sStatus      = document.getElementById('s-status');
    const sFrame       = document.getElementById('s-frame');
    const sPlaceholder = document.getElementById('s-placeholder');

    sStart.addEventListener('click', () => {
      const model   = document.getElementById('s-model').value.trim();
      const prompt  = document.getElementById('s-prompt').value.trim();
      const size    = document.getElementById('s-size').value.trim();
      const seconds = document.getElementById('s-seconds').value;
      const fps     = document.getElementById('s-fps').value;
      if (!model || !prompt) { alert('Model and Prompt are required.'); return; }
      const params = new URLSearchParams({ model, prompt });
      if (size)    params.set('size', size);
      if (seconds) params.set('seconds', seconds);
      if (fps)     params.set('fps', fps);
      // Same-origin GET — no CORS preflight, browser decodes MJPEG natively.
      sFrame.src = '/stream?' + params.toString();
      sFrame.style.display = 'block';
      sPlaceholder.style.display = 'none';
      sStart.disabled = true;
      sStop.disabled  = false;
      sStatus.textContent = 'Streaming…';
    });

    sStop.addEventListener('click', () => {
      sFrame.src = '';
      sFrame.style.display = 'none';
      sPlaceholder.style.display = 'block';
      sStart.disabled = false;
      sStop.disabled  = true;
      sStatus.textContent = 'Stopped.';
    });

    sFrame.addEventListener('error', () => {
      sStatus.textContent = 'Stream error or ended.';
      sStart.disabled = false;
      sStop.disabled  = true;
    });

    // ── MP4 Tab ──
    const mGen         = document.getElementById('m-gen');
    const mCancel      = document.getElementById('m-cancel');
    const mSpinner     = document.getElementById('m-spinner');
    const mStatus      = document.getElementById('m-status');
    const mVideo       = document.getElementById('m-video');
    const mDownload    = document.getElementById('m-download');
    const mPlaceholder = document.getElementById('m-placeholder');
    let mAbort    = null;   // AbortController for the fetch
    let mQueue    = [];     // blob URLs waiting to be played
    let mPlaying  = false;  // whether a clip is currently in the <video> element
    let mReceived = 0;
    let mTotal    = 1;

    function b64ToBlob(b64) {
      const bin = atob(b64);
      const buf = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
      return new Blob([buf], { type: 'video/mp4' });
    }

    function mPlayNext() {
      if (!mQueue.length) { mPlaying = false; return; }
      mPlaying = true;
      const url = mQueue.shift();
      mVideo.src             = url;
      mDownload.href         = url;
      mVideo.style.display   = 'block';
      mDownload.style.display = 'inline';
      mPlaceholder.style.display = 'none';
      mVideo.play().catch(() => {});
    }

    mVideo.addEventListener('ended', () => {
      if (mQueue.length) {
        mPlayNext();
      } else if (mAbort) {
        // still receiving — wait; next message handler will call mPlayNext
        mPlaying = false;
        mStatus.textContent = 'Loading next clip…';
      } else {
        mPlaying = false;
        mStatus.textContent = 'Done.';
      }
    });

    mGen.addEventListener('click', async () => {
      const model   = document.getElementById('m-model').value.trim();
      const prompt  = document.getElementById('m-prompt').value.trim();
      const size    = document.getElementById('m-size').value.trim();
      const seconds = document.getElementById('m-seconds').value;
      const fps     = document.getElementById('m-fps').value;
      const count   = parseInt(document.getElementById('m-count').value) || 3;
      if (!model || !prompt) { alert('Model and Prompt are required.'); return; }

      const params = new URLSearchParams({ model, prompt, count });
      if (size)    params.set('size', size);
      if (seconds) params.set('seconds', seconds);
      if (fps)     params.set('fps', fps);

      mQueue    = [];
      mPlaying  = false;
      mReceived = 0;
      mTotal    = count;

      mGen.disabled              = true;
      mCancel.style.display      = 'inline';
      mSpinner.style.display     = 'block';
      mStatus.textContent        = `Generating clip 1/${count}…`;
      mVideo.style.display       = 'none';
      mDownload.style.display    = 'none';
      mPlaceholder.style.display = 'block';
      mPlaceholder.textContent   = 'Generating…';

      mAbort = new AbortController();
      try {
        const resp = await fetch('/mp4?' + params.toString(), { signal: mAbort.signal });
        if (!resp.ok) throw new Error(await resp.text());

        const reader  = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });

          // Process all complete SSE events (delimited by \n\n).
          let idx;
          while ((idx = buf.indexOf('\n\n')) !== -1) {
            const block = buf.slice(0, idx);
            buf = buf.slice(idx + 2);

            let evtType = 'message', evtData = '';
            for (const line of block.split('\n')) {
              if (line.startsWith('event: ')) evtType = line.slice(7).trim();
              else if (line.startsWith('data: ')) evtData += line.slice(6);
            }

            if (evtType === 'done') { reader.cancel(); return; }
            if (evtType === 'error') throw new Error(evtData);

            // Default message: a base64-encoded MP4 clip.
            mReceived++;
            const url = URL.createObjectURL(b64ToBlob(evtData));
            mQueue.push(url);
            if (mReceived < mTotal) {
              mStatus.textContent = `Generating clip ${mReceived + 1}/${mTotal}…`;
            } else {
              mSpinner.style.display = 'none';
              mStatus.textContent = 'Playing…';
            }
            if (!mPlaying) mPlayNext();
          }
        }
      } catch (e) {
        if (e.name !== 'AbortError') mStatus.textContent = 'Error: ' + e.message;
        else mStatus.textContent = 'Cancelled.';
      } finally {
        mAbort             = null;
        mGen.disabled      = false;
        mCancel.style.display  = 'none';
        mSpinner.style.display = 'none';
      }
    });

    mCancel.addEventListener('click', () => { if (mAbort) mAbort.abort(); });
  </script>
</body>
</html>
"""

# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_jpeg_frames(data: bytes) -> list[bytes]:
    """Extract individual JPEG frames from a multipart/x-mixed-replace stream."""
    frames = []
    for part in data.split(b"--frame"):
        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        body = part[header_end + 4 :].rstrip(b"\r\n")
        if body.startswith(b"\xff\xd8"):  # JPEG SOI marker
            frames.append(body)
    return frames


async def _encode_mp4(frames: list[bytes], fps: int) -> bytes:
    """Pipe JPEG frames through ffmpeg and return MP4 bytes."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-f",
        "image2pipe",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "mp4",
        "-movflags",
        "frag_keyframe+empty_moov",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=b"".join(frames))
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode(errors="replace"))
    return stdout


# ── Handlers ──────────────────────────────────────────────────────────────────


async def handle_index(request: web.Request) -> web.Response:
    return web.Response(text=_HTML, content_type="text/html")


async def handle_stream(request: web.Request) -> web.StreamResponse:
    """GET /stream — converts query params to a POST body and proxies to Dynamo.

    The browser can point an <img src="/stream?model=...&prompt=..."> here and
    receive a native MJPEG multipart/x-mixed-replace stream without any CORS
    preflight, because the request is same-origin.
    """
    q = request.rel_url.query
    model = q.get("model", "video-stream-model")
    prompt = q.get("prompt", "")

    body: dict = {"model": model, "prompt": prompt}
    if q.get("size"):
        body["size"] = q["size"]
    if q.get("seconds"):
        body["seconds"] = int(q["seconds"])

    nvext: dict = {}
    if q.get("fps"):
        nvext["fps"] = int(q["fps"])
    if q.get("num_frames"):
        nvext["num_frames"] = int(q["num_frames"])
    if nvext:
        body["nvext"] = nvext

    dynamo_url = request.app["dynamo_url"]
    timeout = ClientTimeout(total=None, connect=10)

    try:
        async with ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{dynamo_url}/v1/videos/stream",
                json=body,
            ) as upstream:
                if upstream.status != 200:
                    text = await upstream.text()
                    raise web.HTTPBadGateway(
                        reason=f"Dynamo error {upstream.status}: {text}"
                    )

                response = web.StreamResponse(
                    headers={
                        "Content-Type": "multipart/x-mixed-replace; boundary=frame"
                    }
                )
                await response.prepare(request)

                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)

    except web.HTTPException:
        raise
    except Exception as exc:
        logger.error("Stream proxy error: %s", exc)
        raise web.HTTPBadGateway(reason=str(exc))

    return response


async def handle_mp4(request: web.Request) -> web.StreamResponse:
    """GET /mp4 — generate `count` MP4 clips sequentially, emitting each as SSE.

    Each clip is a complete invocation of Dynamo's /v1/videos/stream. Clips are
    base64-encoded and sent as SSE `data:` events so the browser can decode them
    to Blob URLs and play them in sequence as they arrive.
    """
    q = request.rel_url.query
    model = q.get("model", "video-stream-model")
    prompt = q.get("prompt", "")
    fps = int(q.get("fps", "25"))
    count = max(1, int(q.get("count", "1")))

    body: dict = {"model": model, "prompt": prompt}
    if q.get("size"):
        body["size"] = q["size"]
    if q.get("seconds"):
        body["seconds"] = int(q["seconds"])

    nvext: dict = {}
    if q.get("fps"):
        nvext["fps"] = fps
    if q.get("num_frames"):
        nvext["num_frames"] = int(q["num_frames"])
    if nvext:
        body["nvext"] = nvext

    dynamo_url = request.app["dynamo_url"]
    timeout = ClientTimeout(total=None, connect=10)

    sse = web.StreamResponse(
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
    await sse.prepare(request)

    for i in range(count):
        try:
            async with ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{dynamo_url}/v1/videos/stream",
                    json=body,
                ) as upstream:
                    if upstream.status != 200:
                        text = await upstream.text()
                        await sse.write(
                            f"event: error\ndata: Dynamo error {upstream.status}: {text}\n\n".encode()
                        )
                        break
                    raw = await upstream.read()

            frames = _extract_jpeg_frames(raw)
            if not frames:
                await sse.write(b"event: error\ndata: No frames received\n\n")
                break

            mp4_bytes = await _encode_mp4(frames, fps)
            b64 = base64.b64encode(mp4_bytes).decode()
            await sse.write(f"data: {b64}\n\n".encode())

        except Exception as exc:
            logger.error("MP4 stream error at clip %d: %s", i, exc)
            await sse.write(f"event: error\ndata: {exc}\n\n".encode())
            break

    await sse.write(b"event: done\ndata: \n\n")
    return sse


async def handle_proxy(request: web.Request) -> web.Response:
    """Generic reverse proxy for all other /v1/* requests (e.g. GET /v1/models)."""
    dynamo_url = request.app["dynamo_url"]
    url = f"{dynamo_url}{request.path_qs}"
    timeout = ClientTimeout(total=30)

    try:
        async with ClientSession(timeout=timeout) as session:
            async with session.request(
                request.method,
                url,
                headers={
                    k: v for k, v in request.headers.items() if k.lower() != "host"
                },
                data=await request.read(),
            ) as upstream:
                body = await upstream.read()
                return web.Response(
                    status=upstream.status,
                    body=body,
                    content_type=upstream.content_type,
                )
    except Exception as exc:
        logger.error("Proxy error for %s: %s", url, exc)
        raise web.HTTPBadGateway(reason=str(exc))


# ── App factory ───────────────────────────────────────────────────────────────


def make_app(dynamo_url: str) -> web.Application:
    app = web.Application()
    app["dynamo_url"] = dynamo_url.rstrip("/")
    app.router.add_get("/", handle_index)
    app.router.add_get("/stream", handle_stream)
    app.router.add_get("/mp4", handle_mp4)
    app.router.add_route("*", "/v1/{path_info:.*}", handle_proxy)
    return app


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Proxy web server for Dynamo video streaming"
    )
    parser.add_argument(
        "--dynamo", default="http://localhost:8000", help="Dynamo HTTP server URL"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8888, help="Listen port (default: 8888)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Proxying to Dynamo at %s", args.dynamo)
    logger.info(
        "Open http://%s:%d in your browser",
        args.host if args.host != "0.0.0.0" else "localhost",
        args.port,
    )
    web.run_app(make_app(args.dynamo), host=args.host, port=args.port)
