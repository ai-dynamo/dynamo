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
  POST /api/video   → POST /v1/videos with stream:true (SSE) and
                      response_format:"b64_json". The demo decodes each SSE
                      event's inline base64 once and returns raw MP4 bytes
                      to the browser with Content-Type: video/mp4. Subsequent
                      chunks are buffered server-side as bytes.
  GET  /api/next    → return the next buffered MP4 chunk as raw bytes, or
                      404 when exhausted.
  GET  /api/debug   → current request body / response chunks (still JSON).

Demo→browser is always raw video/mp4: the browser does `await resp.blob()`
and points the <video> element at an Object URL, with no atob() loop. The
worker→demo hop stays on b64_json so this demo has no filesystem-coupling
requirement with the worker (works with any --media-output-fs-url, including
none — only the inline payloads are read).

Usage:
  python krea_realtime_demo.py [--dynamo URL] [--host HOST] [--port PORT]

Requires: aiohttp (pip install aiohttp).
"""

import argparse
import asyncio
import base64
import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

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
      // response_format=b64_json keeps the worker→demo hop self-contained: each
      // SSE event carries the MP4 bytes inline (base64-encoded). The demo
      // decodes once into raw bytes and serves video/mp4 to the browser, so
      // no filesystem coordination between worker and demo is required.
      const body = {
        model,
        prompt,
        output_format: "mp4",
        stream: true,
        response_format: "b64_json",
      };
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

    // Receive raw video/mp4 bytes from the server, swap them into the <video>
    // element via an Object URL. The previous Object URL (if any) is revoked
    // to avoid leaking blob storage on long sessions.
    async function playChunkBytes(resp, videoEl, prevUrlRef) {
      const blob = await resp.blob();
      if (prevUrlRef.value) URL.revokeObjectURL(prevUrlRef.value);
      const url = URL.createObjectURL(blob);
      prevUrlRef.value = url;
      videoEl.src = url;
      videoEl.style.display = 'block';
      videoEl.play().catch(() => {});
    }

    const msPrevRef = { value: null };

    // Prefetched response for the next chunk. Populated by startPrefetch() the
    // moment a chunk begins playback, consumed by playNextChunk() when the
    // current chunk fires `ended`. The server-side long-poll on /api/next
    // overlaps with playback of the previous chunk, so generation latency is
    // hidden as long as it's less than the chunk's playback duration. Without
    // prefetch, the <video> sits on the last frame of chunk N while
    // /api/next blocks waiting for chunk N+1 to be generated.
    let msPrefetch = null;
    let msStreamDone = false;

    function startPrefetch() {
      if (msStreamDone || msPrefetch) return;
      msPrefetch = fetch('/api/next').catch((e) => ({ _error: e }));
    }

    async function playNextChunk() {
      if (msStreamDone) return;
      // `ended` shouldn't fire before `playing`, but be defensive in case the
      // browser fires them out of order on src changes.
      if (!msPrefetch) startPrefetch();
      const resp = await msPrefetch;
      msPrefetch = null;

      if (resp && resp._error) {
        msStatus.textContent = 'Error: ' + resp._error.message;
        return;
      }
      if (resp.status === 404) {
        msStreamDone = true;
        msNext.disabled = true;
        msStatus.textContent = 'Stream complete.';
        return;
      }
      if (resp.status === 504) {
        // Long-poll timed out without a chunk; let the user retry manually.
        msStatus.textContent = 'Timed out waiting for chunk. Click Next to retry.';
        msNext.disabled = false;
        return;
      }
      if (!resp.ok) {
        let msg;
        try { msg = (await resp.json()).error || ('Error ' + resp.status); }
        catch (_) { msg = 'Error ' + resp.status; }
        msStatus.textContent = 'Error: ' + msg;
        return;
      }
      const chunkIdx = resp.headers.get('X-Chunk-Index') || '?';
      await playChunkBytes(resp, msVideo, msPrevRef);
      msStatus.textContent = `Playing chunk ${parseInt(chunkIdx) + 1}.`;
      msNext.disabled = false;
    }

    msGen.addEventListener('click', async () => {
      const body = readMsForm();
      if (!body) return;
      msGen.disabled = true;
      msSpinner.style.display = 'block';
      msStatus.textContent = 'Generating…';
      msPh.style.display = 'none';
      msVideo.style.display = 'none';
      msPrefetch = null;
      msStreamDone = false;
      try {
        const resp = await fetch('/api/video', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!resp.ok) {
          // Errors come back as JSON; success comes back as video/mp4 bytes.
          let msg;
          try { msg = (await resp.json()).error || ('Dynamo error ' + resp.status); }
          catch (_) { msg = 'Dynamo error ' + resp.status; }
          throw new Error(msg);
        }
        await playChunkBytes(resp, msVideo, msPrevRef);
        msStatus.textContent = 'Playing chunk 1.';
        msNext.disabled = false;
      } catch (e) {
        msStatus.textContent = 'Error: ' + e.message;
      } finally {
        msGen.disabled = false;
        msSpinner.style.display = 'none';
      }
    });

    // Manual Next Chunk button: same as auto-advance but useful for stepping
    // through or retrying after a 504.
    msNext.addEventListener('click', () => {
      if (!msPrefetch) startPrefetch();
      playNextChunk();
    });

    // Auto-advance via prefetch: kick off the next /api/next fetch the moment
    // a chunk *starts* playing, so the long-poll overlaps with playback. By
    // the time `ended` fires we usually have the response already.
    msVideo.addEventListener('playing', () => {
      if (!msNext.disabled && !msStreamDone) startPrefetch();
    });
    msVideo.addEventListener('ended', () => {
      if (!msNext.disabled && !msStreamDone) playNextChunk();
    });

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


def _bytes_from_item(item: dict) -> bytes:
    """Extract raw video bytes from one element of an `NvVideosResponse.data` list.

    Supports both `response_format` values the realtime worker emits:
      - b64_json: base64-decode the embedded payload.
      - url: file:// only (the worker's default --media-output-fs-url). Reads
             the file from disk. http(s):// URLs are fetched via urllib.
    """
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])
    url = item.get("url")
    if not url:
        raise ValueError("response item has neither b64_json nor url")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        return Path(parsed.path).read_bytes()
    if parsed.scheme in ("http", "https"):
        with urllib.request.urlopen(url) as r:  # noqa: S310
            return r.read()
    raise ValueError(f"unsupported url scheme: {parsed.scheme!r}")


def _mp4_response(data: bytes, chunk_idx: int, total_so_far: int) -> web.Response:
    """Return raw MP4 bytes with metadata headers so the client can decide
    whether to fetch /api/next or stop. `total_so_far` is the current count of
    buffered chunks — it's monotonically non-decreasing while SSE is streaming.
    """
    return web.Response(
        body=data,
        content_type="video/mp4",
        headers={
            "X-Chunk-Index": str(chunk_idx),
            "X-Total-Chunks-So-Far": str(total_so_far),
        },
    )


async def handle_video(request: web.Request) -> web.Response:
    """POST /api/video → Dynamo POST /v1/videos.

    Pass-through proxy. Body is sent verbatim — prompt/size/seconds/nvext fields
    propagate as-is to the realtime worker via Dynamo's CreateVideoRequest.

    When body["stream"] is true the upstream is SSE: we consume the event
    stream, decode every chunk to raw MP4 bytes into request.app["clips"]
    (a list[bytes]), and return the first chunk's bytes immediately with
    Content-Type: video/mp4. Subsequent chunks are fetched via /api/next.

    Two coordination signals shared with /api/next:
      - request.app["clip_available"]: asyncio.Event — set on every new chunk
        (cleared after handle_next reads). Lets /api/next long-poll for the
        next chunk instead of returning 404 prematurely while the worker is
        still generating later chunks.
      - request.app["sse_done"]: asyncio.Event — set when the SSE consumer
        finishes (successfully or with error). Combined with clip_available,
        /api/next can distinguish "not ready yet, please wait" from "no more
        chunks are coming."
    """
    body = await request.json()
    streaming = bool(body.get("stream"))

    debug = request.app["debug"]
    debug["request"] = json.dumps(body, indent=2)
    debug["chunks"] = ""
    request.app["clips"] = []
    request.app["next_clip"] = 1
    request.app["clip_available"] = asyncio.Event()
    request.app["sse_done"] = asyncio.Event()

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
                                    except Exception:
                                        continue
                                    items = (
                                        event.get("data")
                                        if isinstance(event.get("data"), list)
                                        else [event]
                                    )
                                    new_indices: list[int] = []
                                    for item in items:
                                        if not isinstance(item, dict):
                                            continue
                                        try:
                                            chunk_bytes = _bytes_from_item(item)
                                        except Exception as e:
                                            logger.warning("Skipping chunk: %s", e)
                                            continue
                                        request.app["clips"].append(chunk_bytes)
                                        new_indices.append(
                                            len(request.app["clips"]) - 1
                                        )
                                    # Wake any /api/next waiter blocked on the
                                    # next chunk's arrival.
                                    if new_indices:
                                        request.app["clip_available"].set()
                                        await asyncio.sleep(0)
                                    # Log a sanitized view of the event with
                                    # placeholders instead of the actual bytes.
                                    sanitized = json.loads(data_str)
                                    if isinstance(sanitized.get("data"), list):
                                        for i, item in enumerate(sanitized["data"]):
                                            if "b64_json" in item:
                                                item[
                                                    "b64_json"
                                                ] = f"<video_data_{new_indices[i] if i < len(new_indices) else '?'}>"
                                            if "url" in item:
                                                item[
                                                    "url"
                                                ] = f"<video_url_{new_indices[i] if i < len(new_indices) else '?'}>"
                                    debug["chunks"] += json.dumps(sanitized, indent=2)
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
                    # Stream is closed (success or error). Wake any /api/next
                    # waiter so it can see the final clip count and return
                    # 404 if it's already drained.
                    request.app["sse_done"].set()
                    request.app["clip_available"].set()

            asyncio.create_task(consume_sse())
            await first_clip_ready.wait()

            if not request.app["clips"]:
                raise web.HTTPBadGateway(reason="SSE stream ended without a video clip")
            return _mp4_response(
                request.app["clips"][0],
                chunk_idx=0,
                total_so_far=len(request.app["clips"]),
            )

        async with ClientSession(timeout=timeout) as session:
            async with session.post(f"{dynamo_url}/v1/videos", json=body) as upstream:
                if upstream.status != 200:
                    text = await upstream.text()
                    raise web.HTTPBadGateway(
                        reason=f"Dynamo error {upstream.status}: {text}"
                    )
                json_body = await upstream.json()
                items = (
                    json_body.get("data")
                    if isinstance(json_body.get("data"), list)
                    else []
                )
                for item in items:
                    if isinstance(item, dict):
                        try:
                            request.app["clips"].append(_bytes_from_item(item))
                        except Exception as e:
                            logger.warning("Skipping non-streaming chunk: %s", e)
                sanitized = {
                    **{k: v for k, v in json_body.items() if k != "data"},
                    "data": [
                        {
                            **{
                                k: v
                                for k, v in item.items()
                                if k not in ("b64_json", "url")
                            },
                            **(
                                {"b64_json": f"<video_data_{i}>"}
                                if isinstance(item, dict) and "b64_json" in item
                                else {}
                            ),
                            **(
                                {"url": f"<video_url_{i}>"}
                                if isinstance(item, dict) and "url" in item
                                else {}
                            ),
                        }
                        for i, item in enumerate(items)
                    ],
                }
                debug["chunks"] = json.dumps(sanitized, indent=2)
                if not request.app["clips"]:
                    raise web.HTTPBadGateway(reason="upstream returned no video data")
                return _mp4_response(
                    request.app["clips"][0],
                    chunk_idx=0,
                    total_so_far=len(request.app["clips"]),
                )
    except web.HTTPException:
        raise
    except Exception as exc:
        logger.error("Video proxy error: %s", exc)
        raise web.HTTPBadGateway(reason=str(exc))


_NEXT_LONG_POLL_TIMEOUT_S = 10.0


async def handle_next(request: web.Request) -> web.Response:
    """GET /api/next → return the next buffered MP4 chunk as raw bytes.

    Long-polls if the next chunk hasn't arrived yet. The browser fires the
    `ended` event the instant a chunk finishes playing (~1 s of video at
    12 fps), but the worker often hasn't produced the next one yet (~2 s at
    4 inference steps on H100-class). Returning 404 immediately would cause
    the client to permanently stop the stream after the first chunk.

    Returns:
      200 + video/mp4: when the next chunk becomes available.
      404 + JSON: when the SSE stream has finished AND we've already served
                  every buffered chunk.
      504 + JSON: if no chunk arrived within `_NEXT_LONG_POLL_TIMEOUT_S`
                  (defensive — bounded so the client can retry rather than
                  hang on a stalled worker).
    """
    clip_available: asyncio.Event = request.app["clip_available"]
    sse_done: asyncio.Event = request.app["sse_done"]

    deadline = asyncio.get_event_loop().time() + _NEXT_LONG_POLL_TIMEOUT_S
    while True:
        clips = request.app["clips"]
        idx = request.app["next_clip"]
        if idx < len(clips):
            chunk_bytes = clips[idx]
            request.app["next_clip"] = idx + 1
            # If we just drained the buffer and more chunks may still arrive,
            # clear the event so the next /api/next call blocks again.
            if (
                request.app["next_clip"] >= len(request.app["clips"])
                and not sse_done.is_set()
            ):
                clip_available.clear()
            return _mp4_response(chunk_bytes, chunk_idx=idx, total_so_far=len(clips))

        if sse_done.is_set():
            return web.json_response(
                {"error": "stream complete; no more chunks"}, status=404
            )

        while True:
            if deadline - asyncio.get_event_loop().time() <= 0:
                return web.json_response({"error": "timeout"}, status=504)
            await asyncio.sleep(0.1)
            if clip_available.is_set():
                break


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
