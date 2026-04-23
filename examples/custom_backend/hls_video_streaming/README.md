# Synthetic HLS/CMAF Video Streaming POC

This example packages a prerecorded MP4 into real HLS CMAF assets at worker startup, then simulates live generation by gradually revealing `.m4s` fragments through the experimental Dynamo pull routes.

`CMAF` stands for `Common Media Application Format`. In this example, that means the HLS stream is built from:

- one MP4 initialization fragment, `init.mp4`
- a sequence of fragmented MP4 media segments, `0.m4s`, `1.m4s`, and so on

That is different from older HLS setups that use MPEG-TS (`.ts`) segments. We use CMAF here because it keeps the experiment focused on modern fragmented MP4 delivery, works cleanly with HLS players, and leaves room for future experiments that could reuse the same segment format for DASH as well.

## Architecture note

This POC intentionally uses worker-affine routing instead of shared storage.

- The Python worker packages one reusable CMAF asset set at startup: `init.mp4` plus numbered `.m4s` fragments.
- Each generation request creates a lightweight in-memory session record with timing and visibility state.
- The worker encodes `session_id`, `worker_id`, `namespace`, and `component` into the opaque `stream_id`.
- The Rust frontend decodes that `stream_id` on later `GET` requests and direct-routes playlist / fragment fetches back to the exact worker instance that owns the session.

This keeps the experiment focused on Dynamo's worker-to-frontend binary delivery path instead of introducing a separate storage tier.

## Requirements

- `ffmpeg` on `PATH`
- A local source MP4 file

## Worker

Start the standalone worker with a real MP4:

```bash
python examples/custom_backend/hls_video_streaming/worker.py \
  --source-mp4 /path/to/source.mp4 \
  --segment-seconds 2 \
  --emit-cadence-ms 1000 \
  --playlist-window-segments 4 \
  --model synthetic-hls-cmaf-video-stream
```

The worker registers a `ModelType.Videos` backend and serves four Dynamo endpoints on the same worker instance:

- `generate`
- `playlist`
- `init_fragment`
- `segment`

## Frontend routes

With Dynamo frontend running, the relevant routes are:

- `POST /v1/videos/stream/hls`
  Starts a synthetic HLS session and returns `stream_id`, `playlist_url`, `target_duration_seconds`, and `expires_at`.
- `GET /v1/videos/stream/hls/{stream_id}/playlist.m3u8`
  Returns the live playlist as `application/vnd.apple.mpegurl`.
- `GET /v1/videos/stream/hls/{stream_id}/init.mp4`
  Returns the CMAF init fragment as `video/mp4`.
- `GET /v1/videos/stream/hls/{stream_id}/segment/{segment_id}.m4s`
  Returns each CMAF media fragment as `video/mp4`.

## Reference client

Serve the HTML client:

```bash
python -m http.server 8080 -d examples/custom_backend/hls_video_streaming
```

Then open `http://localhost:8080/client.html`.

The page uses native HLS playback where available and falls back to `hls.js` everywhere else. For local browser testing, serve the page from the same origin as Dynamo or put a small reverse proxy in front of both ports so the browser can fetch the kickoff JSON, playlist, and fragments without cross-origin restrictions.

The `Frontend Base URL` field must be reachable from the browser, not just from the shell. If you are using a remote IDE, SSH session, dev container, or forwarded-port environment, browser `http://localhost:8000` may point at your laptop instead of the machine running Dynamo. In that case:

- forward the Dynamo frontend port to the browser
- set `Frontend Base URL` to that browser-visible frontend URL
- verify the same URL works in the browser by opening `/v1/models` before testing playback

## Browser CORS note

This POC enables `CORS` on the Dynamo videos router so the reference client can run in a normal browser even when the HTML page and the Dynamo frontend are served from different origins.

`CORS` stands for `Cross-Origin Resource Sharing`. Browsers normally prevent JavaScript loaded from one origin from reading responses from another origin. In practice, that means a page served from one forwarded VS Code URL or port cannot call `POST /v1/videos/stream/hls` on a different frontend URL unless the server explicitly opts in with CORS response headers.

We enable it here because this example is intentionally lightweight:

- the reference client is a standalone static `client.html`
- developers often serve that page from a different port or forwarded URL than the Dynamo frontend
- the HLS workflow needs multiple browser-side requests: kickoff JSON, playlist fetches, init fragment fetch, and repeated CMAF segment fetches

Without CORS, those browser requests fail even though the same URLs work with `curl`.

For this POC, the current CORS setting is intentionally broad so local testing is easy:

- it allows any origin
- it allows `GET`, `POST`, and `OPTIONS`
- it allows arbitrary request headers
- it applies to the videos router, not only the HLS routes

That is acceptable for a local experiment, but it is not the security posture we would want for a production deployment.

## Limitations

- This is a worker-memory POC, not a production HLS architecture. If the owning worker exits, restarts, or becomes unreachable, the stream is lost.
- There is no shared object store, CDN, or cache layer. Every playlist and fragment request must route back to the original worker instance.
- The implementation is intentionally synthetic. It reuses one prerecorded MP4 packaged at worker startup rather than generating fresh media per request.
- Session state is in-memory only and protected by a simple TTL. There is no recovery, replication, checkpointing, or cross-worker failover.
- The `stream_id` is opaque but unsigned. It is suitable for a local experiment, not for a hardened public API.
- The frontend pull path adds per-fragment RPC overhead. That is acceptable for this experiment, but it is not the scaling model you would want for large fanout or long-lived streams.
- This uses standard live HLS with CMAF, not Low-Latency HLS.
- The reference client is minimal by design. It is useful for validation, not for production playback UX or telemetry.
- The browser demo currently relies on permissive CORS settings to support cross-origin local testing. CORS is not authentication or authorization; it only controls what browser JavaScript is allowed to read.

## Follow-ups

- Add a second implementation that writes playlists and fragments to shared storage so we can compare worker-affine routing against a mock CDN-style design.
- Measure the operational tradeoff between this HLS pull path and the existing SSE push path: startup latency, per-fragment overhead, frontend CPU, worker memory, and failure behavior.
- Decide whether future binary media experiments should stay worker-affine, move to shared storage, or use a hybrid model where workers upload finalized fragments and the frontend only brokers kickoff.
- If this path continues, harden the stream token, add authz checks, and define clearer expiration and cleanup semantics.
- If browser reach and interoperability matter, consider a follow-on experiment with CMAF shared between both HLS and DASH manifests.
- If this stays browser-facing, narrow the CORS policy to the HLS routes only and allow only the expected frontend origin or reverse-proxy origin instead of `*`.
- Prefer a same-origin setup for future demos, either by serving the client from the Dynamo frontend or by putting both the client and API behind a single reverse proxy.
