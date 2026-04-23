# CMAF Binary Video Streaming

This example is a minimal Proof of Concept for continuous video playback over a
single Dynamo `POST` response. It keeps Dynamo aligned with its natural
push-streaming model while improving the browser experience beyond the existing
clip-by-clip video streaming shape.

## What This POC Evaluates

The worker packages a prerecorded MP4 into real CMAF-style fragmented MP4:

- one initialization fragment, `init.mp4`
- a sequence of media fragments, `.m4s`

`CMAF` stands for `Common Media Application Format`. In practice, that means the
video is broken into streamable MP4 fragments that a browser can append into one
continuous playback timeline.

Unlike HLS, this example does not introduce playlist polling or per-segment
`GET` requests. The browser opens a single `POST /v1/videos/stream/binary/cmaf`
request, and the Dynamo frontend returns a framed binary stream containing:

1. metadata JSON
2. the CMAF init fragment
3. each media fragment
4. a final done frame

## Why This Branch Exists

This branch reimplements the CMAF binary POC from scratch on top of `main` so
we can see the smallest Dynamo-side change set clearly.

Minimal Dynamo changes in this branch:

- one new frontend route in `lib/llm/src/http/service/openai.rs`
- no new shared Rust protocol structs
- no pull-based HLS routes
- no worker-affinity routing
- no frontend session storage
- no added CORS behavior

The worker uses the existing video response contract and tags each payload with:

- `cmaf:metadata`
- `cmaf:init`
- `cmaf:segment:<n>`

The frontend decodes those tagged `b64_json` payloads and rewrites them into a
compact binary response body for the browser client.

## Requirements

- `ffmpeg`
- `ffprobe`
- a local MP4 file to use as prerecorded source material

## Run The Worker

```bash
python examples/custom_backend/cmaf_binary_video_streaming/worker.py \
  --source-mp4 /path/to/source.mp4 \
  --segment-seconds 2 \
  --emit-cadence-ms 750 \
  --model synthetic-binary-cmaf-video-stream
```

The worker registers the model on the standard Dynamo backend endpoint:

- namespace: `${DYN_NAMESPACE:-dynamo}`
- component: `backend`
- endpoint: `generate`

Discovery backend defaults:

- local runs: `file`
- in-cluster runs: `kubernetes`

That matches the existing `examples/diffusers/worker.py` behavior and avoids a
common local failure mode where the frontend and worker use different discovery
backends, so the frontend never enables the video routes for the registered
model.

## Frontend Route

This branch adds:

- `POST /v1/videos/stream/binary/cmaf`

That route forces `response_format="b64_json"` and adds the request annotation:

- `experimental_binary_cmaf`

The synthetic worker checks for that annotation so it only responds to the new
binary CMAF path.

## Binary Frame Format

Each response frame is encoded as:

1. `1 byte` frame kind
2. `4 bytes` big-endian payload length
3. `N bytes` payload

Frame kinds:

- `0x01` metadata JSON
- `0x02` CMAF init fragment
- `0x03` CMAF media fragment
- `0x04` error JSON
- `0x05` stream done

## Browser Client

The reference client is [client.html](./client.html). It:

- opens one `POST` request to `/v1/videos/stream/binary/cmaf`
- parses the framed response body
- creates a `MediaSource`
- appends the init fragment and media fragments into a continuous timeline
- lets you request a specific duration in seconds, or leave it blank to stream
  the full source video

The client uses `MediaSource`, so browser support depends on MSE availability.

## Same-Origin Demo Launcher

This example also includes [run_demo.py](./run_demo.py), which gives the browser
one origin without changing Dynamo CORS behavior.

What it does:

- starts `python -m dynamo.frontend` on an internal port
- serves `client.html` on a browser-facing port
- reverse-proxies `/v1/*` to the internal frontend

That means the browser only talks to one origin, so the CMAF `POST` stream works
without enabling CORS in Dynamo itself.

Example:

```bash
python examples/custom_backend/cmaf_binary_video_streaming/run_demo.py \
  --proxy-port 8080 \
  --frontend-port 18001
```

Then open:

```text
http://localhost:8080/
```

The client defaults its `Frontend Base URL` to the page origin, so no manual URL
editing should be needed in the simple case.

If you want to reuse an already-running frontend instead of starting a new one:

```bash
python examples/custom_backend/cmaf_binary_video_streaming/run_demo.py \
  --proxy-port 8080 \
  --frontend-port 8001 \
  --no-start-frontend
```

If you need extra frontend flags, pass them after `--`:

```bash
python examples/custom_backend/cmaf_binary_video_streaming/run_demo.py \
  --proxy-port 8080 \
  --frontend-port 18001 \
  -- --router-mode round-robin
```

## Browser Testing Note

This minimal branch intentionally does **not** add CORS support. That keeps the
Dynamo delta smaller, but it means the browser client needs to be served from
the same origin as the Dynamo frontend, or you need an external same-origin
proxy for local testing. `run_demo.py` is the built-in same-origin helper for
this example.

`curl` does not enforce browser CORS rules, so protocol inspection works even
when a browser page served from another origin would fail.

## Quick `curl` Check

Inspect the headers:

```bash
curl -i -N -X POST http://localhost:8000/v1/videos/stream/binary/cmaf \
  -H 'Content-Type: application/json' \
  -d '{"model":"synthetic-binary-cmaf-video-stream","prompt":"test binary cmaf stream","seconds":8,"size":"832x480"}'
```

Save the binary stream:

```bash
curl -N -X POST http://localhost:8000/v1/videos/stream/binary/cmaf \
  -H 'Content-Type: application/json' \
  -d '{"model":"synthetic-binary-cmaf-video-stream","prompt":"test binary cmaf stream","seconds":8,"size":"832x480"}' \
  -o /tmp/cmaf-stream.bin
```

Inspect the first bytes:

```bash
xxd -g 1 -l 96 /tmp/cmaf-stream.bin
```

## Tradeoffs

Compared with clip-based `video_streaming`:

- better UX: one continuous timeline instead of separate MP4 clips
- supports audio as part of the CMAF fragments
- still fits Dynamo’s single-request push model

Compared with `hls_video_streaming`:

- much smaller Dynamo delta
- no playlist generation or segment fetch routes
- no sticky routing or session lookup
- but the browser client is more custom because it uses `MediaSource` directly

## Follow-Ups

- Compare wire efficiency against the HLS/CMAF pull model
- Decide whether the frontend should keep this binary framing protocol or move
  to a more general binary media contract
- If browser testing from different origins is important, add a narrow CORS
  layer only for `/v1/videos/stream/binary/cmaf`
- Consider carrying richer metadata frames, such as duration, timestamp, or
  optional subtitles, if the protocol moves beyond this POC
