# Synthetic Video Streaming POC

This example provides:

- A synthetic backend worker that loads a prerecorded MP4, slices it into replayable standalone MP4 clips, and emits one `NvVideosResponse` per clip.
- A browser reference client that can consume either the canonical SSE route or the experimental binary route and play the clips sequentially.

## Why standalone clips?

This POC intentionally uses self-contained MP4 clips as the replayable unit instead of low-latency fragmented MP4. That is a deliberate shortcut to validate the protocol contract first: each streamed chunk is independently playable on its own, which is what the eventual backend mux stage will need to guarantee.

## Worker

Start the worker with a real local MP4 file:

```bash
python examples/custom_backend/video_streaming/worker.py \
  --source-mp4 /path/to/source.mp4 \
  --fragment-seconds 2 \
  --emit-cadence-ms 750 \
  --model synthetic-video-stream
```

The worker registers itself as a `ModelType.Videos` backend and emits one `NvVideosResponse` per clip.

## Frontend routes

With Dynamo frontend running, the relevant routes are:

- `POST /v1/videos/stream`
  Canonical SSE route. Each `data:` event contains an `NvVideosResponse` JSON payload with `data[0].b64_json`. For this POC, the route contract assumes replayable MP4 clips.
- `POST /v1/videos/stream/binary`
  Experimental framed binary route. The framing protocol is:

```text
[kind:1][len_be_u32:4][payload:len]
```

Frame kinds:

- `0x01`: MP4 clip bytes
- `0x02`: error JSON
- `0x03`: end-of-stream marker

## Reference client

Serve the HTML client from this directory:

```bash
python -m http.server 8080 -d examples/custom_backend/video_streaming
```

Then open `http://localhost:8080/client.html`.

The client uses `fetch()` so it can `POST` to the video streaming endpoints, parse either SSE or the binary framing, and play the queued MP4 clips one after another.
