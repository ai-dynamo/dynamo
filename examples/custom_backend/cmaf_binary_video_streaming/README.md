# Synthetic CMAF Binary Video Streaming POC

This example is a third streaming POC that sits between the existing `video_streaming` and `hls_video_streaming` experiments.

- Like `video_streaming`, it keeps a single long-lived `POST` request open and streams bytes from the worker to the frontend to the browser.
- Like `hls_video_streaming`, it uses real CMAF/fMP4 media so the browser sees one continuous timeline instead of a sequence of standalone MP4 clips.

The goal is to test whether Dynamo can deliver a better browser playback experience without adopting HLS's pull-based playlist-and-segment routing model.

## What CMAF means here

`CMAF` stands for `Common Media Application Format`.

In this example, the source MP4 is packaged at worker startup into:

- one MP4 initialization fragment, `init.mp4`
- a sequence of fragmented MP4 media segments, `.m4s`

Unlike the clip-based `video_streaming` POC, those fragments are not independently playable files. They are meant to be appended to a browser `MediaSource` buffer to form one continuous stream.

## Why this POC exists

The existing `video_streaming` example validates Dynamo's single-request push model with minimal frontend changes, but it intentionally streams replayable standalone MP4 clips. That simplifies the transport contract, but it also limits UX:

- no continuous playback timeline
- visible clip boundaries
- no preserved audio track in the current synthetic worker

The HLS POC solves those UX issues, but it requires more Dynamo-specific machinery:

- kickoff route
- opaque stream IDs
- pull-based playlist and segment routes
- worker-affine routing for later `GET` requests

This example explores a middle path:

- keep Dynamo's natural push-streaming shape
- upgrade the media format from standalone MP4 clips to CMAF fragments
- use `MediaSource` in the browser instead of HLS playlists

## Requirements

- `ffmpeg` on `PATH`
- `ffprobe` on `PATH` (usually installed alongside `ffmpeg`)
- A local source MP4 file

## Worker

Start the standalone worker with a real MP4:

```bash
python examples/custom_backend/cmaf_binary_video_streaming/worker.py \
  --source-mp4 /path/to/source.mp4 \
  --segment-seconds 2 \
  --emit-cadence-ms 750 \
  --model synthetic-binary-cmaf-video-stream
```

The worker registers a `ModelType.Videos` backend and serves the usual `generate` endpoint. At startup it:

- inspects the source MP4 with `ffprobe`
- packages the source into CMAF fragments with `ffmpeg`
- preserves audio when the source contains an audio track
- prepares a metadata blob, one init fragment, and a sequence of media fragments

At request time it emits those pieces over the normal video `generate` stream using tagged `VideoData` items:

- `cmaf:metadata`
- `cmaf:init`
- `cmaf:segment:<n>`

## Frontend route

With Dynamo frontend running, the relevant route is:

- `POST /v1/videos/stream/binary/cmaf`

This is an experimental sibling of the existing binary route. It keeps one streaming `POST` response open and writes a framed byte stream:

```text
[kind:1][len_be_u32:4][payload:len]
```

Frame kinds:

- `0x01`: metadata JSON
- `0x02`: CMAF init fragment
- `0x03`: CMAF media fragment
- `0x04`: error JSON
- `0x05`: end-of-stream marker

The response also advertises:

- `Content-Type: application/octet-stream`
- `x-dynamo-video-binary-protocol: dynamo-video-binary-cmaf-v1`

## Reference client

Serve the HTML client:

```bash
python -m http.server 8080 -d examples/custom_backend/cmaf_binary_video_streaming
```

Then open `http://localhost:8080/client.html`.

The client:

- `POST`s once to `/v1/videos/stream/binary/cmaf`
- parses the binary frames
- creates a `MediaSource`
- appends the init fragment and media fragments into a `SourceBuffer`
- plays the result as one continuous video element timeline

This is a browser-native continuous playback experience, but without introducing HLS playlists or pull-based segment fetches.

## Browser note

This client depends on `MediaSource` / `SourceBuffer` support. Modern Chromium-based browsers and Safari should handle this reasonably well; older or constrained browsers may not.

If the page and the Dynamo frontend are served from different origins, browser testing also depends on the CORS headers currently enabled on the videos router.

As with the HLS example, the `Frontend Base URL` field must be reachable from the browser, not just from the shell. In remote IDE or forwarded-port setups, browser `localhost` may not point at the machine running Dynamo.

## Tradeoffs

Compared with `video_streaming`:

- better UX: continuous playback instead of clip-by-clip swapping
- optional preserved audio when the source MP4 has audio
- still one streaming `POST`, so no HLS session routing layer
- slightly more frontend protocol complexity because the browser must use `MediaSource`

Compared with `hls_video_streaming`:

- fewer Dynamo changes: no playlist route, no per-segment `GET`, no stream ID routing
- less browser standardization: this is a custom binary protocol, not HLS
- no reuse of off-the-shelf HLS players or CDN-style segment delivery

## Limitations

- This is still a synthetic worker using one prerecorded MP4 packaged at startup.
- The binary framing is custom and browser-specific. It is useful for evaluating Dynamo's push-streaming shape, not as a production-standard playback protocol.
- `MediaSource` support and codec compatibility depend on the browser.
- The worker reuses packaged fragments in memory. There is no durable storage tier or CDN offload.
- The current route is experimental and intentionally scoped to this POC contract.
- Browser testing currently depends on permissive CORS settings on the videos router when the page and API are on different origins.

## Follow-ups

- Compare this CMAF-over-binary path directly against the existing clip-based binary stream and the HLS pull path: startup latency, playback smoothness, audio continuity, and frontend CPU.
- Decide whether future binary media work should prefer:
  - standalone replayable units,
  - single-request CMAF over binary,
  - or HLS/CMAF with browser-standard pull semantics.
- If this path continues, consider tightening the binary contract further with typed metadata structures and more explicit fragment sequencing / timestamp information.
- Narrow CORS to the specific experimental routes or move the browser demo behind a same-origin proxy if we keep using standalone static clients.
