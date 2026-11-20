<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CMAF Video Streaming

Clients of `POST /v1/videos` wait for the whole clip to be generated, encoded,
and stored before the first frame plays. This route streams the encoding and
delivery instead: once generation has completed, the clip goes out as CMAF
(fragmented MP4), an init segment followed by media fragments, each one playable
as it arrives. Generation itself is not incremental yet.

```
POST /v1/videos/stream/cmaf
  -> [kind:u8][len:u32 big-endian][payload] ...
```

| Kind | Name | Payload |
|------|------|---------|
| `0x01` | metadata | JSON: `mime_type`, `video_codec`, `width`, `height`, `fps`, `target_duration`, `segment_count` |
| `0x02` | init | fMP4 init segment (`ftyp` + `moov`), sent once |
| `0x03` | segment | one media fragment (`moof` + `mdat`), starts with a keyframe |
| `0x04` | error | UTF-8 message; terminal, the output is incomplete |
| `0x05` | done | empty; terminal, end of stream |

The timeline is authored by one long-lived ffmpeg process per request, so
`mfhd.sequence_number` and `tfdt.baseMediaDecodeTime` are continuous and the
client does no timestamp arithmetic. Requesting this route is the only trigger:
nothing in a request body switches a plain `/v1/videos` call to CMAF output.

The encoder is software VP9 (`libvpx-vp9`), the same codec `/v1/videos` produces,
run from `DYN_FFMPEG_PATH` (default `/usr/local/bin/ffmpeg`).
`DYN_CMAF_GOP_FRAMES` sets the segment length (default 8 frames; must be a
positive multiple of 4). The stream carries one video track; generated audio is
not included. The vLLM-Omni backend implements this route.

> [!NOTE]
> Not every runtime image ships `/usr/local/bin/ffmpeg`. If the worker refuses
> to start a stream because that path is not executable, point
> `DYN_FFMPEG_PATH` at an ffmpeg the image does carry that includes
> `libvpx-vp9`.

## Run it

Start a video worker, for example the aggregated vLLM-Omni one:

```bash
./examples/backends/vllm/launch/agg_omni_video.sh
```

Then stream from the headless client, which writes `init.mp4`, `segNNN.m4s`, a
playable concatenated `stream.mp4`, and prints the arrival time of every frame:

```bash
python examples/custom_backend/cmaf_binary_video_streaming/cmaf_client.py \
    --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "Dog running on a beach" \
    --num-frames 33 --steps 20 --out /tmp/cmaf
```

## Play it in a browser

[`client.html`](client.html) feeds each fragment to an MSE `SourceBuffer` as it
lands. Serve it with [`run_proxy.py`](run_proxy.py), which waits for the frontend
and then answers both the page and `/v1/*` on one port:

```bash
python examples/custom_backend/cmaf_binary_video_streaming/run_proxy.py \
    --bind 0.0.0.0 --proxy-port 8080 --frontend-port 8000
```

Open `http://<host>:8080/` and press **Stream**. The page must share an origin
with the API: the frontend sends no CORS headers, so a page served on another
port cannot POST JSON to it. The browser's preflight is refused and `fetch`
fails with `TypeError: Failed to fetch` before the request is sent. A plain
`python -m http.server` will not work for that reason.

The page asks for a fixed 30 frames at 16 fps, 832x480, 20 denoising steps. The
server default is 97 frames, and since nothing reaches the browser until the
whole clip is denoised, the default looks like a hang on a single accelerator.
The `NUM_FRAMES`, `FPS`, `SIZE` and `NUM_INFERENCE_STEPS` constants at the top of
`client.html`'s script change it.
