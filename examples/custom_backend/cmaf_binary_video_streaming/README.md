# CMAF Video Streaming

Clients of `POST /v1/videos` wait for the whole clip before the first frame plays.
This route streams a generated clip as CMAF (fragmented MP4) instead: an init
segment followed by media fragments, each one playable as it arrives.

```
POST /v1/videos/stream/cmaf
  -> [kind:u8][len:u32 big-endian][payload] ...
```

| Kind | Name | Payload |
|------|------|---------|
| `0x01` | metadata | JSON: `mime_type`, `width`, `height`, `fps`, `target_duration`, `segment_count` |
| `0x02` | init | fMP4 init segment (`ftyp` + `moov`), sent once |
| `0x03` | segment | one media fragment (`moof` + `mdat`), starts with an IDR |
| `0x04` | error | UTF-8 message; terminal, the output is incomplete |
| `0x05` | done | empty; terminal, end of stream |

The timeline is authored by one long-lived ffmpeg process per request, so
`mfhd.sequence_number` and `tfdt.baseMediaDecodeTime` are continuous and the
client does no timestamp arithmetic. Requesting this route is the only trigger —
no request field switches a plain `/v1/videos` call to CMAF output.

Encoder selection follows DEP 0016: hardware AV1 (`av1_vaapi`) when
`DYN_XPU_FFMPEG_PATH` points at a VA-API ffmpeg, software VP9 (`libvpx-vp9`)
otherwise, from `DYN_FFMPEG_PATH` (default `/usr/local/bin/ffmpeg`). There is no
fallback in either direction. `DYN_CMAF_GOP_FRAMES` sets the segment length
(default 8 frames; must be a positive multiple of 4).

> [!NOTE]
> Not every runtime image ships `/usr/local/bin/ffmpeg`. If the worker refuses
> to start a stream because that path is not executable, point
> `DYN_FFMPEG_PATH` at the ffmpeg the image does carry — `/usr/bin/ffmpeg` on
> the XPU image.

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
port cannot POST JSON to it — the browser's preflight is refused and `fetch`
fails with `TypeError: Failed to fetch` before the request is sent. A plain
`python -m http.server` will not work for that reason.

The page asks for a fixed 30 frames at 16 fps, 832x480, 20 denoising steps. The
server default is 97 frames, and since generation is not incremental yet nothing
reaches the browser until the whole clip is denoised, so the default looks like a
hang on a single accelerator. The `NUM_FRAMES`, `FPS`, `SIZE` and
`NUM_INFERENCE_STEPS` constants at the top of `client.html`'s script change it.

See DEP 0017 for the design and DEP 0016 for the encoder policy it builds on.
