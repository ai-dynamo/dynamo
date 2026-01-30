# Media decoding in the frontend


This component performs media download, base64 decoding, media decoding and NIXL registration. Today, this is used in the OpenAI preprocessor, to transform multimodal inputs (image_url, video_url, audio_url) into fully decoded data (pixel values, ...) accessible to the backends via NIXL.

## Usage

Media decoding is enabled when registering the MDC:

Set HTTP download options:

```python
from dynamo.llm import MediaFetcher
fetcher = MediaFetcher()
fetcher.user_agent("dynamo")
fetcher.timeout_ms(15000)
fetcher.allow_direct_ip(True)
fetcher.allow_direct_port(False)
fetcher.allowed_media_domains(["google.com"])
```

Set media decoding default options and limits:

```python
from dynamo.llm import MediaDecoder
decoder = MediaDecoder()
decoder.enable_image({"limits": {"max_image_width": 4096, "max_image_height": 4096, "max_alloc": 16*1024*1024}})
decoder.enable_video({"fps": 2.0, "max_frames": 128, "limits": {"max_alloc": 1024*1024*128*3}})
```

If `enable_image` or `enable_video` are not called, requests containing the corresponding modality will be rejected.

Register the LLM as usual, adding the media configuration:

```python
register_llm(
  ...,
  media_decoder=decoder,
  media_fetcher=fetcher,
)
```


## Known Limitations

> [!WARNING]
> **Incompatible with `Dockerfile.frontend`**: Frontend media decoding (enabled with `--features media-nixl`) is not supported when using `Dockerfile.frontend`. The frontend image built from `Dockerfile.frontend` does not enable the feature + include the required NIXL/UCX dependencies.

> [!WARNING]
> **Requires GPU node**: The frontend must run on a node with GPU access. During media processing, decoded tensors are written to GPU memory via NIXL, which requires `libcuda.so.1` to be available. Running the frontend on a CPU-only node will fail with something like: `Failed to initialize required backends: [UCX: No UCX plugin found]`.

> [!WARNING]
> **Video decoding**: Video decoding needs to be enabled via the `dynamo-llm/media-ffmpeg` rust feature. The following ffmpeg dynamic libraries must be available on the system: `libavcodec`, `libavdevice`, `libavfilter`, `libavformat`, `libswresample`, `libswscale`. These are available in dynamo containers built with `container/build.sh --enable-media-ffmpeg ...`

## Image decoding options

### Limits (not overridable at runtime via `media_io_kwargs`)
- **limits.max_image_width** (uint32, > 0): If the image width exceeds this value, abort the decoding.
- **limits.max_image_height** (uint32, > 0): If the image height exceeds this value, abort the decoding.
- **limits.max_alloc** (uint64, > 0): Maximum allowed total allocation (RAM) of the decoder in bytes

## Video decoding options
### Sampling
There are two ways to configure video sampling: either with a fixed number of frames, or with FPS-based sampling. Sampled frames are distributed uniformly in both cases.

- **num_frames** (uint32, > 0): Attempt to decode exactly this number of frames from the input video.
- **fps** (float32, > 0) and optionally **max_frames** (uint32, > 0): Attempt to decode at a given framerate, with a potential cap on the number of decoded frames.

### Others
- **strict** (bool): if strict mode is enabled, any failure to decode a requested frame will abort the whole video decoding and error out. When strict mode is disabled, it is possible that the decoding of some requested frame fails, and the resulting set of decoded frames might container fewer frames than expected.

### Limits (not overridable at runtime via `media_io_kwargs`)
- **limits.max_alloc** (usize, > 0): If the total number of bytes in the decoded frames would exceed this value, abort the decoding.


## Runtime media decoding options (`media_io_kwargs`)

Parameters of the decoders, can also be set at runtime via an extension to the OpenAI chat completions API. Limits defined in the MDC such as maximum image size, maximum RAM allocation, cannot be overridden at runtime.

This can be used for example to set the video sampling strategy for a request, that differs from the default one registered in the MDC:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": ...,
    "messages": ...,
    "media_io_kwargs": {
        "video": {
            "fps": 1.0,
            "max_frames": 16
        }
    }
  }'
```

## TODOs

### Modalities

- [x] Image decoding
- [x] Video decoding
- [ ] Audio decoding

### Performance

- [x] Image SW decoding
- [ ] Video HW decoding (NVDEC)
- [ ] JPEG HW decoding (nvJPEG)
- [x] Sparse video sampling (seek-forward)
- [ ] Memory slab pre-allocation/registration

### Memory management
- [ ] Memory spilling to lower storage tiers
- [ ] Early-free memory on client notifications

### Misc
- [ ] Observability on performance, memory usage and input distributions
- [x] Per-request decoding options
