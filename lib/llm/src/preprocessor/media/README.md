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

Set media decoding options:

```python
from dynamo.llm import MediaDecoder
decoder = MediaDecoder()
decoder.image_decoder({"max_image_width": 4096, "max_image_height": 4096, "max_alloc": 16*1024*1024})
decoder.video_decoder({"strict": True, "fps": 2.0, "max_frames": 128, "max_alloc": 1024*1024*128*3})
```

And register the LLM as usual, adding the media configuration:

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

## Image decoding options

- **max_image_width** (uint32, > 0): If the image width exceeds this value, abort the decoding.
- **max_image_height** (uint32, > 0): If the image height exceeds this value, abort the decoding.
- **max_alloc** (uint64, > 0): Maximum allowed total allocation (RAM) of the decoder in bytes

## Video decoding options
###  Sampling
There are two ways to configure video sampling: either with a fixed number of frames, or with FPS-based sampling. Sampled frames are distributed uniformly in both cases.

- **num_frames** (uint32, > 0): Attempt to decode exactly this number of frames from the input video.
- **fps** (float32, > 0) and optionally **max_frames** (uint32, > 0): Attempt to decode at a given framerate, with a potential cap on the number of decoded frames.

### Others
- **strict** (bool): if strict mode is enabled, any failure to decode a requested frame will abort the whole video decoding and error out. When strict mode is disabled, it is possible that the decoding of some requested frame fails, and the resulting set of decoded frames might container fewer frames than expected.

- **max_alloc** (usize, > 0): If the total number of bytes in the decoded frames would exceed this value, abort the decoding.



## TODOs

### Modalities

- [x] Image decoding
- [x] Video decoding
- [ ] Audio decoding

### Performance

- [x] Image SW decoding
- [ ] Video HW decoding (NVDEC)
- [ ] JPEG HW decoding (nvJPEG)
- [ ] Sparse video sampling (seek-forward)
- [ ] Memory slab pre-allocation/registration

### Memory management
- [ ] Memory spilling to lower storage tiers
- [ ] Early-free memory on client notifications

### Misc
- [ ] Observability on performance, memory usage and input distributions
- [ ] Per-request decoding options
