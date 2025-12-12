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

## TODOs

### Modalities

- [x] Image decoding
- [ ] Video decoding
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
