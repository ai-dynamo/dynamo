# Media decoding in the frontend


This component performs media download, base64 decoding, media decoding and NIXL registration. Today, this is used in the OpenAI preprocessor, to transform multimodal inputs (image_url, video_url, audio_url) into fully decoded data (pixel values, ...) accessible to the backends via NIXL.



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

### Observability
- [ ] Observability on performance, memory usage and input distributions
