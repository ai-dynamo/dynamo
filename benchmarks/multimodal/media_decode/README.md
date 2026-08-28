# Media Decode Benchmarks

Reproducible entry points for benchmarking frontend media decoding. The Rust
Criterion implementations remain with their owning crates; these scripts set
the comparison environment and invoke the relevant benchmark target.

## Image

`run_image.sh` compares Rust `image::ImageReader` with the default
libjpeg-turbo decoder. It runs:

- A synthetic 2400x1080 RGB JPEG latency benchmark.
- A 3840x2160 RGB JPEG sweep with 100 images at C1, C8, and C32.

Install the `libturbojpeg` runtime package, then run from any directory:

```bash
./benchmarks/multimodal/media_decode/run_image.sh
```

Criterion arguments can be passed through to select a benchmark or save a
baseline:

```bash
./benchmarks/multimodal/media_decode/run_image.sh \
  image_decode_jpeg_3840x2160_batch_100

./benchmarks/multimodal/media_decode/run_image.sh --save-baseline before
```

Results are written under `target/criterion/`. The script requires
libjpeg-turbo instead of allowing the configured decoder to fall back to
`image::ImageReader`.

The implementation is in
[`lib/llm/benches/image_decode.rs`](../../../lib/llm/benches/image_decode.rs).
The 4K sweep remains opt-in at the Cargo benchmark level, so
`cargo test --all-targets` does not execute it in CI.
