# Sidecars

Rust sidecars connect Dynamo workers to inference engines over their native
gRPC APIs. Dynamo owns worker registration and request handling; the engine
runs in a separate process.

```text
common/     Shared gRPC arguments, transport, and errors
sglang/     SGLang sidecar
trtllm/     TensorRT-LLM sidecar
vllm/       vLLM sidecar
Dockerfile  Unified CPU-only image holding all three sidecar executables
```

Engine protocols and request conversion remain in each engine's crate.

## Build the image

There is no published sidecar image yet. `Dockerfile` builds one CPU-only image
containing the vLLM, SGLang, and TensorRT-LLM executables. The image has no
default entrypoint: each deployment selects the executable it needs with its
container `command`. Official packaging is deferred to a follow-up change.

Build a multi-arch image from the repository root so it runs on any node —
`amd64` (x86) or `arm64` (GB200/Grace):

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -f lib/sidecar/Dockerfile \
  -t <your-registry>/dynamo-sidecar:1.3.0 --push .
```

To build faster for one architecture, pass just that platform (for example
`linux/arm64` for GB200/Grace).
