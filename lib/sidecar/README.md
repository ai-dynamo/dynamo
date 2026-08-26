# Sidecars

Rust sidecars connect Dynamo workers to inference engines over their native
gRPC APIs. Dynamo owns worker registration and request handling; the engine
runs in a separate process.

```text
common/         Shared gRPC arguments, transport, and errors
sglang/         SGLang sidecar
trtllm/         TensorRT-LLM sidecar
vllm/           vLLM sidecar
Dockerfile      Builds all three sidecar executables into a CPU-only image
dynamo-sidecar  Convenience entrypoint mapping vllm/sglang/trtllm to the above
```

Engine protocols and request conversion remain in each engine's crate.

## Build the image

There is no published sidecar image yet. `Dockerfile` builds one CPU-only image
carrying all three engine-specific executables — `dynamo-vllm-sidecar`,
`dynamo-sglang-sidecar`, and `dynamo-trtllm-sidecar` — in `/usr/local/bin`.
Official packaging is deferred to a follow-up change.

Build a multi-arch image from the repository root so it runs on any node —
`amd64` (x86) or `arm64` (GB200/Grace):

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -f lib/sidecar/Dockerfile \
  -t <your-registry>/dynamo-sidecar:1.3.0 --push .
```

To build faster for one architecture, pass just that platform (for example
`linux/arm64` for GB200/Grace).

### Selecting an engine

Deployments run the executable they need directly, as the container `command`
(see each backend's `deploy/` manifests):

```yaml
command:
- dynamo-vllm-sidecar
args:
- --grpc-endpoint
- 127.0.0.1:50051
```

The image's default entrypoint, `dynamo-sidecar`, is a convenience wrapper that
maps the short names `vllm`, `sglang`, and `trtllm` onto those executables, so
ad-hoc `docker run` needs only the engine name. Deployments override it with
`command`, so the two paths never interact:

```bash
docker run --rm <your-registry>/dynamo-sidecar:1.3.0 vllm --help
docker run --rm <your-registry>/dynamo-sidecar:1.3.0 sglang --help
docker run --rm <your-registry>/dynamo-sidecar:1.3.0 trtllm --help
```

Invoked without an engine name it prints that usage and exits `2`, so a
container that sets `command` but forgets `args` fails loudly.
