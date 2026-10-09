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

## Runtime configuration

All three sidecars accept the same Dynamo runtime flags through both their
standalone executables and Python module launchers:

| Flag | Environment variable | Values | Default |
|------|----------------------|--------|---------|
| `--discovery-backend` | `DYN_DISCOVERY_BACKEND` | `etcd`, `kubernetes`, `file`, `mem` | `etcd` |
| `--request-plane` | `DYN_REQUEST_PLANE` | `tcp`, `nats` | `tcp` |
| `--response-plane` | `DYN_RESPONSE_PLANE` | `tcp`, `quic` | `tcp` |
| `--event-plane` | `DYN_EVENT_PLANE` | `zmq`, `nats` | `zmq` |

Explicit CLI values override the corresponding environment variables. When both
are unset, the runtime defaults apply. These settings are resolved before
connecting runtime dependencies or discovering engine metadata.

For example, with a vLLM engine listening on port 50051:

```bash
python -m dynamo.vllm.sidecar \
  --grpc-endpoint 127.0.0.1:50051 \
  --discovery-backend file \
  --request-plane tcp \
  --response-plane tcp \
  --event-plane zmq
```

Configure the frontend with matching transports and discovery settings. File
discovery requires all participants to share the same directory, selected by
`DYN_FILE_KV` (default: `$TMPDIR/dynamo_store_kv`). The engine's native gRPC
connection is configured separately with `--grpc-endpoint`.

## Health and startup

Set `DYN_SYSTEM_PORT` to enable the sidecar HTTP server. The standalone sidecar
executables and Python module launchers (`python -m dynamo.<engine>.sidecar`)
initialize runtime dependencies, then bind one listener before waiting for
engine metadata. A startup probe on `/live` remains unsuccessful until the
listener starts:

- `/live` returns HTTP 200 whenever the listener can respond, including while
  the engine is absent or loading and after runtime connectivity is lost.
- `/health` returns HTTP 503 while a required discovery or NATS connection is
  unavailable, and once shutdown starts.
  It returns HTTP 200 when those dependencies are reachable, independently of
  engine readiness and model registration. Discovery reads have a one-second
  timeout; NATS readiness follows the client connection state. Neither check
  issues inference requests.

The distributed runtime owns the listener. Sidecar launchers select runtime-only
probes; integrated workers retain their existing worker-health probe behavior.

The existing `DYN_SYSTEM_LIVE_PATH` and `DYN_SYSTEM_HEALTH_PATH` settings also
apply. Metrics, metadata, and engine routes are registered on the same listener
at startup. Keep a separate engine startup/readiness probe: a
healthy sidecar alone does not mean the engine can serve requests.

Shutdown withdraws sidecar readiness immediately. Metadata discovery before
`Worker.start()` is cancelled; once engine startup has begun, the existing Worker
startup, drain, and cleanup ordering remains in effect.

Continuous engine health reconciliation, engine replacement and KV recovery, and
changes to shutdown drain policy remain separate work. The synchronous engine
constructors used by embedded callers retain their existing behavior.

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

Plain `docker run` with no arguments uses the image `CMD` of `--help`, prints
usage, and exits `0`. Under Kubernetes, a container that overrides `command`
but omits `args` reaches the entrypoint with no engine name; it prints usage to
standard error and exits `2`, so the misconfiguration fails loudly.
