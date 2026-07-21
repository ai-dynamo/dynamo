# Sidecars

Rust sidecars connect Dynamo workers to inference engines over their native
gRPC APIs. Dynamo owns worker registration and request handling; the engine
runs in a separate process.

```text
common/  Shared gRPC arguments, transport, and errors
sglang/  SGLang sidecar
vllm/    vLLM sidecar
```

Engine protocols and request conversion remain in each engine's crate.
