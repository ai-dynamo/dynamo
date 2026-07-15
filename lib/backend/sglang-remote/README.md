# SGLang remote backend

`dynamo-sglang-remote` connects Dynamo's unified worker lifecycle to an
out-of-process SGLang engine through SGLang's native gRPC service. It is a
standalone Rust executable, not a Python extension or importable sidecar
module.

Install the binary-only wheel to place the executable on `PATH`:

```bash
pip install ai-dynamo-sglang-remote
dynamo-sglang-remote --sglang-endpoint http://127.0.0.1:30001
```

## SGLang-managed process contract

An SGLang launcher can supervise the executable directly after its native gRPC
listener is ready:

```bash
python3 -m sglang.launch_server \
    <args> \
    --grpc-port 30001 \
    --sidecar-executable dynamo-sglang-remote
```

The corresponding SGLang implementation should:

1. Resolve the executable without invoking a shell.
2. Spawn `dynamo-sglang-remote --sglang-endpoint <loopback-url>` only after
   native gRPC is listening. Preserve the parent environment so Dynamo's
   namespace, discovery, and observability settings reach the worker.
3. Keep the executable as the directly supervised child. A spawn wrapper may
   install SGLang's parent-death handling and then call `os.execvp`, which
   preserves the child PID across the exec.
4. Treat an unexpected or non-zero child exit as fatal to the SGLang server.
5. On shutdown, send `SIGTERM`, wait for Dynamo's graceful lifecycle, and then
   kill the remaining process tree if the deadline expires.

The supervisor timeout must be configurable. Its default must exceed Dynamo's
combined release-mode shutdown budget: the 5-second
`DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS` default plus the 30-second
`DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT` default. A 40-second default leaves a
small supervision margin. Operators that increase either Dynamo value must
increase the SGLang timeout as well.

SGLang's gRPC discovery response supplies the model identity and aggregated,
prefill, or decode role, so the managed executable needs only the endpoint
argument. Prefill deployments may additionally set
`SGLANG_DISAGGREGATION_BOOTSTRAP_HOST` when the discovered address is not
routable from decode workers.
