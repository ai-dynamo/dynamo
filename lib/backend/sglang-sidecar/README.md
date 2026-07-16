# SGLang sidecar

`dynamo-sglang-sidecar` connects Dynamo's unified worker lifecycle to an
out-of-process SGLang engine through SGLang's native gRPC service. It is a
standalone Rust executable and is also compiled into `ai-dynamo-runtime` for
the importable `dynamo.sglang.sidecar` launcher.

Build and run it directly from the Dynamo workspace:

```bash
cargo build --release -p dynamo-sglang-sidecar
./target/release/dynamo-sglang-sidecar \
    --sglang-endpoint http://127.0.0.1:30001
```

Distribution and container packaging for the standalone executable are
intentionally deferred to a follow-up change.

## SGLang-managed module contract

SGLang can load the Python entry point and supply the gRPC endpoint arguments:

```bash
python3 -m sglang.launch_server \
    <args> \
    --grpc-port 30001 \
    --sidecar dynamo.sglang.sidecar
```

The entry point configures Dynamo logging when `main()` runs, then calls the
private `dynamo._core.backend._run_sglang_sidecar(argv)` binding. The binding
prepends the executable name expected by clap, releases the GIL, and runs the
same unified worker lifecycle as the standalone executable.

## SGLang-managed executable contract

An SGLang launcher can supervise the executable directly after its native gRPC
listener is ready:

```bash
python3 -m sglang.launch_server \
    <args> \
    --grpc-port 30001 \
    --sidecar-executable dynamo-sglang-sidecar
```

The corresponding SGLang implementation should:

1. Resolve the executable without invoking a shell.
2. Spawn `dynamo-sglang-sidecar --sglang-endpoint <loopback-url>` only after
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
