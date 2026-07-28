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
