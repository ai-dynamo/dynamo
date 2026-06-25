# OTEL and Tokio Console Profiling

This harness is for the single-frontend, 32-mocker KV-routing benchmark:

- worktree: `/home/jothomson/Desktop/dynamo/.worktrees/singlefe-32mockers-kv-profile-20260625`
- harness: `/home/jothomson/Desktop/dynamo/.worktrees/singlefe-32mockers-kv-profile-20260625/benchmark-harness/singlefe-32mockers-kv-profile`
- default benchmark: one frontend on `0-7`, 32 mocker workers plus AIPerf on `8-N`, block size `16`, mocker speedup `25`, AgentX 060526 trace.

## OTEL

OTEL requires the normal runtime logging build. If the worktree was last built
with `dynamo-runtime/tokio-console`, switch it back first:

```bash
cd /home/jothomson/Desktop/dynamo/.worktrees/singlefe-32mockers-kv-profile-20260625/benchmark-harness/singlefe-32mockers-kv-profile/bin
./build_otel_frontend.sh
```

The system `otelcol` owns the default OTLP ports on this host, so the local file collector uses:

- gRPC: `127.0.0.1:14317`
- HTTP: `127.0.0.1:14318`
- collector self metrics: `127.0.0.1:18888`

Run a frontend-only OTEL pass:

```bash
cd /home/jothomson/Desktop/dynamo/.worktrees/singlefe-32mockers-kv-profile-20260625/benchmark-harness/singlefe-32mockers-kv-profile/bin

LOAD_SECS=120 GRACE=90 WARMUP=32 SETTLE=15 PROFILE_MODE=none \
  ./run_kv_32mockers_otel_once.sh 64
```

Add worker-side spans as well:

```bash
OTEL_INCLUDE_MOCKERS=1 LOAD_SECS=120 GRACE=90 WARMUP=32 SETTLE=15 PROFILE_MODE=none \
  ./run_kv_32mockers_otel_once.sh 64
```

Summarize the output:

```bash
./summarize_otel_spans.py ../profiles/singlefe-32mockers-kv-otel/<run>/otel-traces.json
```

The collector intentionally drops OTEL logs. Keep `frontend-debug-logging.toml` at `info` unless you explicitly want massive debug payloads; broad debug logging produced multi-GB artifacts and gRPC max-message errors.

## Tokio Console

Dynamo gates `console-subscriber` behind the `dynamo-runtime/tokio-console` Cargo feature. Build a console-enabled frontend outside the sandbox:

```bash
cd /home/jothomson/Desktop/dynamo/.worktrees/singlefe-32mockers-kv-profile-20260625/benchmark-harness/singlefe-32mockers-kv-profile/bin
./build_tokio_console_frontend.sh
```

Install the CLI under the harness:

```bash
./install_tokio_console_cli.sh
```

Smoke-test that the console-enabled frontend opens the console server:

```bash
./smoke_tokio_console_setup.sh
```

Run the benchmark with console recording enabled:

```bash
LOAD_SECS=120 GRACE=90 WARMUP=32 SETTLE=15 PROFILE_MODE=none \
  ./run_kv_32mockers_tokio_console_once.sh 64
```

In a second terminal, attach the TUI while the load is active:

```bash
cd /home/jothomson/Desktop/dynamo/.worktrees/singlefe-32mockers-kv-profile-20260625/benchmark-harness/singlefe-32mockers-kv-profile/bin
./run_tokio_console_tui.sh
```

Notes:

- Current Dynamo code binds the console subscriber to `0.0.0.0:6669` when the feature is enabled.
- `TOKIO_CONSOLE_RECORD_PATH` is set by the run wrapper so the frontend writes a persisted console recording under `profiles/singlefe-32mockers-kv-tokio-console/`.
- The current `cfg(feature = "tokio-console")` logging path replaces the normal JSONL/OTLP setup, so use separate builds/runs for Tokio Console and OTEL unless runtime logging is changed.
- Avoid importing the console-enabled Python extension inside the sandbox; it tries to bind the console subscriber socket during logging initialization.
