---
name: kvbm-diagnose
description: Read-only triage of a failed KVBM run — tails vllm log, grep for stack-health signatures, fetch live /metrics
user-invocable: true
disable-model-invocation: true
---

# KVBM Diagnose

Read-only triage aid for a failed or suspicious KVBM run. Tails the most recent per-test vllm log, matches it against the signatures of a healthy stack, and (optionally) fetches live `/metrics` from a running server. Surfaces known-bad patterns with concrete fix pointers.

**Never modifies anything**. No rebuilds, no test runs, no apt/pip. Only `grep`, `tail`, `curl`, `ls`, `awk`.

## Arguments

`/dynamo:kvbm:diagnose [--log PATH] [--metrics-url URL] [--spec SPEC_ID]`

- **--log PATH**: Explicit log file to inspect. If omitted, uses the most recent `/tmp/dynamo_tests/*/ServerType.vllm_server_*.log`.
- **--metrics-url URL**: Metrics endpoint to query (e.g. `http://localhost:8081`). Defaults to `$KVBM_EXTERNAL_METRICS_PORT` on localhost if set.
- **--spec SPEC_ID**: If provided, compare host hit rate and wall time to the recorded phase-5 baseline for that spec.

## Step 1: Locate The Log

```bash
if [ -z "$LOG" ]; then
    LOG=$(ls -t /tmp/dynamo_tests/*/ServerType.vllm_server_*.log 2>/dev/null | head -1)
fi
test -f "$LOG" || { echo "no log found — pass --log or run a test first"; exit 1; }
echo "log: $LOG"
ls -la "$LOG"
```

Show file size and mtime so the user can confirm it's the run they intended.

## Step 2: Stack Health Checklist

Grep the log for each expected line and report PRESENT/MISSING:

| Expected line | Meaning |
|---|---|
| `KvConnectorWorker initialized with worker_id:` | v1/v2: connector worker bootstrapped |
| `Auto-detected device layout:` | v1/v2: tensor layout introspected |
| `Layout: num_layers=` | v1/v2: block geometry resolved |
| `KvbmWorker num_device_blocks=` | v1: GPU block pool sized |
| `ConnectorLeader initialized with onboard mode onboard_mode=` | v2 only: leader launched with onboard mode (Intra|Inter) |
| `Application startup complete` | vllm fully booted |
| `engine is ready` / `Engine started` | vllm engine bootstrapped |

Print as:

```
Stack Health
────────────
[OK]   KvConnectorWorker initialized
[OK]   Auto-detected device layout (LayerSeparate)
[OK]   Layout resolved (num_layers=28 page_size=16)
[OK]   KvbmWorker num_device_blocks=512
[MISS] ConnectorLeader initialized with onboard mode   <-- v2 only
[OK]   Application startup complete
```

For v2 logs, also extract the actual onboard mode: `grep -oE 'onboard_mode=(Intra|Inter)' "$LOG" | head -1`.

## Step 3: Known-Bad Signatures

Grep for failure patterns and report any hits with a fix pointer:

| Pattern | Diagnosis | Fix |
|---|---|---|
| `cudaErrorNoKernelImageForDevice` | sm_121 venv missing Blackwell kernels | `/dynamo:kvbm:sandbox-venv` |
| `undefined symbol:` (in torch or kvbm import) | torch/kvbm ABI drift | `/dynamo:kvbm:maturin-dev --clean` |
| `undefined symbol: ncclDevCommDestroy` | nccl rolled back to 2.28.9 | `uv pip install --force-reinstall --no-deps 'nvidia-nccl-cu13>=2.29'` |
| `AttributeError: module '_core.v2' has no attribute 'RustScheduler'` | Expected when scheduler module commented out (`src/v2/mod.rs:8`). ONLY a problem if outside the widened `except` in `schedulers/dynamo.py:47` | Verify `schedulers/dynamo.py` catches `(ImportError, AttributeError)` |
| `FP8` / `sm_120 kernel not found` | FP8 path on sm_121 (eugr/spark-vllm-docker#143) | Switch to a non-FP8 model variant |
| `no kv_connector_module_path` | kv-transfer-config missing connector module path | Check `KvbmServerManager` builder in `tests/kvbm_integration/fixtures/server.py` |
| `zmq` connect timeout | Leader↔worker ZMQ handshake stalled | Check worker died first; look for earlier CUDA OOM / kernel errors |
| `CUDA out of memory` | `--gpu-memory-utilization` too high for the spec | Lower `KVBM_GPU_BLOCKS` or bump `KVBM_GPU_MEMORY_UTILIZATION` knob |
| `ConnectorLeader initialized with onboard mode onboard_mode=Inter` when intra was requested | Builder didn't inject `leader.onboard.mode` — default is `Inter` at `lib/kvbm-config/src/onboard.rs:40` | Check `build_kv_transfer_config` in `tests/kvbm_integration/fixtures/server.py` |

Print any matches with 2 lines of context before and after.

## Step 4: Live Metrics Snapshot (If URL Provided Or KVBM_EXTERNAL_METRICS_PORT Set)

```bash
if [ -z "$METRICS_URL" ] && [ -n "$KVBM_EXTERNAL_METRICS_PORT" ]; then
    METRICS_URL="http://localhost:$KVBM_EXTERNAL_METRICS_PORT"
fi

if [ -n "$METRICS_URL" ]; then
    curl -sf "$METRICS_URL/metrics" | grep '^kvbm_' || echo "metrics endpoint unreachable or no kvbm_ counters"
fi
```

Highlight the offload/onboard counters specifically:

```
kvbm_offload_blocks_d2h  <value>   (expect > 0 after first cache reset)
kvbm_onboard_blocks_h2d  <value>   (expect > 0 after first cache reset)
```

Also surface host hit rate if present in the log (search for `Host: ` / `host hit rate`):

```bash
grep -oE 'Host: [0-9.]+%' "$LOG" | tail -5
```

## Step 5: Baseline Comparison (If --spec Provided)

Recorded baselines from ACTIVE_PLAN phase 5, Qwen3-0.6B at default iteration counts on GB10:

| Spec | Wall time | Host hit rate |
|---|---|---|
| `v1-Qwen3-0.6B` | 190.47s | 66.4% |
| `v2-Qwen3-0.6B-intra` | 182.63s | 67.0% |
| `v2-Qwen3-0.6B-inter` | 183.76s | 66.7% |

If the current run's metrics are materially off (>10% deviation on hit rate, >30% on wall time), flag it.

## Step 6: Historical Gotchas

If the stack-health grep showed garbage output on a v2 intra run, call out the phase-5 regression:

```
Historical gotcha: intra onboard silent no-op
─────────────────────────────────────────────
ACTIVE_PLAN phase 5 found three spurious #[cfg(feature = "nccl")] gates
wrapping pure-CUDA code (H2D stream + per-layer CudaEvent::record) in:

  lib/kvbm-engine/src/worker/physical.rs
  lib/kvbm-connector/src/connector/worker/mod.rs

When gated off, intra onboard becomes a silent no-op and vLLM reads
uninitialized KV slots. If this branch re-introduced those gates,
the fix is to delete them (the code is not nccl-dependent; cudarc is
unconditional in kvbm-engine).
```

## Step 7: Report

Present a single structured summary:

```
KVBM Diagnose — <log filename>
──────────────────────────────
Log:     <path>
Age:     <N> minutes
Size:    <bytes>

Stack Health:
  [N/M checks passed]
  [List missing or failed checks]

Known-Bad Hits:
  [List matches with fix pointers]

Metrics (if available):
  kvbm_offload_blocks_d2h = <N>
  kvbm_onboard_blocks_h2d = <N>
  host hit rate           = <N>%

Baseline deviation (if --spec):
  wall time  <actual> vs <baseline>   [WITHIN | ABOVE | BELOW]
  hit rate   <actual>% vs <baseline>% [WITHIN | ABOVE | BELOW]

Recommended next step:
  <concrete skill invocation based on findings>
```

## Reference: Log Locations

| Location | Content |
|---|---|
| `/tmp/dynamo_tests/<test-id>/ServerType.vllm_server_*.log` | Per-test vllm stdout+stderr from `KvbmServerManager` |
| `/tmp/kvbm-run-server-logs/` | Logs from decomposed `run_server.sh` launches |
| vllm stdout/stderr during `run_server.sh` | Foreground on shell 2 |
