---
name: kvbm-decomposed-run
description: Run the three-shell decomposed KVBM determinism flow (deps → server → eval) with spec-id handshake
user-invocable: true
disable-model-invocation: true
---

# KVBM Decomposed Run

Run the spec-id-driven decomposed determinism flow introduced in ACTIVE_PLAN phase 2. Three shells (or three pytest sessions), independently restartable, driven by a `KVBM_SPEC_ID` handshake so shell 3 can never pick up a parametrization shell 2 wasn't launched for.

The fast local iteration loop for KVBM: iterate the eval without re-spawning vllm.

## Arguments

`/dynamo:kvbm:decomposed-run <spec-id> [--fast] [--attach URL] [--enable-mla]`

- **spec-id** (required): `KvbmServerSpec.id` from `tests/kvbm_integration/test_determinism_agg.py:_CACHE_RESET_SPECS`. See table below.
- **--fast**: Use reduced iteration counts (`KVBM_MAX_ITERATIONS=2 KVBM_NUM_ITERATIONS=2 KVBM_REQUEST_DELAY=2`). Drops a full run to <1 min.
- **--attach URL**: Skip shells 1+2 entirely; run only shell 3 against a server already running at URL. Pairs with `KVBM_EXTERNAL_METRICS_PORT`.
- **--enable-mla**: Set `KVBM_ENABLE_MLA=1` for MLA specs (DeepSeek-V2-Lite).

## Spec ID Reference

Authoritative source: `_CACHE_RESET_SPECS` in `tests/kvbm_integration/test_determinism_agg.py`.

| Spec id | Model | KVBM | Onboard | Notes |
|---|---|---|---|---|
| `v1-DeepSeek-R1-Distill-Llama-8B` | R1-Distill-Llama-8B | v1 | n/a | Default spec; slow on GB10 (~30 min at default iterations) |
| `v1-DeepSeek-V2-Lite` | V2-Lite | v1 | n/a | MLA — requires `--enable-mla` |
| `v2-DeepSeek-R1-Distill-Llama-8B-intra` | R1-Distill-Llama-8B | v2 | intra | |
| `v2-DeepSeek-R1-Distill-Llama-8B-inter` | R1-Distill-Llama-8B | v2 | inter | |
| `v2-DeepSeek-V2-Lite-intra` | V2-Lite | v2 | intra | MLA |
| `v2-DeepSeek-V2-Lite-inter` | V2-Lite | v2 | inter | MLA |

**Qwen3-0.6B path (standing GB10 model)**: the default R1-Distill spec model is read from `KVBM_MODEL_ID`. To run `v1-Qwen3-0.6B` / `v2-Qwen3-0.6B-intra` / `v2-Qwen3-0.6B-inter`, set `KVBM_MODEL_ID=Qwen/Qwen3-0.6B` before invoking this skill. When unset, the first `_MODEL_CONFIGS` entry is R1-Distill and the first spec id has the `-DeepSeek-R1-Distill-Llama-8B` suffix.

**ID verification** (run before launching if unsure):

```bash
.sandbox/bin/python - <<'PY'
import os
os.environ.setdefault("KVBM_MODEL_ID", "Qwen/Qwen3-0.6B")
from tests.kvbm_integration.test_determinism_agg import _CACHE_RESET_SPECS
for s in _CACHE_RESET_SPECS:
    print(s.id)
PY
```

## Step 1: Preflight

```bash
# Venv ready?
test -x .sandbox/bin/python || { echo "run /dynamo:kvbm:sandbox-venv first"; exit 1; }

# kvbm imports?
.sandbox/bin/python -c "import kvbm; from kvbm.v1.vllm.connector import DynamoConnector; from kvbm.v2.vllm.connector import DynamoConnector" \
    || { echo "run /dynamo:kvbm:maturin-dev first"; exit 1; }

# GPU available?
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
```

If `--attach URL` was passed, also verify the URL responds:

```bash
curl -sf "$URL/v1/models" >/dev/null || { echo "attach URL not reachable: $URL"; exit 1; }
```

## Step 2: Show Plan and Confirm

```
KVBM Decomposed Run
───────────────────
Spec:        <spec-id>
KVBM:        <v1|v2>
Onboard:     <n/a|intra|inter>
Model:       <model-id>
Fast mode:   <yes|no>
MLA gate:    <enabled|disabled>
Mode:        <spawn|attach>

Shells:
  1. deps    (bash run_deps_v1.sh)        [v1 only]
  2. server  (bash run_server.sh <spec>)  [skipped if --attach]
  3. eval    (bash run_eval.sh)
```

Note for v2 specs: **shell 1 is skipped** — v2 agg discovery defaults to filesystem/None and doesn't need NATS/etcd (per `lib/kvbm-config/src/messenger.rs:43`). `run_server.sh` will reject v2 specs that have `NATS_SERVER`/`ETCD_ENDPOINTS` set.

Confirm before proceeding.

## Step 3: Shell 1 — deps (v1 only)

Skip for v2 specs or `--attach`.

```bash
bash tests/kvbm_integration/scripts/run_deps_v1.sh
```

The script probes `NATS_SERVER` / `ETCD_ENDPOINTS` (defaults `nats://localhost:4222` / `http://localhost:2379`). If both are reachable, it prints the exports and exits 0 immediately. Otherwise it spawns fresh instances and stays foreground (Ctrl-C to stop).

Capture the two `export` lines the script prints:

```
export NATS_SERVER=nats://localhost:4222
export ETCD_ENDPOINTS=http://localhost:2379
```

Export them in the shell that will run step 4.

## Step 4: Shell 2 — server

Skip if `--attach URL` was passed; jump to step 5.

```bash
# For v1 specs, the NATS/ETCD exports from step 3 must be in this shell's env
export NATS_SERVER=... ETCD_ENDPOINTS=...

# Optionally set Qwen path
export KVBM_MODEL_ID=Qwen/Qwen3-0.6B  # only if using a Qwen spec id

# For MLA specs
export KVBM_ENABLE_MLA=1  # if --enable-mla

# Optional cache-size overrides
export KVBM_CPU_BLOCKS=2000
export KVBM_GPU_BLOCKS=512

bash tests/kvbm_integration/scripts/run_server.sh <spec-id>
```

The script imports `_CACHE_RESET_SPECS` from the test module, looks up the spec by id, applies `KVBM_CPU_BLOCKS`/`KVBM_GPU_BLOCKS` overrides via `dataclasses.replace`, and launches `KvbmServerManager`. On READY it prints:

```
================================================================
[server] READY. Export these in shell 3 (run_eval.sh):
  export KVBM_EXTERNAL_BASE_URL=http://localhost:<port>
  export KVBM_EXTERNAL_METRICS_PORT=<port>
  export KVBM_SPEC_ID=<spec-id>
================================================================
[server] Ctrl-C to stop.
```

Copy the three `export` lines. The server stays foreground; Ctrl-C when done.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Clean shutdown after Ctrl-C |
| 2 | Usage error (missing spec id, bad prefix) |
| 4 | v1 spec without `NATS_SERVER`/`ETCD_ENDPOINTS` |
| 5 | `mgr.start_server()` returned False (check log) |
| 6 | Unknown spec id — script prints the known list |
| 7 | MLA spec without `KVBM_ENABLE_MLA=1` |

## Step 5: Shell 3 — eval

In a fresh shell (or the same shell as step 4 after backgrounding/Ctrl-Z the server):

```bash
# Paste the exports from step 4 (or --attach URL)
export KVBM_EXTERNAL_BASE_URL=http://localhost:<port>
export KVBM_EXTERNAL_METRICS_PORT=<port>
export KVBM_SPEC_ID=<spec-id>

# --fast knobs (if used)
export KVBM_MAX_ITERATIONS=2 KVBM_NUM_ITERATIONS=2 KVBM_REQUEST_DELAY=2

bash tests/kvbm_integration/scripts/run_eval.sh
```

`run_eval.sh` defaults to `test_determinism_agg_with_cache_reset` filtered by `-k $KVBM_SPEC_ID`. Positional args override entirely.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Test passed |
| 2 | Missing `KVBM_EXTERNAL_BASE_URL` / `KVBM_EXTERNAL_METRICS_PORT` / `KVBM_SPEC_ID` |
| other | Pytest failure — tail the per-test log |

## Step 6: Live Metrics Snapshot

While shell 2 is still running, fetch kvbm counters to confirm the offload/onboard path is active:

```bash
curl -s "http://localhost:$KVBM_EXTERNAL_METRICS_PORT/metrics" | grep '^kvbm_'
```

Expect after first cache reset:
- `kvbm_offload_blocks_d2h` > 0
- `kvbm_onboard_blocks_h2d` > 0

Phase 5 recorded (Qwen3-0.6B, default iterations):
- v1: host hit rate 66.4%
- v2 intra: 67.0%
- v2 inter: 66.7%

## Step 7: Report

Present:

```
KVBM Decomposed Run — <spec-id>
────────────────────────────────
deps:   <reused|spawned|skipped-v2|skipped-attach>
server: <OK (<N>s to READY)|FAILED>
eval:   <PASSED (<N>s)|FAILED>

Host hit rate: <N>% (baseline for this spec: <baseline>%)
Offload:       <N> blocks d2h
Onboard:       <N> blocks h2d
```

If eval failed:

```
Per-test vllm log:
  /tmp/dynamo_tests/<test-id>/ServerType.vllm_server_*.log

Diagnose: /dynamo:kvbm:diagnose --log <path>
```

## Troubleshooting

| Symptom | Check |
|---|---|
| `unknown spec id` (exit 6) | Verify spec list with the `.sandbox/bin/python` one-liner above. Qwen ids require `KVBM_MODEL_ID=Qwen/Qwen3-0.6B`. |
| Server never reaches READY | Tail the vllm log in `/tmp/dynamo_tests/`. Run `/dynamo:kvbm:diagnose`. |
| `cudaErrorNoKernelImageForDevice` in server log | sm_121 venv issue. Run `/dynamo:kvbm:sandbox-venv`. |
| Server hangs waiting for ZMQ handshake | v2 only: check `ConnectorLeader initialized with onboard mode onboard_mode=...` appears in the log. |
| Determinism test fails on v2 intra with garbage output | Historical bug: `#[cfg(feature = "nccl")]` incorrectly wrapping pure-CUDA code in `lib/kvbm-engine/src/worker/physical.rs` and `lib/kvbm-connector/src/connector/worker/mod.rs`. See ACTIVE_PLAN phase 5 deviations. Grep for the gates on the intra path. |
| `KVBM_SPEC_ID must be set` (exit 2 from eval) | Copy the export line printed by run_server.sh. |
| MLA spec rejected (exit 7) | Re-run with `--enable-mla`. |
| v2 spec rejected because NATS is set | `unset NATS_SERVER ETCD_ENDPOINTS` before launching v2 server. |

## Reference: The Three Scripts

| Script | Purpose | Exits |
|---|---|---|
| `tests/kvbm_integration/scripts/run_deps_v1.sh` | Reuse-or-spawn NATS + etcd. Mirrors conftest `runtime_services` fixture. | 0 on reuse, foreground on spawn |
| `tests/kvbm_integration/scripts/run_server.sh <spec-id>` | Look up `KvbmServerSpec` by id, launch `KvbmServerManager`, print exports | See exit-code table |
| `tests/kvbm_integration/scripts/run_eval.sh` | Run determinism eval against external server | See exit-code table |
