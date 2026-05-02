---
name: disagg-bringup
description: End-to-end bringup for KVBM conditional disaggregation — kill stale procs, start kvbm-hub with the CD prefill dispatcher enabled, launch one Prefill + one Decode vLLM (Qwen3-0.6B) on a single GPU, and verify both register on the hub. Self-contained — owns its launch scripts.
---

# Skill: KVBM Disagg Bringup

Reproduce the validated single-GPU two-instance disagg topology. After
this completes, `curl http://127.0.0.1:8337/v1/features/conditional-disagg/instances`
returns one entry under `prefill` and one under `decode`.

## Skill assets (in this directory)

| File | Purpose |
|---|---|
| `env.sh` | Source-able CUDA + venv exports for non-interactive bash. |
| `start-hub.sh` | Launches `kvbm_hub` with `--prefill-vllm-url` + `--prefill-vllm-model` baked in. |
| `launch-prefill.sh` | Single-instance vLLM prefill launcher (port 8000, role=prefill). Inherits `KVBM_DISAGG_LEADER` from caller's env. |
| `launch-decode.sh` | Single-instance vLLM decode launcher (port 8001, role=decode). Inherits `KVBM_DISAGG_LEADER` from caller's env. |
| `new-experiment.sh` | Mints `/tmp/kvbm-experiments/<ts>-<label>/` and prints the path. |

The `KVBM_DISAGG_LEADER` env var (recognized by `kvbm-connector::init`) selects between the legacy `ConditionalDisaggLeader` (default / unset / `legacy`) and the new `UnifiedDisaggLeader` (`unified`). Set it in the caller's environment before invoking `launch-{prefill,decode}.sh`; the scripts inherit it.

## Prerequisites

1. **Canonical venv at `/home/ryan/.venvs/dynamo-kvbm`** (symlink to a worktree's `.sandbox/`). If missing: `/kvbm-sandbox-venv`.
2. **Bindings built against the current branch.** If `kvbm._core.__version__` is older than the work-tree commit: `/kvbm-maturin-dev`.
3. **Kernels lib reachable.** Verify `/home/ryan/repos/dynamo/lib/bindings/kvbm/python/kvbm/libkvbm_kernels.so` exists. If not (maturin couldn't set rpath without `patchelf`), copy from `target/release/build/kvbm-kernels-*/out/libkvbm_kernels.so`. The launch scripts already export `LD_LIBRARY_PATH`.
4. **Hub binary built**: `cargo build -p kvbm-hub --bin kvbm_hub`.

## Workflow

### Step 0 — Sanity teardown

Use `/disagg-teardown` (or inline the same `pkill -f vllm.entrypoints / kvbm_hub` commands). Confirm `nvidia-smi --query-compute-apps` is empty.

### Step 1 — Mint a logs directory

```bash
SKILL=/home/ryan/repos/dynamo/.claude/skills/disagg-bringup
LOGS=$(bash "$SKILL/new-experiment.sh" disagg-bringup)
echo "logs in $LOGS"
```

### Step 2 — Start the hub

```bash
bash "$SKILL/start-hub.sh" "$LOGS/hub.log" &
HUB_PID=$!
sleep 2
curl -sS http://127.0.0.1:8337/v1/features/conditional-disagg/instances
# expect: {"prefill":[],"decode":[]}
```

### Step 3 — Launch Prefill + Decode

```bash
bash "$SKILL/launch-prefill.sh" > "$LOGS/prefill.log" 2>&1 &
bash "$SKILL/launch-decode.sh"  > "$LOGS/decode.log"  2>&1 &
# Optionally pin the leader: prepend  KVBM_DISAGG_LEADER=unified
#   KVBM_DISAGG_LEADER=unified bash "$SKILL/launch-prefill.sh" ...

for f in "$LOGS/prefill.log" "$LOGS/decode.log"; do
    until grep -qE "Application startup complete|Traceback|panicked|out of memory" "$f"; do
        sleep 5
    done
done
```

### Step 4 — Verify

```bash
curl -sS http://127.0.0.1:8337/v1/features/conditional-disagg/instances | jq

curl -sS http://127.0.0.1:8001/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"hello","max_tokens":8,"temperature":0}' | jq
```

Expect both UUIDs listed and a coherent completion.

## Why these specific values

| Knob | Value | Why |
|---|---|---|
| `--gpu-memory-utilization` | `0.15` | GB10 reports `Memory-Usage: Not Supported`; with two instances on GPU 0, 0.35 OOMs. 0.15 leaves headroom for both at `--max-num-seqs=8`. |
| `--max-model-len` | `1024` | Smaller KV reservation per slot; combined with `max-num-seqs=8`, two instances fit. |
| `DYN_KVBM_CPU_CACHE_GB` | `2` | Qwen3-0.6B is tiny; 2 GB of pinned host KV per instance is plenty for smoke. |
| `LD_LIBRARY_PATH=.../kvbm/` | required | `_core.abi3.so` has an unset rpath because maturin's patchelf was missing during build. |
| `kv_connector_module_path` | `kvbm.v2.vllm.schedulers.connector` | v2 (Rust scheduler) path. v1 path doesn't honor the `disagg` config block. |
| `--no-enable-prefix-caching` | required | Disables vLLM's own block prefix cache so the CD path's local-match (G2) lookup is the source of truth. |
| `--prefill-vllm-url` (hub) | `http://127.0.0.1:8000` | Tells the hub's CD prefill dispatcher where to POST queued prefill requests. Without this the hub queues them but never dispatches → decode hangs at lifecycle watchdog. |
| `--prefill-vllm-model` (hub) | `Qwen/Qwen3-0.6B` | Body field for dispatched POSTs; must match the prefill vLLM's `--model`. |

## See also

- `/disagg-teardown` — clean up the processes this skill starts
- `/disagg-hub-curl` — exercise the hub's HTTP surface after bringup
- `/disagg-smoke` — drive a full R1+R2 workload (legacy / unified / equiv modes)
- `/disagg-trace` — render `trace.html` from an experiment dir
- `/kvbm-maturin-dev` — rebuild bindings (run after pulling)
- `/kvbm-sandbox-venv` — provision the canonical venv if missing
- `/kvbm-diagnose` — triage failed bringups using known log patterns
- `/run-vllm-qwen-pd` — single-instance launch helper (P or D); this skill's `launch-{prefill,decode}.sh` are the canonical templates.
