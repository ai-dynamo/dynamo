---
name: disagg-bringup
description: End-to-end bringup for KVBM conditional disaggregation — kill stale procs, start kvbm-hub with the CD prefill dispatcher enabled, launch one Prefill + one Decode vLLM, and verify both register on the hub. Hardware/model sizing is caller-controlled; the old Spark/Qwen path is a profile, not the workflow definition.
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
| `launch-prefill.sh` | Single-instance vLLM prefill launcher (port 8000, role=prefill). Honors `KVBM_BLOCK_LAYOUT`. |
| `launch-decode.sh` | Single-instance vLLM decode launcher (port 8001, role=decode). Honors `KVBM_BLOCK_LAYOUT`. |
| `verify-block-layout.sh` | Post-bringup assertion: confirms a registered instance's G2 layout matches the requested mode (via hub describe). |
| `new-experiment.sh` | Mints `/tmp/kvbm-experiments/<ts>-<label>/` and prints the path. |

### Fresh-artifact guarantees (read before debugging a "registration failed" hang)

Two stale-artifact traps have bitten this bringup; both are now guarded, do not regress them:

1. **Stale hub binary.** `start-hub.sh` runs `cargo build --bin kvbm_hub` (incremental — a no-op when fresh) *before* launching, so it can never run an out-of-date binary. A `test -x $HUB` existence check does **not** catch staleness. The 2026-05-19 incident: a May-15 debug `kvbm_hub` ran against May-19 hub source; the fresh connector's CD registration handshake didn't match the old hub → `Exception: conditional-disagg hub registration failed` in `leader.initialize_workers()` → EngineCore died → the smoke hung on its readiness loop. If you ever pin `KVBM_HUB_BIN` or set `KVBM_HUB_SKIP_BUILD=1`, you own binary freshness yourself. Cheapest post-run check: `ls -l --time-style=+%F target/debug/kvbm_hub` should show *today*.
2. **Wrong/foreign venv.** `launch-{prefill,decode}.sh` honor `KVBM_VENV`
   and default to `<worktree>/.sandbox`. Keep that default or export an
   explicit current venv so you exercise this worktree's kvbm build.

Bringup waits are bounded: `two-request-smoke.sh` waits on the hub `/health` (default 300 s, covers a cold rebuild) before launching vLLMs, then on each vLLM `/v1/models` (default 240 s) with per-process death detection — a dead EngineCore aborts in seconds with a log tail instead of hanging. Tune via `KVBM_HUB_READY_TIMEOUT` / `KVBM_VLLM_READY_TIMEOUT`.

The `KVBM_DISAGG_LEADER` env var (recognized by `kvbm-connector::init`) selects between the legacy `ConditionalDisaggLeader` (default / unset / `legacy`) and the new `UnifiedDisaggLeader` (`unified`). Set it in the caller's environment before invoking `launch-{prefill,decode}.sh`; the scripts inherit it.

### Universal-mode launch

`KVBM_BLOCK_LAYOUT=universal` activates Universal G2 layout (fused permute kernels, cross-TP/PP canonical equality). **Do not rely on the env var reaching vLLM's EngineCore subprocess** — vLLM spawns EngineCore in a new process that does not inherit the parent environment. The connector reads `KvbmConfig` in the EngineCore process, not in `api_server`, so the env var is absent when `kvbm-config`'s figment merge runs.

The fix: inject `block_layout` into the `--kv-transfer-config` JSON under `kv_connector_extra_config.default.block_layout`. vLLM serializes the config object and re-loads it on the worker side, so JSON injection survives the spawn boundary.

```bash
KVBM_BLOCK_LAYOUT=universal bash launch-prefill.sh > prefill.log 2>&1 &
KVBM_BLOCK_LAYOUT=universal bash launch-decode.sh  > decode.log  2>&1 &
```

The launch scripts inject the value into `"default": { "block_layout": "..." }` inside `kv_connector_extra_config`. Because `KvbmRuntime.build_leader/build_worker` calls `from_figment_with_json_for_leader/worker` with `.nested()`, the `default` profile applies to both leader and worker roles.

After bringup, `verify-block-layout.sh` confirms the connector actually registered in the requested mode:

```bash
bash verify-block-layout.sh "$PREFILL_ID" universal
bash verify-block-layout.sh "$DECODE_ID"  universal
```

This is the reproducer for the env-var stripping bug: the assertion fails when only env-var propagation is used and passes after JSON injection.

## Prerequisites

1. **A current Dynamo/KVBM Python environment.** In experiment containers this
   is usually `/opt/dynamo/venv`; for local sandbox work use
   `/kvbm-sandbox-venv`.
2. **Bindings built against the current branch.** If `kvbm._core.__version__`
   is older than the work-tree commit: `/kvbm-maturin-dev`.
3. **Kernels lib reachable.** The launch scripts export `LD_LIBRARY_PATH`;
   verify `lib/bindings/kvbm/python/kvbm/libkvbm_kernels.so` exists if imports
   fail.
4. **Hub binary built**: `cargo build -p kvbm-hub --bin kvbm_hub`. The hub
   launcher rebuilds incrementally by default.

## Hardware Profiles

The workflow is the same across machines; only sizing changes. Profile defaults
live in `hardware-profiles.sh`; launcher scripts source that file and consume
env vars rather than embedding model, GPU, cache, or max-length assumptions.

| Profile | Model | Placement | Memory sizing |
|---|---|---|---|
| `h100-a100` | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | one process per GPU | higher GMU, longer context |
| `spark-gb10` | `Qwen/Qwen3-0.6B` | two processes can share GPU 0 | low GMU, short context |
| `custom` | caller-set | caller-set | caller-set |

## Workflow

### Step 0 — Sanity teardown

Use `/disagg-teardown` (or inline the same `pkill -f vllm.entrypoints / kvbm_hub` commands). Confirm `nvidia-smi --query-compute-apps` is empty.

### Step 1 — Mint a logs directory

```bash
SKILL=$PWD/.claude/skills/disagg-bringup
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
# Defaults to spark-gb10. Set KVBM_HARDWARE_PROFILE=h100-a100 for the
# DeepSeek/H100-or-A100 sizing profile, or custom with explicit env overrides.
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

MODEL=${MODEL:-Qwen/Qwen3-0.6B}
curl -sS http://127.0.0.1:8001/v1/completions \
  -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"hello\",\"max_tokens\":8,\"temperature\":0}" | jq
```

Expect both UUIDs listed and a coherent completion.

## Why these specific values

| Knob | Value | Why |
|---|---|---|
| `--gpu-memory-utilization` | profile-set | Use low values for shared-GPU Spark/GB10; use normal H100/A100 values when each instance owns a GPU. |
| `--max-model-len` | profile-set | Smaller values reduce KV reservation and make local smoke profiles fit. |
| `DYN_KVBM_CPU_CACHE_GB` | profile-set | Size to model and workload; small values are enough for Qwen smoke, larger values for DeepSeek 8B experiments. |
| `LD_LIBRARY_PATH=.../kvbm/` | required | `_core.abi3.so` has an unset rpath because maturin's patchelf was missing during build. |
| `kv_connector_module_path` | `kvbm.v2.vllm.connector` | Canonical v2 lazy facade; avoids binding callers to the scheduler module layout. |
| `--no-enable-prefix-caching` | required | Disables vLLM's own block prefix cache so the CD path's local-match (G2) lookup is the source of truth. |
| `--prefill-vllm-url` (hub) | `http://127.0.0.1:8000` | Tells the hub's CD prefill dispatcher where to POST queued prefill requests. Without this the hub queues them but never dispatches → decode hangs at lifecycle watchdog. |
| `--prefill-vllm-model` (hub) | caller-set | Body field for dispatched POSTs; must match the prefill vLLM's `--model`. |

## See also

- `/disagg-teardown` — clean up the processes this skill starts
- `/disagg-hub-curl` — exercise the hub's HTTP surface after bringup
- `/disagg-smoke` — drive a full R1+R2 workload (legacy / unified / equiv modes)
- `/disagg-trace` — render `trace.html` from an experiment dir
- `/kvbm-maturin-dev` — rebuild bindings (run after pulling)
- `/kvbm-sandbox-venv` — provision the canonical venv if missing
- `/kvbm-diagnose` — triage failed bringups using known log patterns
- `/run-vllm-qwen-pd` — single-instance launch helper (P or D); this skill's `launch-{prefill,decode}.sh` are the canonical templates.
