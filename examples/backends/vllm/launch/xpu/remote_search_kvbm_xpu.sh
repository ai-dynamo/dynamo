#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${KVBM_REPO:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"
SMOKE_DIR="$REPO/.claude/skills/remote-search-smoke"

HUB_DISC="${KVBM_HUB_DISCOVERY_PORT:-1337}"
HUB_CTRL="${KVBM_HUB_CONTROL_PORT:-8337}"
HUB_URL="${KVBM_HUB_URL:-http://127.0.0.1:$HUB_DISC}"

ROOT="${1:-${KVBM_EXPERIMENTS_DIR:-$REPO}/$(date +%Y%m%d-%H%M%S)-xpu-onboarding-smoke}"
mkdir -p "$ROOT"

MODEL="${KVBM_MODEL:-Qwen/Qwen3-0.6B}"

resolve_local_model() {
  # Prefer an already-cached local snapshot for offline test environments.
  local model="$1"
  if [[ -d "$model" ]]; then
    printf '%s\n' "$model"
    return 0
  fi

  if [[ "$model" == "Qwen/Qwen3-0.6B" ]]; then
    local cache_root
    cache_root="/workspace/components/backends/vllm/.cache/huggingface/models--Qwen--Qwen3-0.6B/snapshots"
    if [[ -d "$cache_root" ]]; then
      local snap
      snap=$(ls -1dt "$cache_root"/* 2>/dev/null | head -n 1 || true)
      if [[ -n "$snap" && -f "$snap/config.json" ]]; then
        printf '%s\n' "$snap"
        return 0
      fi
    fi
  fi

  printf '%s\n' "$model"
}

MODEL="$(resolve_local_model "$MODEL" | tr -d '\r\n')"
PROMPT_DEFAULT=$(cat <<'PROMPT_EOF'
The quick brown fox jumps over the lazy dog and then keeps on running through the meadow until it reaches the river where it finally stops to drink some water and rest for a while before continuing its journey home through the forest path under a bright sky. You are a strict extraction engine. Read the context and output JSON with keys topic, entities, risks, actions.\nContext:\nThe Saturn mission review board met on 2034-02-17. Flight software build FS-77 introduced adaptive guidance smoothing. During thermal-vacuum cycle TV-12, valve cluster VC-9 showed intermittent response at 0.3 Hz under low-pressure helium purge. Engineering note EN-442 recommends replacing actuator lot L-118 before integrated systems test. Power budget margin was reported as 7.4 percent in the nominal cruise profile and 2.1 percent in safe-hold. Communications relay window dropped by 14 minutes after antenna gimbal friction increase was detected.\nProduce concise JSON only.\nSuffix: classify for Operations Team Alpha.
PROMPT_EOF
)
PROMPT="${KVBM_PROMPT:-$PROMPT_DEFAULT}"
MAX_TOKENS="${KVBM_MAX_TOKENS:-8}"
MAX_MODEL_LEN="${KVBM_MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${KVBM_MAX_NUM_SEQS:-2}"
GMU="${KVBM_GPU_MEMORY_UTILIZATION:-0.15}"
CPU_CACHE_GB="${KVBM_CPU_CACHE_GB:-2}"
BLOCK_SIZE="${KVBM_BLOCK_SIZE:-64}"
# Prefer UCX transport by default for XPU smoke tests (no RDMA).
NIXL_BACKEND_POSIX="${KVBM_NIXL_BACKEND_POSIX:-false}"
NIXL_BACKEND_UCX="${KVBM_NIXL_BACKEND_UCX:-true}"
UCX_TLS_NO_RDMA="${KVBM_UCX_TLS:-tcp,self}"
UCX_NET_DEVICES_NO_RDMA="${KVBM_UCX_NET_DEVICES:-all}"
UCX_SOCKADDR_TLS_NO_RDMA="${KVBM_UCX_SOCKADDR_TLS_PRIORITY:-tcp}"
UCX_MEMTYPE_CACHE_NO_RDMA="${KVBM_UCX_MEMTYPE_CACHE:-n}"
KV_BUFFER_DEVICE="${KVBM_KV_BUFFER_DEVICE:-cpu}"
KV_BUFFER_SIZE="${KVBM_KV_BUFFER_SIZE:-134217728}"
A_MASK="${KVBM_XPU_AFFINITY_MASK_A:-0}"
B_MASK="${KVBM_XPU_AFFINITY_MASK_B:-1}"
A_PORT="${KVBM_A_PORT:-8000}"
B_PORT="${KVBM_B_PORT:-8001}"
A_SYS_PORT="${KVBM_A_SYS_PORT:-8081}"
B_SYS_PORT="${KVBM_B_SYS_PORT:-8082}"
TIMEOUT_SECS="${KVBM_WAIT_TIMEOUT_SECS:-300}"

HUB_LOG="$ROOT/hub.log"
A_LOG="$ROOT/instance_a.log"
B_LOG="$ROOT/instance_b.log"

cleanup() {
  pkill -9 -f kvbm_hub 2>/dev/null || true
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
}

fail() {
  echo "ERROR: $1" >&2
  if [[ -n "${2:-}" && -f "$2" ]]; then
    echo "--- $2 ---" >&2
    tail -n 120 "$2" >&2 || true
  fi
  exit 1
}

wait_http() {
  local url="$1"
  local deadline=$(( $(date +%s) + TIMEOUT_SECS ))
  until curl -fsS -m 5 "$url" >/dev/null 2>&1; do
    if [[ $(date +%s) -ge $deadline ]]; then
      fail "timeout waiting for $url"
    fi
    sleep 2
  done
}

json_body() {
  MODEL="$MODEL" PROMPT="$PROMPT" MAX_TOKENS="$MAX_TOKENS" \
    python3 -c 'import json,os; print(json.dumps({"model": os.environ["MODEL"], "prompt": os.environ["PROMPT"], "max_tokens": int(os.environ["MAX_TOKENS"]), "temperature": 0}))'
}

extract_text() {
  python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])'
}

request_completion() {
  local port="$1"
  json_body | curl -fsS -m 120 -X POST "http://127.0.0.1:$port/v1/completions" \
    -H 'Content-Type: application/json' \
    -d @- | extract_text
}

start_instance() {
  local port="$1"
  local affinity_mask="$2"
  local sys_port="$3"
  local log_file="$4"

  (
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export RUST_LOG="${RUST_LOG:-info,kvbm_connector=debug,kvbm_engine=debug,kvbm_audit=info}"
    export KVBM_DECODE_OFFLOAD="${KVBM_DECODE_OFFLOAD:-true}"
    export VLLM_TARGET_DEVICE=xpu
    export ZE_AFFINITY_MASK="$affinity_mask"
    export DYN_SYSTEM_PORT="$sys_port"
    export DYN_KVBM_CPU_CACHE_GB="$CPU_CACHE_GB"
    export VLLM_ATTENTION_BACKEND=FLASH_ATTN
    export KVBM_SKIP_VLLM_VERSION_CHECK=1
    export KVBM_MODEL="$MODEL"
    export KVBM_MAX_NUM_SEQS="$MAX_NUM_SEQS"
    export KVBM_GPU_MEMORY_UTILIZATION="$GMU"
    export KVBM_CPU_CACHE_GB="$CPU_CACHE_GB"
    export DYN_KVBM_NIXL_BACKEND_POSIX="$NIXL_BACKEND_POSIX"
    export DYN_KVBM_NIXL_BACKEND_UCX="$NIXL_BACKEND_UCX"

    # Force UCX to non-RDMA transports for bring-up environments.
    export UCX_TLS="$UCX_TLS_NO_RDMA"
    export UCX_NET_DEVICES="$UCX_NET_DEVICES_NO_RDMA"
    export UCX_SOCKADDR_TLS_PRIORITY="$UCX_SOCKADDR_TLS_NO_RDMA"
    export UCX_MEMTYPE_CACHE="$UCX_MEMTYPE_CACHE_NO_RDMA"

    # Capture runtime limits that commonly cause UCX shared-memory failures.
    echo "[startup] UCX_TLS=$UCX_TLS UCX_NET_DEVICES=$UCX_NET_DEVICES UCX_SOCKADDR_TLS_PRIORITY=$UCX_SOCKADDR_TLS_PRIORITY UCX_MEMTYPE_CACHE=$UCX_MEMTYPE_CACHE"
    echo "[startup] /dev/shm usage:"
    df -h /dev/shm || true
    echo "[startup] memlock_kb=$(ulimit -l 2>/dev/null || echo unknown)"

    KVBMCTL="${KVBM_KVBMCTL_BIN:-$REPO/target/debug/kvbmctl}"
    KVBM_VENV="${KVBM_VENV:-$REPO/.sandbox}"
    if [[ ! -x "$KVBM_VENV/bin/python3" && -x /opt/dynamo/venv/bin/python3 ]]; then
      KVBM_VENV="/opt/dynamo/venv"
    fi
    PY_BIN="$KVBM_VENV/bin/python3"
    [[ -x "$PY_BIN" ]] || fail "python3 not found at $PY_BIN (set KVBM_VENV)" "$log_file"
    source "$REPO/.claude/skills/kvbm-hub-bringup/hub-lib.sh"

    KV_RENDERED=$(kvbm_hub_render_vllm "$KVBMCTL" "$HUB_URL" indexer,p2p \
      --kv-connector-module-path kvbm.v2.vllm.connector) \
      || fail "kvbmctl render failed (is the hub up at $HUB_URL?)" "$HUB_LOG"
    eval "KV_ARGS=( $KV_RENDERED )"

    # Keep transfer staging on host memory for XPU bring-up unless overridden.
    for i in "${!KV_ARGS[@]}"; do
      if [[ "${KV_ARGS[$i]}" == "--kv-transfer-config" ]]; then
        j=$((i + 1))
        if [[ $j -lt ${#KV_ARGS[@]} ]]; then
          KV_ARGS[$j]=$(KV_JSON="${KV_ARGS[$j]}" KV_BUFFER_DEVICE="$KV_BUFFER_DEVICE" KV_BUFFER_SIZE="$KV_BUFFER_SIZE" "$PY_BIN" - <<'PYEOF'
import json
import os

cfg = json.loads(os.environ["KV_JSON"])
cfg["kv_buffer_device"] = os.environ["KV_BUFFER_DEVICE"]
cfg["kv_buffer_size"] = int(os.environ["KV_BUFFER_SIZE"])
print(json.dumps(cfg, separators=(",", ":")))
PYEOF
)
        fi
        break
      fi
    done

    NIXL_LIBS="${KVBM_NIXL_LIBS:-}"
    if [[ -z "$NIXL_LIBS" ]]; then
      for cand in "$KVBM_VENV"/lib/python*/site-packages/.nixl_cu12.mesonpy.libs \
                  "$KVBM_VENV"/lib/python*/site-packages/.nixl_cu13.mesonpy.libs; do
        [[ -d "$cand" ]] && NIXL_LIBS="$cand" && break
      done
    fi
    if [[ -n "$NIXL_LIBS" ]]; then
      export LD_LIBRARY_PATH="$NIXL_LIBS:$NIXL_LIBS/plugins:${LD_LIBRARY_PATH:-}"
      export NIXL_PLUGIN_DIR="$NIXL_LIBS/plugins"
    fi

    exec "$PY_BIN" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --served-model-name "$MODEL" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GMU" \
      --enable-chunked-prefill \
      --enforce-eager \
      --port "$port" \
      "${KV_ARGS[@]}" \
      --block-size "$BLOCK_SIZE"
  ) >"$log_file" 2>&1 &
}

main() {
  trap cleanup EXIT

  export KVBM_CPU_CACHE_GB="$CPU_CACHE_GB"
  export KVBM_MAX_MODEL_LEN="$MAX_MODEL_LEN"
  export KVBM_MODEL="$MODEL"
  # Keep hub primary config aligned with vLLM runtime block size for handshake.
  export KVBM_HUB_BLOCK_SIZE="${KVBM_HUB_BLOCK_SIZE:-$BLOCK_SIZE}"
  # Force onboarding smoke to publish freshly computed blocks to G2 immediately.
  export KVBM_HUB_KVBM="${KVBM_HUB_KVBM:-$(cat <<'EOF'
leader.tokio.worker_threads=2
worker.tokio.worker_threads=2
leader.control.metrics=true
default.metrics.cache_stats_log_interval_secs=5
leader.remote_search.enabled=true
leader.audit.enabled=true
leader.onboard.mode=intra
worker.onboard.mode=intra
offload.g1_to_g2.policies=["pass_all"]
worker.nixl.backends.UCX={}
EOF
)}"
  # This smoke path uses indexer,p2p only; prefill-router requires disagg.
  export KVBM_HUB_PREFILL_ROUTER="${KVBM_HUB_PREFILL_ROUTER:-0}"
  # Keep local control-plane probes on loopback, not corporate HTTP proxies.
  export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
  export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost,::1"


  pkill -9 -f kvbm_hub 2>/dev/null || true
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true

  echo "Logs: $ROOT"
  echo "Model: $MODEL"
  echo "Affinity masks: A=$A_MASK B=$B_MASK"

  bash "$SMOKE_DIR/start-hub.sh" "$HUB_LOG" &
  wait_http "http://127.0.0.1:$HUB_CTRL/health"

  start_instance "$A_PORT" "$A_MASK" "$A_SYS_PORT" "$A_LOG"
  wait_http "http://127.0.0.1:$A_PORT/v1/models"

  start_instance "$B_PORT" "$B_MASK" "$B_SYS_PORT" "$B_LOG"
  wait_http "http://127.0.0.1:$B_PORT/v1/models"

  echo "A1: warm A from cold"
  A1_TEXT=$(request_completion "$A_PORT")
  echo "A1 text: $A1_TEXT"

  echo "A2: prove A warm-local skip"
  A2_TEXT=$(request_completion "$A_PORT")
  echo "A2 text: $A2_TEXT"

  echo "B1: cold local, warm in G2, should onboard"
  B_TEXT=$(request_completion "$B_PORT")
  echo "B text: $B_TEXT"

  audit_deadline=$(( $(date +%s) + TIMEOUT_SECS ))
  while ! grep -qaE 'kvbm_audit.*event="request_finished"' "$B_LOG" 2>/dev/null; do
    if [[ $(date +%s) -ge $audit_deadline ]]; then
      fail "timeout waiting for kvbm audit request_finished in $B_LOG" "$B_LOG"
    fi
    sleep 1
  done

  A_OFFLOAD=$(grep -acE 'kvbm_audit.*event="offload_register_complete".*dst="kvbm_engine::G2"' "$A_LOG" 2>/dev/null || true)
  A_OPEN=$(grep -acE 'kvbm_audit.*event="transfer_session_opened"' "$A_LOG" 2>/dev/null || true)
  A_SELF_PULL=$(grep -acE 'kvbm_audit.*event="transfer_pull_started"' "$A_LOG" 2>/dev/null || true)
  B_PENDING=$(grep -acE 'kvbm_audit.*event="gnmt_pending"' "$B_LOG" 2>/dev/null || true)
  B_MATCHED=$(grep -acE 'kvbm_audit.*event="gnmt_matched".*matched_tokens=[1-9]' "$B_LOG" 2>/dev/null || true)
  B_PULL_START=$(grep -acE 'kvbm_audit.*event="transfer_pull_started"' "$B_LOG" 2>/dev/null || true)
  B_PULL_DONE=$(grep -acE 'kvbm_audit.*event="transfer_pull_completed"' "$B_LOG" 2>/dev/null || true)
  # Onboarding completion audit names differ by code path:
  # - leader/onboard.rs emits event="onboard_complete"
  # - disagg/decode_leader.rs emits event="mark_onboarding_complete"
  # Keep both audit variants plus a physical-layer completion fallback to avoid false negatives.
  B_ONBOARD_AUDIT=$(grep -acE 'kvbm_audit.*event="onboard_complete"' "$B_LOG" 2>/dev/null || true)
  B_ONBOARD_MARK=$(grep -acE 'kvbm_audit.*event="mark_onboarding_complete"' "$B_LOG" 2>/dev/null || true)
  B_ONBOARD_LW=$(grep -acE 'kvbm_engine::worker::physical: Layer-wise onboard complete' "$B_LOG" 2>/dev/null || true)
  B_ONBOARD=$((B_ONBOARD_AUDIT + B_ONBOARD_MARK + B_ONBOARD_LW))
  B_FIN=$(grep -acE 'kvbm_audit.*event="request_finished"' "$B_LOG" 2>/dev/null || true)

  printf '\nSummary\n'
  printf '  A offload_register_complete=%s\n' "$A_OFFLOAD"
  printf '  A transfer_session_opened=%s transfer_pull_started=%s\n' "$A_OPEN" "$A_SELF_PULL"
  printf '  B gnmt_pending=%s gnmt_matched=%s\n' "$B_PENDING" "$B_MATCHED"
  printf '  B transfer_pull_started=%s transfer_pull_completed=%s\n' "$B_PULL_START" "$B_PULL_DONE"
  printf '  B onboard_complete(audit)=%s mark_onboarding_complete(audit)=%s onboard_complete(layerwise)=%s request_finished=%s\n' "$B_ONBOARD_AUDIT" "$B_ONBOARD_MARK" "$B_ONBOARD_LW" "$B_FIN"

  [[ "$A_OFFLOAD" -ge 1 ]] || fail "A never registered an offload into G2" "$A_LOG"
  [[ "$A_OPEN" -ge 1 ]] || fail "A never opened a transfer session for B" "$A_LOG"
  [[ "$A_SELF_PULL" -eq 0 ]] || fail "A unexpectedly initiated a pull; it should stay the holder" "$A_LOG"
  [[ "$B_PENDING" -ge 1 ]] || fail "B never entered remote-search pending state" "$B_LOG"
  [[ "$B_PULL_START" -ge 1 ]] || fail "B never started pulling blocks from A" "$B_LOG"
  [[ "$B_PULL_DONE" -ge 1 ]] || fail "B never completed the pull" "$B_LOG"
  [[ "$B_MATCHED" -ge 1 ]] || fail "B never resolved to an external match" "$B_LOG"
  [[ "$B_ONBOARD" -ge 1 ]] || fail "B never completed onboarding (no audit onboard_complete, no audit mark_onboarding_complete, and no layer-wise onboard completion log)" "$B_LOG"
  [[ "$B_FIN" -ge 1 ]] || fail "B request never finished" "$B_LOG"

  if [[ "$A2_TEXT" != "$B_TEXT" ]]; then
    fail "A warm completion and B completion differ; remote pull did not reproduce the same decode" "$B_LOG"
  fi

  echo
  echo "PASS: XPU onboarding test completed"
  echo "Logs: $ROOT"
}

main "$@"
