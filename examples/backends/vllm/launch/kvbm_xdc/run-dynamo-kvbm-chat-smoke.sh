#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/hardware-profiles.sh"
. "$SCRIPT_DIR/common.sh"

ARTIFACT_DIR=${ARTIFACT_DIR:?ARTIFACT_DIR is required}
NODE_ROLE=${NODE_ROLE:-all}
case "$NODE_ROLE" in
  frontend|hub|hub-prefill-frontend|client|trace) profile_scope=model-only ;;
  decode|prefill) profile_scope=worker ;;
  all) profile_scope=all ;;
  *) profile_scope=all ;;
esac
kvbm_xdc_apply_hardware_profile "$profile_scope"

WORKTREE=${WORKTREE:-/workspace}
NAMESPACE=${NAMESPACE:-dynamo}
HTTP_PORT=${HTTP_PORT:-8000}
FRONTEND_HOST=${FRONTEND_HOST:-127.0.0.1}
READY_TIMEOUT=${READY_TIMEOUT:-900}
TRANSFER_EVIDENCE_TIMEOUT=${TRANSFER_EVIDENCE_TIMEOUT:-30}
VENV=${VENV:-}
START_LOCAL_INFRA=${START_LOCAL_INFRA:-1}
INFRA_HOST=${INFRA_HOST:-127.0.0.1}
PREFETCH_MODEL=${PREFETCH_MODEL:-1}
ISOLATE_HF_CACHE=${ISOLATE_HF_CACHE:-0}
DECODE_SIDE_CHANNEL_PORT=${DECODE_SIDE_CHANNEL_PORT:-20096}
PREFILL_SIDE_CHANNEL_PORT=${PREFILL_SIDE_CHANNEL_PORT:-20097}
PREFILL_KV_EVENTS_PORT=${PREFILL_KV_EVENTS_PORT:-20081}
ENFORCE_EAGER=${ENFORCE_EAGER:-1}
VLLM_RUNNER=${VLLM_RUNNER:-generate}
KVBM_CONNECTOR_MODULE_PATH=${KVBM_CONNECTOR_MODULE_PATH:-kvbm.v2.vllm.connector}
PREFILL_KV_CONNECTOR_WRAPPER=${PREFILL_KV_CONNECTOR_WRAPPER:-PdConnector}
SOURCE_KVBM_RUNTIME_ENV=${SOURCE_KVBM_RUNTIME_ENV:-0}
ENABLE_PERMUTE_LOCAL_KV=${ENABLE_PERMUTE_LOCAL_KV:-1}
VLLM_KV_CACHE_LAYOUT=${VLLM_KV_CACHE_LAYOUT:-HND}
RUST_LOG=${RUST_LOG:-info,kvbm_connector=debug,kvbm_audit=info}
DYN_LOG=${DYN_LOG:-$RUST_LOG}
KVBM_PLAIN_AUDIT=${KVBM_PLAIN_AUDIT:-1}
KVBM_EXPERIMENT_ID=${KVBM_EXPERIMENT_ID:-diagnostic}
KVBM_TRANSFER_TOPOLOGY=${KVBM_TRANSFER_TOPOLOGY:-nixl-pd}
KVBM_HUB_URL=${KVBM_HUB_URL:-http://127.0.0.1:1337}
KVBM_HUB_PREFILL_NAMESPACE=${KVBM_HUB_PREFILL_NAMESPACE:-${NAMESPACE}_hub_prefill}
KVBM_HUB_PREFILL_HTTP_PORT=${KVBM_HUB_PREFILL_HTTP_PORT:-8001}
KVBM_HUB_PREFILL_HOST=${KVBM_HUB_PREFILL_HOST:-$FRONTEND_HOST}
KVBM_BLOCK_LAYOUT=${KVBM_BLOCK_LAYOUT:-operational}
KVBM_ONBOARD_MODE=${KVBM_ONBOARD_MODE:-inter}
KVBM_WORKER_NIXL_BACKENDS=${KVBM_WORKER_NIXL_BACKENDS:-}
KVBM_RECORD_NETWORK_ENDPOINTS=${KVBM_RECORD_NETWORK_ENDPOINTS:-0}
KVBM_FORCE_UCX_LOCAL_FALLBACK=${KVBM_FORCE_UCX_LOCAL_FALLBACK:-auto}
KVBM_TRANSFER_SUBSTRATE_PREFLIGHT=${KVBM_TRANSFER_SUBSTRATE_PREFLIGHT:-auto}
KVBM_ALLOW_SUBSTRATE_FAILURE_ARTIFACT=${KVBM_ALLOW_SUBSTRATE_FAILURE_ARTIFACT:-0}
KVBM_SERVE_ONLY=${KVBM_SERVE_ONLY:-0}
KVBM_CLEANUP_STALE_PROCESSES=${KVBM_CLEANUP_STALE_PROCESSES:-1}
PYTHON_BIN=${PYTHON_BIN:-}

: "${MODEL:?MODEL must be set by KVBM_HARDWARE_PROFILE or env override}"
MODEL_PATH=${MODEL_PATH:-$MODEL}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$MODEL}
case "$KVBM_TRANSFER_TOPOLOGY" in
  nixl-pd|kvbm-hub) ;;
  *) echo "KVBM_TRANSFER_TOPOLOGY must be nixl-pd or kvbm-hub, got $KVBM_TRANSFER_TOPOLOGY" >&2; exit 2 ;;
esac
if [[ -z "${KVBM_HUB_PREFILL_URL+x}" ]]; then
  if [[ "$KVBM_TRANSFER_TOPOLOGY" == "kvbm-hub" ]]; then
    KVBM_HUB_PREFILL_URL="http://$KVBM_HUB_PREFILL_HOST:$KVBM_HUB_PREFILL_HTTP_PORT"
  else
    KVBM_HUB_PREFILL_URL="http://$FRONTEND_HOST:$HTTP_PORT"
  fi
fi
if [[ -z "${FRONTEND_ENFORCE_DISAGG+x}" ]]; then
  if [[ "$KVBM_TRANSFER_TOPOLOGY" == "kvbm-hub" ]]; then
    FRONTEND_ENFORCE_DISAGG=0
  else
    FRONTEND_ENFORCE_DISAGG=1
  fi
fi
case "$KVBM_BLOCK_LAYOUT" in
  operational|universal) ;;
  *) echo "KVBM_BLOCK_LAYOUT must be operational or universal, got $KVBM_BLOCK_LAYOUT" >&2; exit 2 ;;
esac
case "$KVBM_ONBOARD_MODE" in
  inter|intra) ;;
  *) echo "KVBM_ONBOARD_MODE must be inter or intra, got $KVBM_ONBOARD_MODE" >&2; exit 2 ;;
esac
case "$KVBM_TRANSFER_SUBSTRATE_PREFLIGHT" in
  auto|strict|warn|0|false|off) ;;
  *) echo "KVBM_TRANSFER_SUBSTRATE_PREFLIGHT must be auto, strict, warn, or 0, got $KVBM_TRANSFER_SUBSTRATE_PREFLIGHT" >&2; exit 2 ;;
esac
case "$KVBM_CLEANUP_STALE_PROCESSES" in
  1|true|yes|on) KVBM_CLEANUP_STALE_PROCESSES=1 ;;
  0|false|no|off) KVBM_CLEANUP_STALE_PROCESSES=0 ;;
  *) echo "KVBM_CLEANUP_STALE_PROCESSES must be boolean, got $KVBM_CLEANUP_STALE_PROCESSES" >&2; exit 2 ;;
esac
case "$NODE_ROLE" in
  all)
    : "${GPU_MEMORY_UTILIZATION:?GPU_MEMORY_UTILIZATION must be set by KVBM_HARDWARE_PROFILE or env override}"
    : "${PREFILL_GPU_MEMORY_UTILIZATION:?PREFILL_GPU_MEMORY_UTILIZATION must be set by KVBM_HARDWARE_PROFILE or env override}"
    : "${DECODE_GPU_MEMORY_UTILIZATION:?DECODE_GPU_MEMORY_UTILIZATION must be set by KVBM_HARDWARE_PROFILE or env override}"
    : "${MAX_MODEL_LEN:?MAX_MODEL_LEN must be set by KVBM_HARDWARE_PROFILE or env override}"
    : "${MAX_NUM_SEQS:?MAX_NUM_SEQS must be set by KVBM_HARDWARE_PROFILE or env override}"
    : "${CPU_CACHE_GB:?CPU_CACHE_GB must be set by KVBM_HARDWARE_PROFILE or env override}"
    : "${DECODE_CUDA_VISIBLE_DEVICES:?DECODE_CUDA_VISIBLE_DEVICES must be set by KVBM_HARDWARE_PROFILE or env override}"
    : "${PREFILL_CUDA_VISIBLE_DEVICES:?PREFILL_CUDA_VISIBLE_DEVICES must be set by KVBM_HARDWARE_PROFILE or env override}"
    ;;
  decode)
    : "${GPU_MEMORY_UTILIZATION:?GPU_MEMORY_UTILIZATION must be set by KVBM_HARDWARE_PROFILE or env override}"
    : "${DECODE_GPU_MEMORY_UTILIZATION:?DECODE_GPU_MEMORY_UTILIZATION must be set by KVBM_HARDWARE_PROFILE or env override}"
    : "${MAX_MODEL_LEN:?MAX_MODEL_LEN must be set by KVBM_HARDWARE_PROFILE or env override}"
    : "${MAX_NUM_SEQS:?MAX_NUM_SEQS must be set by KVBM_HARDWARE_PROFILE or env override}"
    : "${CPU_CACHE_GB:?CPU_CACHE_GB must be set by KVBM_HARDWARE_PROFILE or env override}"
    DECODE_CUDA_VISIBLE_DEVICES=${DECODE_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}
    : "${DECODE_CUDA_VISIBLE_DEVICES:?DECODE_CUDA_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES is required for NODE_ROLE=decode}"
    ;;
  prefill)
    : "${GPU_MEMORY_UTILIZATION:?GPU_MEMORY_UTILIZATION must be set by KVBM_HARDWARE_PROFILE or env override}"
    : "${PREFILL_GPU_MEMORY_UTILIZATION:?PREFILL_GPU_MEMORY_UTILIZATION must be set by KVBM_HARDWARE_PROFILE or env override}"
    : "${MAX_MODEL_LEN:?MAX_MODEL_LEN must be set by KVBM_HARDWARE_PROFILE or env override}"
    : "${MAX_NUM_SEQS:?MAX_NUM_SEQS must be set by KVBM_HARDWARE_PROFILE or env override}"
    : "${CPU_CACHE_GB:?CPU_CACHE_GB must be set by KVBM_HARDWARE_PROFILE or env override}"
    PREFILL_CUDA_VISIBLE_DEVICES=${PREFILL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}
    : "${PREFILL_CUDA_VISIBLE_DEVICES:?PREFILL_CUDA_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES is required for NODE_ROLE=prefill}"
    ;;
esac

mkdir -p "$ARTIFACT_DIR"
if [[ "$ISOLATE_HF_CACHE" == "1" ]]; then
  export HF_HOME="$ARTIFACT_DIR/hf-cache"
  export HF_HUB_CACHE="$HF_HOME/hub"
  export TRANSFORMERS_CACHE="$HF_HOME/transformers"
fi

exec > >(tee "$ARTIFACT_DIR/${NODE_ROLE}.log") 2>&1
kvbm_xdc_prepare_runtime

cd "$WORKTREE"

export DYN_DISCOVERY_BACKEND=${DYN_DISCOVERY_BACKEND:-etcd}
export DYN_REQUEST_PLANE=${DYN_REQUEST_PLANE:-tcp}
export DYN_EVENT_PLANE=${DYN_EVENT_PLANE:-nats}
export ETCD_ENDPOINTS=${ETCD_ENDPOINTS:-127.0.0.1:2379}
export NATS_SERVER=${NATS_SERVER:-nats://127.0.0.1:4222}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

cat >"$ARTIFACT_DIR/metadata.env" <<EOF
timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
experiment=$KVBM_EXPERIMENT_ID
transfer_topology=$KVBM_TRANSFER_TOPOLOGY
topology=$(if [[ "$KVBM_TRANSFER_TOPOLOGY" == "kvbm-hub" ]]; then printf 'dynamo-native-kvbm-hub'; else printf 'dynamo-native-kvbm-nixl-diagnostic'; fi)
experiment_e_transfer_path=kvbm-hub-conditional-disagg-required
experiment_e_complete=false
experiment_e_smoke_valid=false
hardware_profile=$KVBM_HARDWARE_PROFILE
gpu_class=$GPU_CLASS
node_role=$NODE_ROLE
node_label=${NODE_LABEL:-redacted}
hostname_redacted=true
worktree_redacted=true
model=$SERVED_MODEL_NAME
model_load_path=$(if [[ "$KVBM_RECORD_NETWORK_ENDPOINTS" == "1" ]]; then printf '%s' "$MODEL_PATH"; elif [[ "$MODEL_PATH" == "$MODEL" ]]; then printf 'same-as-model'; else printf 'redacted'; fi)
model_load_path_redacted=$(if [[ "$KVBM_RECORD_NETWORK_ENDPOINTS" == "1" || "$MODEL_PATH" == "$MODEL" ]]; then printf 'false'; else printf 'true'; fi)
framework=dynamo.vllm
frontend=dynamo.frontend
raw_vllm=false
endpoint_type=chat
streaming=true
namespace=$NAMESPACE
http_port=$HTTP_PORT
frontend_host=$(if [[ "$KVBM_RECORD_NETWORK_ENDPOINTS" == "1" ]]; then printf '%s' "$FRONTEND_HOST"; else printf 'redacted'; fi)
discovery_backend=$DYN_DISCOVERY_BACKEND
request_plane=$DYN_REQUEST_PLANE
event_plane=$DYN_EVENT_PLANE
etcd_endpoints=$(if [[ "$KVBM_RECORD_NETWORK_ENDPOINTS" == "1" ]]; then printf '%s' "$ETCD_ENDPOINTS"; else printf 'redacted'; fi)
nats_server=$(if [[ "$KVBM_RECORD_NETWORK_ENDPOINTS" == "1" ]]; then printf '%s' "$NATS_SERVER"; else printf 'redacted'; fi)
max_model_len=${MAX_MODEL_LEN:-}
max_num_seqs=${MAX_NUM_SEQS:-}
cpu_cache_gb=${CPU_CACHE_GB:-}
enforce_eager=$ENFORCE_EAGER
vllm_runner=$VLLM_RUNNER
num_gpu_blocks_override=${NUM_GPU_BLOCKS_OVERRIDE:-}
kvbm_connector_module_path=$KVBM_CONNECTOR_MODULE_PATH
prefill_kv_connector_wrapper=$PREFILL_KV_CONNECTOR_WRAPPER
kvbm_hub_url=$(if [[ "$KVBM_RECORD_NETWORK_ENDPOINTS" == "1" ]]; then printf '%s' "$KVBM_HUB_URL"; else printf 'redacted'; fi)
kvbm_hub_prefill_url=$(if [[ "$KVBM_RECORD_NETWORK_ENDPOINTS" == "1" ]]; then printf '%s' "$KVBM_HUB_PREFILL_URL"; else printf 'redacted'; fi)
kvbm_hub_prefill_namespace=$KVBM_HUB_PREFILL_NAMESPACE
kvbm_hub_prefill_http_port=$KVBM_HUB_PREFILL_HTTP_PORT
kvbm_hub_prefill_dispatcher=dynamo-prefill-completions-shim
kvbm_block_layout=$KVBM_BLOCK_LAYOUT
kvbm_onboard_mode=$KVBM_ONBOARD_MODE
kvbm_worker_nixl_backends=${KVBM_WORKER_NIXL_BACKENDS:-}
kvbm_transfer_substrate_preflight=$KVBM_TRANSFER_SUBSTRATE_PREFLIGHT
kvbm_allow_substrate_failure_artifact=$KVBM_ALLOW_SUBSTRATE_FAILURE_ARTIFACT
decode_cuda_visible_devices=${DECODE_CUDA_VISIBLE_DEVICES:-}
prefill_cuda_visible_devices=${PREFILL_CUDA_VISIBLE_DEVICES:-}
decode_side_channel_host=$(if [[ "$KVBM_RECORD_NETWORK_ENDPOINTS" == "1" ]]; then printf '%s' "${DECODE_SIDE_CHANNEL_HOST:-}"; else printf 'redacted'; fi)
prefill_side_channel_host=$(if [[ "$KVBM_RECORD_NETWORK_ENDPOINTS" == "1" ]]; then printf '%s' "${PREFILL_SIDE_CHANNEL_HOST:-}"; else printf 'redacted'; fi)
network_endpoints_redacted=$(if [[ "$KVBM_RECORD_NETWORK_ENDPOINTS" == "1" ]]; then printf 'false'; else printf 'true'; fi)
trace_renderer=examples/backends/vllm/launch/kvbm_xdc/kvbm-xdc-trace.py
enable_permute_local_kv=$ENABLE_PERMUTE_LOCAL_KV
vllm_kv_cache_layout=$VLLM_KV_CACHE_LAYOUT
rust_log=$RUST_LOG
dyn_log=$DYN_LOG
kvbm_plain_audit=$KVBM_PLAIN_AUDIT
ucx_rcache_max_unreleased=${UCX_RCACHE_MAX_UNRELEASED:-}
ucx_tls=${UCX_TLS:-}
ucx_net_devices=${UCX_NET_DEVICES:-}
kvbm_force_ucx_local_fallback=$KVBM_FORCE_UCX_LOCAL_FALLBACK
frontend_enforce_disagg=$FRONTEND_ENFORCE_DISAGG
serve_only=$KVBM_SERVE_ONLY
start_local_infra=$START_LOCAL_INFRA
cleanup_stale_processes=$KVBM_CLEANUP_STALE_PROCESSES
transfer_evidence_timeout=$TRANSFER_EVIDENCE_TIMEOUT
infra_host=$(if [[ "$KVBM_RECORD_NETWORK_ENDPOINTS" == "1" ]]; then printf '%s' "$INFRA_HOST"; else printf 'redacted'; fi)
prefetch_model=$PREFETCH_MODEL
isolate_hf_cache=$ISOLATE_HF_CACHE
hf_home=$(if [[ "$KVBM_RECORD_NETWORK_ENDPOINTS" == "1" ]]; then printf '%s' "${HF_HOME:-}"; else printf 'redacted'; fi)
hf_hub_cache=$(if [[ "$KVBM_RECORD_NETWORK_ENDPOINTS" == "1" ]]; then printf '%s' "${HF_HUB_CACHE:-}"; else printf 'redacted'; fi)
transformers_cache=$(if [[ "$KVBM_RECORD_NETWORK_ENDPOINTS" == "1" ]]; then printf '%s' "${TRANSFORMERS_CACHE:-}"; else printf 'redacted'; fi)
cache_paths_redacted=$(if [[ "$KVBM_RECORD_NETWORK_ENDPOINTS" == "1" ]]; then printf 'false'; else printf 'true'; fi)
EOF

stop_tracked_pid() {
  local name=$1
  local pid=${2:-}
  [[ -n "$pid" ]] || return 1
  if kill -0 "$pid" 2>/dev/null; then
    echo "[smoke] stop $name pid=$pid"
    kill "$pid" 2>/dev/null || true
    return 0
  fi
  return 1
}

force_kill_tracked_pid() {
  local name=$1
  local pid=${2:-}
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[smoke] force stop $name pid=$pid"
    kill -9 "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  set +e
  if [[ "$KVBM_CLEANUP_STALE_PROCESSES" == "1" ]]; then
    pkill -f "dynamo[.]frontend" 2>/dev/null
    pkill -f "dynamo[.]vllm" 2>/dev/null
    pkill -f "[k]vbm-prefill-completions-shim[.]py" 2>/dev/null
    pkill -f "[E]ngineCore" 2>/dev/null
    pkill -x "kvbm_hub" 2>/dev/null
  fi
  local tracked_pid_stopped=0
  if stop_tracked_pid frontend "${FRONTEND_PID:-}"; then tracked_pid_stopped=1; fi
  if stop_tracked_pid decode "${DECODE_PID:-}"; then tracked_pid_stopped=1; fi
  if stop_tracked_pid prefill "${PREFILL_PID:-}"; then tracked_pid_stopped=1; fi
  if stop_tracked_pid hub "${HUB_PID:-}"; then tracked_pid_stopped=1; fi
  if stop_tracked_pid hub-prefill-frontend "${HUB_PREFILL_FRONTEND_PID:-}"; then tracked_pid_stopped=1; fi
  if stop_tracked_pid etcd "${ETCD_PID:-}"; then tracked_pid_stopped=1; fi
  if stop_tracked_pid nats "${NATS_PID:-}"; then tracked_pid_stopped=1; fi
  if [[ "$tracked_pid_stopped" == "1" ]]; then
    sleep 1
  fi
  force_kill_tracked_pid frontend "${FRONTEND_PID:-}"
  force_kill_tracked_pid decode "${DECODE_PID:-}"
  force_kill_tracked_pid prefill "${PREFILL_PID:-}"
  force_kill_tracked_pid hub "${HUB_PID:-}"
  force_kill_tracked_pid hub-prefill-frontend "${HUB_PREFILL_FRONTEND_PID:-}"
  force_kill_tracked_pid etcd "${ETCD_PID:-}"
  force_kill_tracked_pid nats "${NATS_PID:-}"
  set -e
}

write_failure_trace_gate() {
  local raw_reason=${1:-smoke_failed}
  local reason
  reason=$(printf '%s' "$raw_reason" | tr -c 'A-Za-z0-9_.:-' '_')
  reason=${reason%_}
  reason=${reason:-smoke_failed}

  [[ -f "$ARTIFACT_DIR/trace-gate.env" ]] && return 0

  local trace_html_bytes=0
  if [[ -f "$ARTIFACT_DIR/trace.html" ]]; then
    trace_html_bytes=$(wc -c < "$ARTIFACT_DIR/trace.html" 2>/dev/null || echo 0)
  fi

  local runtime_probe_passed=false
  if grep -q '"probe_passed": true' "$ARTIFACT_DIR/runtime-probe.json" 2>/dev/null; then
    runtime_probe_passed=true
  fi

  {
    echo "trace_useful=false"
    echo "useful_trace_complete=false"
    echo "reason=$reason"
    echo "experiment=$KVBM_EXPERIMENT_ID"
    echo "transfer_topology=$KVBM_TRANSFER_TOPOLOGY"
    echo "experiment_e_complete=false"
    echo "experiment_e_smoke_valid=false"
    if [[ "$KVBM_EXPERIMENT_ID" == "E" && "$KVBM_TRANSFER_TOPOLOGY" == "kvbm-hub" ]]; then
      echo "experiment_e_aiperf_required=true"
    else
      echo "experiment_e_aiperf_required=false"
    fi
    echo "runtime_probe_passed=$runtime_probe_passed"
    echo "trace_html_bytes=$trace_html_bytes"
  } > "$ARTIFACT_DIR/trace-gate.env"
}

fail() {
  echo "SMOKE_FAIL: $*" >&2
  if declare -F render_trace >/dev/null 2>&1; then
    render_trace || true
  fi
  write_failure_trace_gate "$*"
  for log in frontend hub-prefill-frontend hub decode prefill client; do
    if [[ -f "$ARTIFACT_DIR/$log.log" ]]; then
      echo "--- tail $log.log ---" >&2
      tail -n 100 "$ARTIFACT_DIR/$log.log" | sed 's/\x1b\[[0-9;]*m//g' >&2
    fi
  done
  exit 1
}

write_substrate_preflight_failure() {
  local reason=$1
  local peermem_state=$2

  {
    echo "trace_useful=false"
    echo "useful_trace_complete=false"
    echo "reason=$reason"
    echo "experiment=$KVBM_EXPERIMENT_ID"
    echo "transfer_topology=$KVBM_TRANSFER_TOPOLOGY"
    echo "experiment_e_complete=false"
    echo "experiment_e_smoke_valid=false"
    echo "experiment_e_aiperf_required=true"
    echo "runtime_probe=not_run"
    echo "nvidia_peermem=$peermem_state"
  } > "$ARTIFACT_DIR/trace-gate.env"

  {
    echo "# Experiment E Same-Node Verdict"
    echo
    echo "status=not_benchmark_valid"
    echo "reason=$reason"
    echo "runtime_probe=not_run"
    echo "experiment_e_complete=false"
    echo "experiment_e_smoke_valid=false"
    echo "trace_useful=false"
    echo "nvidia_peermem=$peermem_state"
    echo "model=$SERVED_MODEL_NAME"
    echo "framework=dynamo.vllm"
    echo "endpoint_type=chat"
    echo "streaming=true"
    echo
    echo "This preflight fails before launch because the selected KVBM+hub host-cache path requires a working NIXL/UCX transfer substrate. Re-run on a verified H100/A100 substrate or set KVBM_ALLOW_SUBSTRATE_FAILURE_ARTIFACT=1 only when intentionally collecting a failure artifact."
  } > "$ARTIFACT_DIR/verdict.md"
}

transfer_substrate_preflight() {
  case "$KVBM_TRANSFER_SUBSTRATE_PREFLIGHT" in
    0|false|off) return 0 ;;
  esac
  [[ "$KVBM_TRANSFER_TOPOLOGY" == "kvbm-hub" ]] || return 0
  case "$NODE_ROLE" in
    all|decode|prefill) ;;
    *) return 0 ;;
  esac
  [[ "${CPU_CACHE_GB:-0}" != "0" ]] || return 0
  case "$GPU_CLASS" in
    *H100*|*A100*) ;;
    *) return 0 ;;
  esac

  local peermem_state=loaded
  if [[ ! -d /sys/module/nvidia_peermem ]]; then
    peermem_state=missing
  fi

  {
    echo "nvidia_peermem=$peermem_state"
    echo "gpu_class=$GPU_CLASS"
    echo "transfer_topology=$KVBM_TRANSFER_TOPOLOGY"
    echo "cpu_cache_gb=${CPU_CACHE_GB:-}"
    echo "node_role=$NODE_ROLE"
  } > "$ARTIFACT_DIR/substrate-preflight.env"

  local strict_gate=0
  if [[ "$KVBM_TRANSFER_SUBSTRATE_PREFLIGHT" == "strict" ]]; then
    strict_gate=1
  elif [[ "$KVBM_TRANSFER_SUBSTRATE_PREFLIGHT" == "auto" && "$KVBM_EXPERIMENT_ID" == "E" ]]; then
    strict_gate=1
  fi

  if [[ "$peermem_state" == "missing" && "$strict_gate" == "1" && "$KVBM_ALLOW_SUBSTRATE_FAILURE_ARTIFACT" != "1" ]]; then
    write_substrate_preflight_failure kvbm_transfer_substrate_preflight_failed "$peermem_state"
    echo "SUBSTRATE_PREFLIGHT_FAIL: nvidia_peermem is missing for KVBM+hub host-cache H100/A100 run" >&2
    echo "SUBSTRATE_PREFLIGHT_FAIL: artifact_dir=$ARTIFACT_DIR" >&2
    exit 2
  fi

  if [[ "$peermem_state" == "missing" ]]; then
    echo "[smoke] transfer substrate preflight warning: nvidia_peermem=missing; artifact is diagnostic unless transfer evidence later proves the path" >&2
    return 0
  fi

  echo "[smoke] transfer substrate preflight: nvidia_peermem=$peermem_state"
}

start_local_infra() {
  [[ "$START_LOCAL_INFRA" == "1" ]] || return 0

  if curl -fsS -m 3 "http://127.0.0.1:2379/health" >/dev/null 2>&1; then
    echo "[infra] etcd already running"
  else
    echo "[infra] start etcd"
    rm -rf "$ARTIFACT_DIR/etcd-data"
    mkdir -p "$ARTIFACT_DIR/etcd-data"
    /usr/local/bin/etcd/etcd \
      --data-dir "$ARTIFACT_DIR/etcd-data" \
      --listen-client-urls "http://0.0.0.0:2379" \
      --advertise-client-urls "http://$INFRA_HOST:2379" \
      >"$ARTIFACT_DIR/etcd.log" 2>&1 &
    ETCD_PID=$!
  fi

  if (exec 3<>/dev/tcp/127.0.0.1/4222) >/dev/null 2>&1; then
    echo "[infra] nats already running"
  else
    echo "[infra] start nats"
    nats-server --addr 0.0.0.0 --port 4222 --http_port 8222 \
      >"$ARTIFACT_DIR/nats.log" 2>&1 &
    NATS_PID=$!
  fi

  local deadline=$(( $(date +%s) + 60 ))
  until curl -fsS -m 3 "http://127.0.0.1:2379/health" >/dev/null 2>&1 \
    && curl -fsS -m 3 "http://127.0.0.1:8222/healthz" >/dev/null 2>&1; do
    [[ "$(date +%s)" -ge "$deadline" ]] && fail "local NATS/etcd infra not ready"
    sleep 1
  done
  echo "[infra] ready"
}

prefetch_model() {
  if [[ -d "$MODEL_PATH" ]]; then
    echo "[smoke] use local model snapshot path for $SERVED_MODEL_NAME"
    printf '%s\n' "$MODEL_PATH" > "$ARTIFACT_DIR/model-snapshot-path.txt"
    return 0
  fi

  [[ "$PREFETCH_MODEL" == "1" ]] || return 0

  echo "[smoke] prefetch model cache for $MODEL_PATH"
  "$PYTHON_BIN" - "$MODEL_PATH" "$ARTIFACT_DIR/model-snapshot-path.txt" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

model, out_path = sys.argv[1:3]
path = snapshot_download(repo_id=model, local_files_only=False)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(path + "\n")
print(path)
print("HF_HOME=" + os.environ.get("HF_HOME", ""))
print("HF_HUB_CACHE=" + os.environ.get("HF_HUB_CACHE", ""))
PY
}

runtime_probe() {
  echo "[smoke] runtime probe: Python imports, torch CUDA, and device matmul"
  "$PYTHON_BIN" - \
    "$ARTIFACT_DIR/runtime-probe.json" \
    "$NODE_ROLE" \
    "${DECODE_CUDA_VISIBLE_DEVICES:-}" \
    "${PREFILL_CUDA_VISIBLE_DEVICES:-}" <<'PY'
import importlib
import json
import sys

out_path, node_role, decode_devices, prefill_devices = sys.argv[1:5]
imports = [
    "dynamo._core",
    "dynamo.vllm.handlers",
    "kvbm",
    "torch",
    "vllm",
]
loaded = {}
for name in imports:
    module = importlib.import_module(name)
    loaded[name] = getattr(module, "__file__", "built-in")

import torch

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")

device_count = torch.cuda.device_count()
requested = []
for raw in (decode_devices, prefill_devices):
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if idx not in requested:
                requested.append(idx)

if node_role == "all" and len(requested) < 2:
    raise SystemExit(
        f"NODE_ROLE=all requires distinct decode/prefill CUDA device ids; got {requested}"
    )

for idx in requested:
    if idx >= device_count:
        raise SystemExit(
            f"requested CUDA device {idx} is outside torch device_count={device_count}"
        )
    torch.cuda.set_device(idx)
    a = torch.ones((8, 8), device=f"cuda:{idx}")
    b = a @ a
    torch.cuda.synchronize(idx)
    if float(b[0, 0].item()) != 8.0:
        raise SystemExit(f"unexpected CUDA matmul result on device {idx}")

result = {
    "imports": loaded,
    "torch_cuda_available": True,
    "torch_cuda_device_count": device_count,
    "requested_cuda_devices": requested,
    "probe": "imports_cuda_matmul",
    "probe_passed": True,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, sort_keys=True)
print(json.dumps(result, indent=2, sort_keys=True))
print("RUNTIME_PROBE_PASS")
PY
}

start_kvbm_hub() {
  [[ "$KVBM_TRANSFER_TOPOLOGY" == "kvbm-hub" ]] || return 0

  local logged_prefill_url=redacted
  if [[ "$KVBM_RECORD_NETWORK_ENDPOINTS" == "1" ]]; then
    logged_prefill_url=$KVBM_HUB_PREFILL_URL
  fi
  echo "[smoke] start kvbm_hub dispatcher prefill_url=$logged_prefill_url"
  env KVBM_REPO="$WORKTREE" KVBM_HUB_MODEL="$SERVED_MODEL_NAME" \
    KVBM_HUB_PREFILL_URL="$KVBM_HUB_PREFILL_URL" \
    RUST_LOG="$RUST_LOG" \
    KVBM_PLAIN_AUDIT="$KVBM_PLAIN_AUDIT" \
    bash "$SCRIPT_DIR/launch-kvbm-hub.sh" "$ARTIFACT_DIR/hub.log" &
  HUB_PID=$!
  echo "[smoke] kvbm_hub pid=$HUB_PID"
}

start_hub_prefill_frontend() {
  [[ "$KVBM_TRANSFER_TOPOLOGY" == "kvbm-hub" ]] || return 0

  echo "[smoke] start Dynamo prefill completions shim namespace=$KVBM_HUB_PREFILL_NAMESPACE port=$KVBM_HUB_PREFILL_HTTP_PORT"
  env WORKTREE="$WORKTREE" VENV="$VENV" PYTHON_BIN="$PYTHON_BIN" \
    MODEL="$SERVED_MODEL_NAME" \
    NAMESPACE="$KVBM_HUB_PREFILL_NAMESPACE" \
    KVBM_HUB_PREFILL_ENDPOINT="$KVBM_HUB_PREFILL_NAMESPACE.backend.generate" \
    KVBM_HUB_PREFILL_SHIM_HOST="$KVBM_HUB_PREFILL_HOST" \
    KVBM_HUB_PREFILL_SHIM_PORT="$KVBM_HUB_PREFILL_HTTP_PORT" \
    SOURCE_KVBM_RUNTIME_ENV="$SOURCE_KVBM_RUNTIME_ENV" \
    bash "$SCRIPT_DIR/launch-kvbm-prefill-shim.sh" \
    >"$ARTIFACT_DIR/hub-prefill-frontend.log" 2>&1 &
  HUB_PREFILL_FRONTEND_PID=$!
  echo "[smoke] prefill completions shim pid=$HUB_PREFILL_FRONTEND_PID"
}

wait_for_frontend() {
  local deadline=$(( $(date +%s) + READY_TIMEOUT ))
  local status_logged=0
  until curl -fsS -m 5 "http://$FRONTEND_HOST:$HTTP_PORT/v1/models" >"$ARTIFACT_DIR/models.json" 2>"$ARTIFACT_DIR/models.err" \
    && "$PYTHON_BIN" - "$ARTIFACT_DIR/models.json" "$SERVED_MODEL_NAME" <<'PY'
import json
import sys

path, model = sys.argv[1:3]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
ids = [item.get("id") for item in data.get("data", []) if isinstance(item, dict)]
raise SystemExit(0 if model in ids else 1)
PY
  do
    [[ "$(date +%s)" -ge "$deadline" ]] && fail "frontend not ready after ${READY_TIMEOUT}s"
    if [[ -n "${FRONTEND_PID:-}" ]]; then
      kill -0 "$FRONTEND_PID" 2>/dev/null || fail "frontend exited before ready"
    fi
    if [[ -n "${DECODE_PID:-}" ]]; then
      kill -0 "$DECODE_PID" 2>/dev/null || fail "decode worker exited before frontend ready"
    fi
    if [[ -n "${PREFILL_PID:-}" ]]; then
      kill -0 "$PREFILL_PID" 2>/dev/null || fail "prefill worker exited before frontend ready"
    fi
    if [[ -n "${HUB_PID:-}" ]]; then
      kill -0 "$HUB_PID" 2>/dev/null || fail "kvbm_hub exited before frontend ready"
    fi
    if [[ "$status_logged" == "0" && -s "$ARTIFACT_DIR/models.json" ]]; then
      echo "[smoke] frontend answered, waiting for model registration"
      cat "$ARTIFACT_DIR/models.json"
      echo
      status_logged=1
    fi
    sleep 5
  done
  echo "[smoke] frontend model ready"
  cat "$ARTIFACT_DIR/models.json"
  echo
}

wait_for_frontend_chat_route() {
  local deadline=$(( $(date +%s) + READY_TIMEOUT ))
  local frontend_log="$ARTIFACT_DIR/frontend.log"
  echo "[smoke] wait for frontend chat route"
  while ! grep -aEq "Chat completions is ready|chat endpoints enabled" "$frontend_log" 2>/dev/null; do
    [[ "$(date +%s)" -ge "$deadline" ]] && fail "frontend chat route not ready after ${READY_TIMEOUT}s"
    if [[ -n "${FRONTEND_PID:-}" ]]; then
      kill -0 "$FRONTEND_PID" 2>/dev/null || fail "frontend exited before chat route ready"
    fi
    if [[ -n "${DECODE_PID:-}" ]]; then
      kill -0 "$DECODE_PID" 2>/dev/null || fail "decode worker exited before frontend chat route ready"
    fi
    if [[ -n "${PREFILL_PID:-}" ]]; then
      kill -0 "$PREFILL_PID" 2>/dev/null || fail "prefill worker exited before frontend chat route ready"
    fi
    if [[ -n "${HUB_PID:-}" ]]; then
      kill -0 "$HUB_PID" 2>/dev/null || fail "kvbm_hub exited before frontend chat route ready"
    fi
    sleep 1
  done
  echo "[smoke] frontend chat route ready"
}

wait_for_hub_prefill_frontend() {
  [[ "$KVBM_TRANSFER_TOPOLOGY" == "kvbm-hub" ]] || return 0

  local deadline=$(( $(date +%s) + READY_TIMEOUT ))
  local status_logged=0
  until curl -fsS -m 5 "http://$KVBM_HUB_PREFILL_HOST:$KVBM_HUB_PREFILL_HTTP_PORT/v1/models" >"$ARTIFACT_DIR/hub-prefill-models.json" 2>"$ARTIFACT_DIR/hub-prefill-models.err" \
    && "$PYTHON_BIN" - "$ARTIFACT_DIR/hub-prefill-models.json" "$SERVED_MODEL_NAME" <<'PY'
import json
import sys

path, model = sys.argv[1:3]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
ids = [item.get("id") for item in data.get("data", []) if isinstance(item, dict)]
raise SystemExit(0 if model in ids else 1)
PY
  do
    [[ "$(date +%s)" -ge "$deadline" ]] && fail "internal prefill frontend not ready after ${READY_TIMEOUT}s"
    if [[ -n "${HUB_PREFILL_FRONTEND_PID:-}" ]]; then
      kill -0 "$HUB_PREFILL_FRONTEND_PID" 2>/dev/null || fail "internal prefill frontend exited before ready"
    fi
    if [[ -n "${PREFILL_PID:-}" ]]; then
      kill -0 "$PREFILL_PID" 2>/dev/null || fail "prefill worker exited before internal frontend ready"
    fi
    if [[ "$status_logged" == "0" && -s "$ARTIFACT_DIR/hub-prefill-models.json" ]]; then
      echo "[smoke] internal prefill frontend answered, waiting for model registration"
      cat "$ARTIFACT_DIR/hub-prefill-models.json"
      echo
      status_logged=1
    fi
    sleep 5
  done
  echo "[smoke] internal prefill frontend model ready"
  cat "$ARTIFACT_DIR/hub-prefill-models.json"
  echo
}

run_client() {
  echo "[smoke] streaming chat requests (R1 cold, R2 shared-prefix reuse)"
  "$PYTHON_BIN" - "$FRONTEND_HOST" "$HTTP_PORT" "$SERVED_MODEL_NAME" "$ARTIFACT_DIR" \
    >"$ARTIFACT_DIR/chat-ttft.json" <<'PY'
import json
import sys
import time
import urllib.request

host, port, model, artifact_dir = sys.argv[1:5]
url = f"http://{host}:{port}/v1/chat/completions"
prefix = (
    "KV cache transfer matters for disaggregated LLM serving because prefill "
    "and decode often run on different workers. A correct implementation must "
    "move reusable prefix blocks without recomputing them, preserve request "
    "ordering, and expose enough telemetry to prove that reuse happened. "
)
prompts = [
    (
        "r1",
        prefix
        + "Explain this in four concise sentences for an inference engineer.",
    ),
    (
        "r2",
        prefix
        + "Now compare the latency implications for a same-node baseline versus "
        "a cross-datacenter deployment in four concise sentences.",
    ),
]

def send(label, prompt):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 96,
        "temperature": 0,
        "stream": True,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_byte = None
    first_token = None
    chunks = 0
    content_chars = 0
    stream_path = f"{artifact_dir}/chat-{label}-stream.jsonl"
    with urllib.request.urlopen(req, timeout=300) as resp, open(stream_path, "w") as out:
        for raw in resp:
            now = time.perf_counter()
            if first_byte is None:
                first_byte = now
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            out.write(line + "\n")
            out.flush()
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            chunks += 1
            obj = json.loads(data)
            if "error" in obj:
                print(
                    "server returned streaming error: "
                    + json.dumps(obj["error"], sort_keys=True),
                    file=sys.stderr,
                )
                raise SystemExit(1)
            if "choices" not in obj:
                print(
                    "streaming chunk missing choices: "
                    + json.dumps(obj, sort_keys=True),
                    file=sys.stderr,
                )
                raise SystemExit(1)
            delta = obj["choices"][0].get("delta", {})
            text = delta.get("content") or ""
            if text and first_token is None:
                first_token = now
            content_chars += len(text)
    end = time.perf_counter()
    if first_token is None or chunks < 1 or content_chars < 1:
        raise SystemExit(f"{label} did not produce streamed text")
    return {
        "label": label,
        "ttfb_seconds": None if first_byte is None else first_byte - start,
        "ttft_seconds": first_token - start,
        "total_seconds": end - start,
        "chunks": chunks,
        "content_chars": content_chars,
        "stream_path": stream_path,
    }

results = [send(label, prompt) for label, prompt in prompts]
with open(f"{artifact_dir}/chat-metrics.env", "w", encoding="utf-8") as env:
    env.write(f"successful_streamed_chat_requests={len(results)}\n")
    for item in results:
        env.write(f"{item['label']}_ttft_seconds={item['ttft_seconds']:.6f}\n")
        env.write(f"{item['label']}_chunks={item['chunks']}\n")
        env.write(f"{item['label']}_content_chars={item['content_chars']}\n")

print(json.dumps({
    "model": model,
    "framework": "dynamo.vllm",
    "endpoint_type": "chat",
    "streaming": True,
    "request_count": len(results),
    "requests": results,
}, indent=2, sort_keys=True))
PY
  cat "$ARTIFACT_DIR/chat-ttft.json"
}

render_trace() {
  echo "[smoke] render trace"
  if [[ -f "$ARTIFACT_DIR/frontend.log" && ! -f "$ARTIFACT_DIR/hub.log" ]]; then
    cp "$ARTIFACT_DIR/frontend.log" "$ARTIFACT_DIR/hub.log"
  fi
  touch "$ARTIFACT_DIR/hub.log"
  "$PYTHON_BIN" "$SCRIPT_DIR/kvbm-xdc-trace.py" "$ARTIFACT_DIR" \
    | tee "$ARTIFACT_DIR/trace-render.log"
}

grep_count() {
  local pattern=$1
  shift
  { grep -aE "$pattern" "$@" 2>/dev/null || true; } | wc -l | tr -d ' '
}

wait_for_trace_evidence() {
  local deadline=$(( $(date +%s) + TRANSFER_EVIDENCE_TIMEOUT ))
  echo "[smoke] wait for transfer evidence"
  while true; do
    local audit_events
    local transfer_events
    local dynamo_transfer_metrics
    local external_cache_hit_events
    local nixl_transfer_errors
    audit_events=$(grep_count "kvbm_audit" \
      "$ARTIFACT_DIR/hub.log" "$ARTIFACT_DIR/frontend.log" "$ARTIFACT_DIR/decode.log" "$ARTIFACT_DIR/prefill.log")
    transfer_events=$(grep_count "event=\"(session_pull_rdma_done|worker_session_pull_returned|worker_g2_to_g1_done|transfer_pull_registered|transfer_pull_completed)\"" \
      "$ARTIFACT_DIR/hub.log" "$ARTIFACT_DIR/frontend.log" "$ARTIFACT_DIR/decode.log" "$ARTIFACT_DIR/prefill.log")
    dynamo_transfer_metrics=$(grep_count "KV Transfer metrics: Num successful transfers=[1-9][0-9]*" \
      "$ARTIFACT_DIR/hub.log" "$ARTIFACT_DIR/frontend.log" "$ARTIFACT_DIR/decode.log" "$ARTIFACT_DIR/prefill.log")
    external_cache_hit_events=$(grep_count "External prefix cache hit rate:[[:space:]]*([1-9][0-9]*([.][0-9]+)?|0*[.][0-9]*[1-9][0-9]*)%" \
      "$ARTIFACT_DIR/hub.log" "$ARTIFACT_DIR/frontend.log" "$ARTIFACT_DIR/decode.log" "$ARTIFACT_DIR/prefill.log")
    nixl_transfer_errors=$(grep_count "createXferReq: no potential backend found" \
      "$ARTIFACT_DIR/hub.log" "$ARTIFACT_DIR/frontend.log" "$ARTIFACT_DIR/decode.log" "$ARTIFACT_DIR/prefill.log")

    if [[ "$nixl_transfer_errors" -gt 0 ]]; then
      echo "[smoke] transfer failure evidence observed; validation will fail"
      return 0
    fi
    if [[ "$audit_events" -ge 1 && "$transfer_events" -ge 1 && "$external_cache_hit_events" -ge 1 ]]; then
      echo "[smoke] audit transfer and external cache-hit evidence observed"
      return 0
    fi
    if [[ "$KVBM_EXPERIMENT_ID" != "E" || "$KVBM_TRANSFER_TOPOLOGY" != "kvbm-hub" ]] \
      && [[ "$dynamo_transfer_metrics" -ge 1 && "$external_cache_hit_events" -ge 1 ]]; then
      echo "[smoke] Dynamo transfer metrics and external cache-hit evidence observed"
      return 0
    fi
    if [[ "$(date +%s)" -ge "$deadline" ]]; then
      echo "[smoke] transfer evidence not observed within ${TRANSFER_EVIDENCE_TIMEOUT}s"
      return 0
    fi
    sleep 1
  done
}

validate_useful_trace() {
  local audit_events
  local transfer_events
  local dynamo_transfer_metrics
  local request_successes
  local selected_prefill
  local selected_decode
  local nixl_transfer_errors
  local external_cache_hit_events
  local kv_load_failure_events
  local client_request_successes
  local remote_slots_empty_events
  local remote_slots_nonzero_events
  local missing_required_role_logs=0

  if [[ "$KVBM_EXPERIMENT_ID" == "E" && "$KVBM_TRANSFER_TOPOLOGY" == "kvbm-hub" ]]; then
    for required_log in frontend hub hub-prefill-frontend decode prefill; do
      if [[ ! -s "$ARTIFACT_DIR/$required_log.log" ]]; then
        echo "  missing required Experiment E role log: $required_log.log"
        missing_required_role_logs=$((missing_required_role_logs + 1))
      fi
    done
  fi

  audit_events=$(grep_count "kvbm_audit" \
    "$ARTIFACT_DIR/hub.log" "$ARTIFACT_DIR/frontend.log" "$ARTIFACT_DIR/decode.log" "$ARTIFACT_DIR/prefill.log")
  transfer_events=$(grep_count "event=\"(session_pull_rdma_done|worker_session_pull_returned|worker_g2_to_g1_done|transfer_pull_registered|transfer_pull_completed)\"" \
    "$ARTIFACT_DIR/hub.log" "$ARTIFACT_DIR/frontend.log" "$ARTIFACT_DIR/decode.log" "$ARTIFACT_DIR/prefill.log")
  dynamo_transfer_metrics=$(grep_count "KV Transfer metrics: Num successful transfers=[1-9][0-9]*" \
    "$ARTIFACT_DIR/hub.log" "$ARTIFACT_DIR/frontend.log" "$ARTIFACT_DIR/decode.log" "$ARTIFACT_DIR/prefill.log")
  request_successes=$(grep_count "request completed .*endpoint.*chat_completions.*status.*success" \
    "$ARTIFACT_DIR/hub.log" "$ARTIFACT_DIR/frontend.log" "$ARTIFACT_DIR/decode.log" "$ARTIFACT_DIR/prefill.log")
  selected_prefill=$(grep_count "Selected worker: worker_type=prefill" \
    "$ARTIFACT_DIR/hub.log" "$ARTIFACT_DIR/frontend.log")
  selected_decode=$(grep_count "Selected worker: worker_type=decode" \
    "$ARTIFACT_DIR/hub.log" "$ARTIFACT_DIR/frontend.log")
  nixl_transfer_errors=$(grep_count "createXferReq: no potential backend found" \
    "$ARTIFACT_DIR/hub.log" "$ARTIFACT_DIR/frontend.log" "$ARTIFACT_DIR/decode.log" "$ARTIFACT_DIR/prefill.log")
  external_cache_hit_events=$(grep_count "External prefix cache hit rate:[[:space:]]*([1-9][0-9]*([.][0-9]+)?|0*[.][0-9]*[1-9][0-9]*)%" \
    "$ARTIFACT_DIR/hub.log" "$ARTIFACT_DIR/frontend.log" "$ARTIFACT_DIR/decode.log" "$ARTIFACT_DIR/prefill.log")
  kv_load_failure_events=$(grep_count "Recovered from KV load failure|Onboarding failed|Failed to start onboarding" \
    "$ARTIFACT_DIR/hub.log" "$ARTIFACT_DIR/frontend.log" "$ARTIFACT_DIR/decode.log" "$ARTIFACT_DIR/prefill.log")
  client_request_successes=$(awk -F= '/^successful_streamed_chat_requests=/ {print $2}' "$ARTIFACT_DIR/chat-metrics.env" 2>/dev/null | tail -1)
  client_request_successes=${client_request_successes:-0}
  remote_slots_empty_events=$(grep_count "event=\"remote_pipeline_complete_set\".*reason=\"remote_slots_empty\"" \
    "$ARTIFACT_DIR/hub.log" "$ARTIFACT_DIR/frontend.log" "$ARTIFACT_DIR/decode.log" "$ARTIFACT_DIR/prefill.log")
  remote_slots_nonzero_events=$(grep_count "event=\"commit_usaa1_state_built\".*remote_slots_len=[1-9][0-9]*" \
    "$ARTIFACT_DIR/hub.log" "$ARTIFACT_DIR/frontend.log" "$ARTIFACT_DIR/decode.log" "$ARTIFACT_DIR/prefill.log")

  echo
  echo "-- useful trace gate --"
  echo "  kvbm_audit events: $audit_events"
  echo "  audit transfer-complete events: $transfer_events"
  echo "  Dynamo transfer metric events: $dynamo_transfer_metrics"
  echo "  successful streamed chat completions: $request_successes"
  echo "  selected prefill workers: $selected_prefill"
  echo "  selected decode workers: $selected_decode"
  echo "  external cache-hit events: $external_cache_hit_events"
  echo "  successful client streamed chat requests: $client_request_successes"
  echo "  KV load failure events: $kv_load_failure_events"
  echo "  NIXL createXferReq failures: $nixl_transfer_errors"
  echo "  remote slots empty events: $remote_slots_empty_events"
  echo "  remote slots nonzero events: $remote_slots_nonzero_events"
  echo "  missing required Experiment E role logs: $missing_required_role_logs"

  write_trace_gate() {
    local useful=$1
    local reason=$2
    {
      echo "trace_useful=$useful"
      echo "useful_trace_complete=$useful"
      echo "reason=$reason"
      echo "experiment=$KVBM_EXPERIMENT_ID"
      echo "transfer_topology=$KVBM_TRANSFER_TOPOLOGY"
      echo "experiment_e_complete=false"
      if [[ "$useful" == "true" && "$KVBM_EXPERIMENT_ID" == "E" && "$KVBM_TRANSFER_TOPOLOGY" == "kvbm-hub" && "$client_request_successes" -ge 2 ]]; then
        echo "experiment_e_smoke_valid=true"
      else
        echo "experiment_e_smoke_valid=false"
      fi
      if [[ "$KVBM_EXPERIMENT_ID" == "E" && "$KVBM_TRANSFER_TOPOLOGY" == "kvbm-hub" ]]; then
        echo "experiment_e_aiperf_required=true"
      else
        echo "experiment_e_aiperf_required=false"
      fi
      echo "trace_html_bytes=$(wc -c < "$ARTIFACT_DIR/trace.html" 2>/dev/null || echo 0)"
      echo "kvbm_audit_events=$audit_events"
      echo "audit_transfer_complete_events=$transfer_events"
      echo "dynamo_transfer_metric_events=$dynamo_transfer_metrics"
      echo "successful_streamed_chat_completions=$request_successes"
      echo "successful_client_streamed_chat_requests=$client_request_successes"
      echo "selected_prefill_workers=$selected_prefill"
      echo "selected_decode_workers=$selected_decode"
      echo "external_cache_hit_events=$external_cache_hit_events"
      echo "kv_load_failure_events=$kv_load_failure_events"
      echo "nixl_create_xfer_req_failures=$nixl_transfer_errors"
      echo "remote_slots_empty_events=$remote_slots_empty_events"
      echo "remote_slots_nonzero_events=$remote_slots_nonzero_events"
      echo "missing_required_experiment_e_role_logs=$missing_required_role_logs"
    } > "$ARTIFACT_DIR/trace-gate.env"
  }

  if [[ "$missing_required_role_logs" -gt 0 ]]; then
    write_trace_gate false missing_required_experiment_e_role_logs
    echo "TRACE_NOT_USEFUL: Experiment E split-role validation is missing required role logs" >&2
    return 2
  fi
  if [[ "$nixl_transfer_errors" -gt 0 ]]; then
    write_trace_gate false nixl_transfer_backend_failure
    echo "TRACE_NOT_USEFUL: NIXL transfer backend failure was captured" >&2
    return 2
  fi
  if [[ "$kv_load_failure_events" -gt 0 ]]; then
    write_trace_gate false kv_load_failure
    echo "TRACE_NOT_USEFUL: KV load failure was captured" >&2
    return 2
  fi
  if [[ "$external_cache_hit_events" -lt 1 ]]; then
    write_trace_gate false no_external_cache_reuse
    echo "TRACE_NOT_USEFUL: no positive external cache-hit evidence was captured" >&2
    return 2
  fi
  if [[ "$client_request_successes" -lt 2 ]]; then
    write_trace_gate false incomplete_r1_r2_chat_smoke
    echo "TRACE_NOT_USEFUL: R1/R2 streamed chat smoke did not complete" >&2
    return 2
  fi
  if [[ "$audit_events" -ge 1 && "$transfer_events" -ge 1 ]]; then
    write_trace_gate true kvbm_audit_transfer_chain_and_cache_reuse
    echo "TRACE_USEFUL: real KVBM transfer audit chain and external cache reuse captured"
    return 0
  fi
  if [[ "$KVBM_EXPERIMENT_ID" == "E" && "$KVBM_TRANSFER_TOPOLOGY" == "kvbm-hub" ]]; then
    if [[ "$remote_slots_empty_events" -gt 0 ]]; then
      write_trace_gate false remote_slots_empty
      echo "TRACE_NOT_USEFUL: Experiment E did not create a remote destination slice" >&2
    elif [[ "$remote_slots_nonzero_events" -gt 0 ]]; then
      write_trace_gate false remote_slots_nonzero_without_destination_pull
      echo "TRACE_NOT_USEFUL: Experiment E created remote slots but no destination pull/completion evidence" >&2
    else
      write_trace_gate false no_destination_transfer_evidence
      echo "TRACE_NOT_USEFUL: Experiment E requires destination pull/completion transfer evidence" >&2
    fi
    return 2
  fi
  if [[ "$dynamo_transfer_metrics" -ge 1 && "$request_successes" -ge 1 && "$selected_prefill" -ge 1 && "$selected_decode" -ge 1 ]]; then
    write_trace_gate true dynamo_vllm_transfer_metrics_and_cache_reuse
    echo "TRACE_USEFUL: Dynamo vLLM routed chat completion captured successful KV transfer metrics and external cache reuse"
    return 0
  fi
  if [[ "$audit_events" -lt 1 && "$dynamo_transfer_metrics" -lt 1 ]]; then
    write_trace_gate false no_transfer_evidence
    echo "TRACE_NOT_USEFUL: real KVBM transfer audit chain was not captured" >&2
    return 2
  fi
  write_trace_gate false incomplete_routed_transfer_evidence
  echo "TRACE_NOT_USEFUL: trace did not capture enough routed transfer evidence" >&2
  return 2
}

common_env=(
  WORKTREE="$WORKTREE"
  VENV="$VENV"
  PYTHON_BIN="$PYTHON_BIN"
  KVBM_HARDWARE_PROFILE="$KVBM_HARDWARE_PROFILE"
  GPU_CLASS="$GPU_CLASS"
  MODEL="$MODEL"
  MODEL_PATH="$MODEL_PATH"
  SERVED_MODEL_NAME="$SERVED_MODEL_NAME"
  NAMESPACE="$NAMESPACE"
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
  CPU_CACHE_GB="${CPU_CACHE_GB:-}"
  ENFORCE_EAGER="$ENFORCE_EAGER"
  VLLM_RUNNER="$VLLM_RUNNER"
  VLLM_USE_AOT_COMPILE="${VLLM_USE_AOT_COMPILE:-0}"
  VLLM_USE_STANDALONE_COMPILE="${VLLM_USE_STANDALONE_COMPILE:-0}"
  VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
  NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-}"
  KVBM_CONNECTOR_MODULE_PATH="$KVBM_CONNECTOR_MODULE_PATH"
  PREFILL_KV_CONNECTOR_WRAPPER="$PREFILL_KV_CONNECTOR_WRAPPER"
  SOURCE_KVBM_RUNTIME_ENV="$SOURCE_KVBM_RUNTIME_ENV"
  KVBM_RUNTIME_ENV_SCRIPT="${KVBM_RUNTIME_ENV_SCRIPT:-}"
  KVBM_TRANSFER_TOPOLOGY="$KVBM_TRANSFER_TOPOLOGY"
  KVBM_HUB_URL="$KVBM_HUB_URL"
  KVBM_BLOCK_LAYOUT="$KVBM_BLOCK_LAYOUT"
  KVBM_ONBOARD_MODE="$KVBM_ONBOARD_MODE"
  KVBM_WORKER_NIXL_BACKENDS="$KVBM_WORKER_NIXL_BACKENDS"
  ENABLE_PERMUTE_LOCAL_KV="$ENABLE_PERMUTE_LOCAL_KV"
  VLLM_KV_CACHE_LAYOUT="$VLLM_KV_CACHE_LAYOUT"
  RUST_LOG="$RUST_LOG"
  DYN_LOG="$DYN_LOG"
  KVBM_PLAIN_AUDIT="$KVBM_PLAIN_AUDIT"
  PYTHONUNBUFFERED=1
)

case "$NODE_ROLE" in
  frontend)
    start_local_infra
    exec env "${common_env[@]}" HTTP_PORT="$HTTP_PORT" ENFORCE_DISAGG="$FRONTEND_ENFORCE_DISAGG" \
      bash "$SCRIPT_DIR/launch-dynamo-frontend.sh"
    ;;
  hub-prefill-frontend)
    start_local_infra
    exec env WORKTREE="$WORKTREE" VENV="$VENV" PYTHON_BIN="$PYTHON_BIN" \
      KVBM_HARDWARE_PROFILE="$KVBM_HARDWARE_PROFILE" \
      MODEL="$SERVED_MODEL_NAME" \
      NAMESPACE="$KVBM_HUB_PREFILL_NAMESPACE" \
      KVBM_HUB_PREFILL_ENDPOINT="$KVBM_HUB_PREFILL_NAMESPACE.backend.generate" \
      KVBM_HUB_PREFILL_SHIM_HOST="$KVBM_HUB_PREFILL_HOST" \
      KVBM_HUB_PREFILL_SHIM_PORT="$KVBM_HUB_PREFILL_HTTP_PORT" \
      SOURCE_KVBM_RUNTIME_ENV="$SOURCE_KVBM_RUNTIME_ENV" \
      KVBM_RUNTIME_ENV_SCRIPT="${KVBM_RUNTIME_ENV_SCRIPT:-}" \
      PYTHONUNBUFFERED=1 \
      bash "$SCRIPT_DIR/launch-kvbm-prefill-shim.sh"
    ;;
  hub)
    exec env KVBM_REPO="$WORKTREE" KVBM_HUB_MODEL="$SERVED_MODEL_NAME" \
      KVBM_HUB_PREFILL_URL="$KVBM_HUB_PREFILL_URL" \
      RUST_LOG="$RUST_LOG" \
      KVBM_PLAIN_AUDIT="$KVBM_PLAIN_AUDIT" \
      bash "$SCRIPT_DIR/launch-kvbm-hub.sh" "$ARTIFACT_DIR/hub.log"
    ;;
  decode)
    transfer_substrate_preflight
    prefetch_model
    exec env "${common_env[@]}" ROLE=decode CUDA_VISIBLE_DEVICES="${DECODE_CUDA_VISIBLE_DEVICES:-}" \
      GPU_MEMORY_UTILIZATION="$DECODE_GPU_MEMORY_UTILIZATION" \
      SIDE_CHANNEL_HOST="${DECODE_SIDE_CHANNEL_HOST:-}" SIDE_CHANNEL_PORT="$DECODE_SIDE_CHANNEL_PORT" \
      bash "$SCRIPT_DIR/launch-kvbm-vllm.sh"
    ;;
  prefill)
    transfer_substrate_preflight
    prefetch_model
    prefill_endpoint_env=()
    if [[ "$KVBM_TRANSFER_TOPOLOGY" == "kvbm-hub" ]]; then
      prefill_endpoint_env=(
        NAMESPACE="$KVBM_HUB_PREFILL_NAMESPACE"
        ENDPOINT="$KVBM_HUB_PREFILL_NAMESPACE.backend.generate"
        ENDPOINT_TYPES=completions
      )
    fi
    exec env "${common_env[@]}" "${prefill_endpoint_env[@]}" ROLE=prefill CUDA_VISIBLE_DEVICES="${PREFILL_CUDA_VISIBLE_DEVICES:-}" \
      GPU_MEMORY_UTILIZATION="$PREFILL_GPU_MEMORY_UTILIZATION" KV_EVENTS_PORT="$PREFILL_KV_EVENTS_PORT" \
      SIDE_CHANNEL_HOST="${PREFILL_SIDE_CHANNEL_HOST:-}" SIDE_CHANNEL_PORT="$PREFILL_SIDE_CHANNEL_PORT" \
      bash "$SCRIPT_DIR/launch-kvbm-vllm.sh"
    ;;
  client)
    wait_for_frontend
    run_client || fail "streaming chat request failed"
    wait_for_trace_evidence
    render_trace
    validate_useful_trace
    echo "SMOKE_DONE artifact_dir=$ARTIFACT_DIR trace=$ARTIFACT_DIR/trace.html"
    ;;
  trace)
    wait_for_trace_evidence
    render_trace
    validate_useful_trace
    echo "TRACE_DONE artifact_dir=$ARTIFACT_DIR trace=$ARTIFACT_DIR/trace.html"
    ;;
  all)
    trap cleanup EXIT
    if [[ "$KVBM_CLEANUP_STALE_PROCESSES" == "1" ]]; then
      echo "[smoke] cleanup stale Dynamo processes"
    else
      echo "[smoke] skip broad stale-process cleanup; tracked top-level PIDs will still be cleaned on exit"
    fi
    cleanup
    sleep 2
    transfer_substrate_preflight
    runtime_probe || fail "runtime_probe_failed"
    start_local_infra
    prefetch_model

    echo "[smoke] start frontend"
    env "${common_env[@]}" HTTP_PORT="$HTTP_PORT" ENFORCE_DISAGG="$FRONTEND_ENFORCE_DISAGG" \
      bash "$SCRIPT_DIR/launch-dynamo-frontend.sh" >"$ARTIFACT_DIR/frontend.log" 2>&1 &
    FRONTEND_PID=$!
    echo "[smoke] frontend pid=$FRONTEND_PID"

    start_hub_prefill_frontend
    start_kvbm_hub

    echo "[smoke] start decode dynamo.vllm"
    env "${common_env[@]}" ROLE=decode CUDA_VISIBLE_DEVICES="$DECODE_CUDA_VISIBLE_DEVICES" \
      GPU_MEMORY_UTILIZATION="$DECODE_GPU_MEMORY_UTILIZATION" \
      SIDE_CHANNEL_HOST="${DECODE_SIDE_CHANNEL_HOST:-}" SIDE_CHANNEL_PORT="$DECODE_SIDE_CHANNEL_PORT" \
      bash "$SCRIPT_DIR/launch-kvbm-vllm.sh" >"$ARTIFACT_DIR/decode.log" 2>&1 &
    DECODE_PID=$!
    echo "[smoke] decode pid=$DECODE_PID"

    echo "[smoke] start prefill dynamo.vllm"
    prefill_endpoint_env=()
    if [[ "$KVBM_TRANSFER_TOPOLOGY" == "kvbm-hub" ]]; then
      prefill_endpoint_env=(
        NAMESPACE="$KVBM_HUB_PREFILL_NAMESPACE"
        ENDPOINT="$KVBM_HUB_PREFILL_NAMESPACE.backend.generate"
        ENDPOINT_TYPES=completions
      )
    fi
    env "${common_env[@]}" "${prefill_endpoint_env[@]}" ROLE=prefill CUDA_VISIBLE_DEVICES="$PREFILL_CUDA_VISIBLE_DEVICES" \
      GPU_MEMORY_UTILIZATION="$PREFILL_GPU_MEMORY_UTILIZATION" KV_EVENTS_PORT="$PREFILL_KV_EVENTS_PORT" \
      SIDE_CHANNEL_HOST="${PREFILL_SIDE_CHANNEL_HOST:-}" SIDE_CHANNEL_PORT="$PREFILL_SIDE_CHANNEL_PORT" \
      bash "$SCRIPT_DIR/launch-kvbm-vllm.sh" >"$ARTIFACT_DIR/prefill.log" 2>&1 &
    PREFILL_PID=$!
    echo "[smoke] prefill pid=$PREFILL_PID"

    wait_for_frontend
    wait_for_frontend_chat_route
    wait_for_hub_prefill_frontend
    if [[ "$KVBM_SERVE_ONLY" == "1" ]]; then
      echo "SERVE_READY artifact_dir=$ARTIFACT_DIR url=http://$FRONTEND_HOST:$HTTP_PORT"
      wait
    fi
    run_client || fail "streaming chat request failed"
    wait_for_trace_evidence
    render_trace
    validate_useful_trace
    echo "SMOKE_DONE artifact_dir=$ARTIFACT_DIR trace=$ARTIFACT_DIR/trace.html"
    ;;
  *)
    echo "NODE_ROLE must be all, frontend, hub-prefill-frontend, hub, decode, prefill, client, or trace; got $NODE_ROLE" >&2
    exit 2
    ;;
esac
