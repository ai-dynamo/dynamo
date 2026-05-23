#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/hardware-profiles.sh"

ARTIFACT_DIR=${ARTIFACT_DIR:?ARTIFACT_DIR is required}
NODE_ROLE=${NODE_ROLE:-all}
case "$NODE_ROLE" in
  frontend|client) profile_scope=model-only ;;
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
VENV=${VENV:-/opt/dynamo/venv}
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

: "${MODEL:?MODEL must be set by KVBM_HARDWARE_PROFILE or env override}"
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
    DECODE_CUDA_VISIBLE_DEVICES=${DECODE_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}
    : "${DECODE_CUDA_VISIBLE_DEVICES:?DECODE_CUDA_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES is required for NODE_ROLE=decode}"
    ;;
  prefill)
    : "${GPU_MEMORY_UTILIZATION:?GPU_MEMORY_UTILIZATION must be set by KVBM_HARDWARE_PROFILE or env override}"
    : "${PREFILL_GPU_MEMORY_UTILIZATION:?PREFILL_GPU_MEMORY_UTILIZATION must be set by KVBM_HARDWARE_PROFILE or env override}"
    : "${MAX_MODEL_LEN:?MAX_MODEL_LEN must be set by KVBM_HARDWARE_PROFILE or env override}"
    : "${MAX_NUM_SEQS:?MAX_NUM_SEQS must be set by KVBM_HARDWARE_PROFILE or env override}"
    : "${CPU_CACHE_GB:?CPU_CACHE_GB must be set by KVBM_HARDWARE_PROFILE or env override}"
    PREFILL_CUDA_VISIBLE_DEVICES=${PREFILL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}
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

cd "$WORKTREE"

export DYN_DISCOVERY_BACKEND=${DYN_DISCOVERY_BACKEND:-etcd}
export DYN_REQUEST_PLANE=${DYN_REQUEST_PLANE:-tcp}
export DYN_EVENT_PLANE=${DYN_EVENT_PLANE:-nats}
export ETCD_ENDPOINTS=${ETCD_ENDPOINTS:-127.0.0.1:2379}
export NATS_SERVER=${NATS_SERVER:-nats://127.0.0.1:4222}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

cat >"$ARTIFACT_DIR/metadata.env" <<EOF
timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
experiment=E
topology=dynamo-native-kvbm-disagg
hardware_profile=$KVBM_HARDWARE_PROFILE
gpu_class=$GPU_CLASS
node_role=$NODE_ROLE
hostname=$(hostname)
worktree=$WORKTREE
model=$MODEL
framework=dynamo.vllm
frontend=dynamo.frontend
raw_vllm=false
endpoint_type=chat
streaming=true
namespace=$NAMESPACE
http_port=$HTTP_PORT
frontend_host=$FRONTEND_HOST
discovery_backend=$DYN_DISCOVERY_BACKEND
request_plane=$DYN_REQUEST_PLANE
event_plane=$DYN_EVENT_PLANE
etcd_endpoints=$ETCD_ENDPOINTS
nats_server=$NATS_SERVER
max_model_len=${MAX_MODEL_LEN:-}
max_num_seqs=${MAX_NUM_SEQS:-}
cpu_cache_gb=${CPU_CACHE_GB:-}
enforce_eager=$ENFORCE_EAGER
vllm_runner=$VLLM_RUNNER
kvbm_connector_module_path=$KVBM_CONNECTOR_MODULE_PATH
prefill_kv_connector_wrapper=$PREFILL_KV_CONNECTOR_WRAPPER
decode_cuda_visible_devices=${DECODE_CUDA_VISIBLE_DEVICES:-}
prefill_cuda_visible_devices=${PREFILL_CUDA_VISIBLE_DEVICES:-}
decode_side_channel_host=${DECODE_SIDE_CHANNEL_HOST:-}
prefill_side_channel_host=${PREFILL_SIDE_CHANNEL_HOST:-}
trace_renderer=.claude/skills/disagg-trace/cd-trace.py
start_local_infra=$START_LOCAL_INFRA
infra_host=$INFRA_HOST
prefetch_model=$PREFETCH_MODEL
isolate_hf_cache=$ISOLATE_HF_CACHE
hf_home=${HF_HOME:-}
hf_hub_cache=${HF_HUB_CACHE:-}
transformers_cache=${TRANSFORMERS_CACHE:-}
EOF

cleanup() {
  set +e
  pkill -f "dynamo.frontend" 2>/dev/null
  pkill -f "dynamo.vllm" 2>/dev/null
  pkill -f "EngineCore" 2>/dev/null
  if [[ -n "${ETCD_PID:-}" ]]; then kill "$ETCD_PID" 2>/dev/null; fi
  if [[ -n "${NATS_PID:-}" ]]; then kill "$NATS_PID" 2>/dev/null; fi
  set -e
}

fail() {
  echo "SMOKE_FAIL: $*" >&2
  for log in frontend decode prefill client; do
    if [[ -f "$ARTIFACT_DIR/$log.log" ]]; then
      echo "--- tail $log.log ---" >&2
      tail -n 100 "$ARTIFACT_DIR/$log.log" | sed 's/\x1b\[[0-9;]*m//g' >&2
    fi
  done
  exit 1
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
  [[ "$PREFETCH_MODEL" == "1" ]] || return 0

  echo "[smoke] prefetch model cache for $MODEL"
  "$VENV/bin/python" - "$MODEL" "$ARTIFACT_DIR/model-snapshot-path.txt" <<'PY'
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

wait_for_frontend() {
  local deadline=$(( $(date +%s) + READY_TIMEOUT ))
  local status_logged=0
  until curl -fsS -m 5 "http://$FRONTEND_HOST:$HTTP_PORT/v1/models" >"$ARTIFACT_DIR/models.json" 2>"$ARTIFACT_DIR/models.err" \
    && "$VENV/bin/python" - "$ARTIFACT_DIR/models.json" "$MODEL" <<'PY'
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

run_client() {
  echo "[smoke] streaming chat request"
  "$VENV/bin/python" - "$FRONTEND_HOST" "$HTTP_PORT" "$MODEL" "$ARTIFACT_DIR/chat-stream.jsonl" \
    >"$ARTIFACT_DIR/chat-ttft.json" <<'PY'
import json
import sys
import time
import urllib.request

host, port, model, out_path = sys.argv[1:5]
payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": (
                "Explain in four concise sentences why KV cache transfer "
                "matters for disaggregated LLM serving."
            ),
        }
    ],
    "max_tokens": 96,
    "temperature": 0,
    "stream": True,
}
url = f"http://{host}:{port}/v1/chat/completions"
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
with urllib.request.urlopen(req, timeout=300) as resp, open(out_path, "w") as out:
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
        delta = obj["choices"][0].get("delta", {})
        text = delta.get("content") or ""
        if text and first_token is None:
            first_token = now
        content_chars += len(text)
end = time.perf_counter()
print(json.dumps({
    "model": model,
    "framework": "dynamo.vllm",
    "endpoint_type": "chat",
    "streaming": True,
    "ttfb_seconds": None if first_byte is None else first_byte - start,
    "ttft_seconds": None if first_token is None else first_token - start,
    "total_seconds": end - start,
    "chunks": chunks,
    "content_chars": content_chars,
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
  "$VENV/bin/python" "$WORKTREE/.claude/skills/disagg-trace/cd-trace.py" "$ARTIFACT_DIR" \
    | tee "$ARTIFACT_DIR/trace-render.log"
}

common_env=(
  WORKTREE="$WORKTREE"
  VENV="$VENV"
  KVBM_HARDWARE_PROFILE="$KVBM_HARDWARE_PROFILE"
  GPU_CLASS="$GPU_CLASS"
  MODEL="$MODEL"
  NAMESPACE="$NAMESPACE"
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
  CPU_CACHE_GB="${CPU_CACHE_GB:-}"
  ENFORCE_EAGER="$ENFORCE_EAGER"
  VLLM_RUNNER="$VLLM_RUNNER"
  VLLM_USE_AOT_COMPILE="${VLLM_USE_AOT_COMPILE:-0}"
  VLLM_USE_STANDALONE_COMPILE="${VLLM_USE_STANDALONE_COMPILE:-0}"
  VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
  KVBM_CONNECTOR_MODULE_PATH="$KVBM_CONNECTOR_MODULE_PATH"
  PREFILL_KV_CONNECTOR_WRAPPER="$PREFILL_KV_CONNECTOR_WRAPPER"
  PYTHONUNBUFFERED=1
)

case "$NODE_ROLE" in
  frontend)
    start_local_infra
    exec env "${common_env[@]}" HTTP_PORT="$HTTP_PORT" \
      bash "$SCRIPT_DIR/launch-dynamo-frontend.sh"
    ;;
  decode)
    prefetch_model
    exec env "${common_env[@]}" ROLE=decode CUDA_VISIBLE_DEVICES="${DECODE_CUDA_VISIBLE_DEVICES:-}" \
      GPU_MEMORY_UTILIZATION="$DECODE_GPU_MEMORY_UTILIZATION" \
      SIDE_CHANNEL_HOST="${DECODE_SIDE_CHANNEL_HOST:-}" SIDE_CHANNEL_PORT="$DECODE_SIDE_CHANNEL_PORT" \
      bash "$SCRIPT_DIR/launch-kvbm-vllm.sh"
    ;;
  prefill)
    prefetch_model
    exec env "${common_env[@]}" ROLE=prefill CUDA_VISIBLE_DEVICES="${PREFILL_CUDA_VISIBLE_DEVICES:-}" \
      GPU_MEMORY_UTILIZATION="$PREFILL_GPU_MEMORY_UTILIZATION" KV_EVENTS_PORT="$PREFILL_KV_EVENTS_PORT" \
      SIDE_CHANNEL_HOST="${PREFILL_SIDE_CHANNEL_HOST:-}" SIDE_CHANNEL_PORT="$PREFILL_SIDE_CHANNEL_PORT" \
      bash "$SCRIPT_DIR/launch-kvbm-vllm.sh"
    ;;
  client)
    wait_for_frontend
    run_client
    render_trace
    echo "SMOKE_DONE artifact_dir=$ARTIFACT_DIR trace=$ARTIFACT_DIR/trace.html"
    ;;
  all)
    trap cleanup EXIT
    echo "[smoke] cleanup stale Dynamo processes"
    cleanup
    sleep 2
    start_local_infra
    prefetch_model

    echo "[smoke] start frontend"
    env "${common_env[@]}" HTTP_PORT="$HTTP_PORT" \
      bash "$SCRIPT_DIR/launch-dynamo-frontend.sh" >"$ARTIFACT_DIR/frontend.log" 2>&1 &
    FRONTEND_PID=$!
    echo "[smoke] frontend pid=$FRONTEND_PID"

    echo "[smoke] start decode dynamo.vllm"
    env "${common_env[@]}" ROLE=decode CUDA_VISIBLE_DEVICES="$DECODE_CUDA_VISIBLE_DEVICES" \
      GPU_MEMORY_UTILIZATION="$DECODE_GPU_MEMORY_UTILIZATION" \
      SIDE_CHANNEL_HOST="${DECODE_SIDE_CHANNEL_HOST:-}" SIDE_CHANNEL_PORT="$DECODE_SIDE_CHANNEL_PORT" \
      bash "$SCRIPT_DIR/launch-kvbm-vllm.sh" >"$ARTIFACT_DIR/decode.log" 2>&1 &
    DECODE_PID=$!
    echo "[smoke] decode pid=$DECODE_PID"

    echo "[smoke] start prefill dynamo.vllm"
    env "${common_env[@]}" ROLE=prefill CUDA_VISIBLE_DEVICES="$PREFILL_CUDA_VISIBLE_DEVICES" \
      GPU_MEMORY_UTILIZATION="$PREFILL_GPU_MEMORY_UTILIZATION" KV_EVENTS_PORT="$PREFILL_KV_EVENTS_PORT" \
      SIDE_CHANNEL_HOST="${PREFILL_SIDE_CHANNEL_HOST:-}" SIDE_CHANNEL_PORT="$PREFILL_SIDE_CHANNEL_PORT" \
      bash "$SCRIPT_DIR/launch-kvbm-vllm.sh" >"$ARTIFACT_DIR/prefill.log" 2>&1 &
    PREFILL_PID=$!
    echo "[smoke] prefill pid=$PREFILL_PID"

    wait_for_frontend
    run_client
    render_trace
    echo "SMOKE_DONE artifact_dir=$ARTIFACT_DIR trace=$ARTIFACT_DIR/trace.html"
    ;;
  *)
    echo "NODE_ROLE must be all, frontend, decode, prefill, or client; got $NODE_ROLE" >&2
    exit 2
    ;;
esac
