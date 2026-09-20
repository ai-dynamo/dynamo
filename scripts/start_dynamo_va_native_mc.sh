#!/usr/bin/env bash
# Dynamo native on Ascend: frontend + dynamo.vllm BOTH inside container vllm-ascend-wm (0.26).
# Integrated with Mooncake TE for KV Cache transfer.
#
# Prereqs:
#   - bash scripts/ascend/start_docker_va.sh
#   - bash scripts/ascend/start_etcd.sh   # skip if DISCOVERY=file
#   - host-built runtime under $WM_ROOT/dynamo-ascend (see bringup doc)
#   - model under /data/models/... (mounted via /data)
#   - mooncake_master binary available in container PATH
#
# Mooncake config (mooncake_config.json) is auto-generated inside the container
# at $MOONCAKE_CONFIG_PATH; no pre-made config file is needed. Tune via:
#   MOONCAKE_RPC_PORT / MC_MASTER_ADDRESS / MC_GLOBAL_SEGMENT_SIZE / MC_LOCAL_BUFFER_SIZE
#
# Usage (from host):
#   bash scripts/ascend/start_dynamo_va_native.sh
#   RESTART=1 bash scripts/ascend/start_dynamo_va_native.sh
#   DISCOVERY=file bash scripts/ascend/start_dynamo_va_native.sh
#   ETCD_ENDPOINTS=http://10.x.x.x:2379 bash scripts/ascend/start_dynamo_va_native.sh
#   WM_ROOT=/data/wm bash scripts/ascend/start_dynamo_va_native.sh
#
# Verify:
#   curl -s localhost:8000/v1/models
#
# Logs: $WM_ROOT/dynamo-native-logs/{frontend,worker,mooncake}.log
# Stop:  bash scripts/ascend/start_dynamo_va_native.sh stop
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WM_ROOT=${WM_ROOT:-/data/wm}

NAME=${NAME:-vllm-ascend-wm}
PORT=${PORT:-8000}
MODEL=${MODEL:-/data/models/Qwen3.8-27B}
SERVED_NAME=${SERVED_NAME:-qwen}
TP=${TP:-4}
DP=${DP:-2}
DISCOVERY=${DISCOVERY:-etcd}   # etcd | file
ETCD_ENDPOINTS=${ETCD_ENDPOINTS:-http://127.0.0.1:2379}
STORE=${STORE:-$WM_ROOT/dynamo_store_kv}
LOGDIR=${LOGDIR:-$WM_ROOT/dynamo-native-logs}
SITE_PACKAGES=/usr/local/python3.12.13/lib/python3.12/site-packages
RUNTIME_SRC=${RUNTIME_SRC:-$WM_ROOT/dynamo-ascend/lib/bindings/python/src}
COMPONENTS_SRC=${COMPONENTS_SRC:-$WM_ROOT/dynamo-ascend/components/src}
RESTART=${RESTART:-0}

# Mooncake Configuration
MOONCAKE_RPC_PORT=${MOONCAKE_RPC_PORT:-50058}
MOONCAKE_CONFIG_PATH=${MOONCAKE_CONFIG_PATH:-$WM_ROOT/mooncake_conf/mooncake_config.json}
MC_MASTER_ADDRESS=${MC_MASTER_ADDRESS:-127.0.0.1}
MC_GLOBAL_SEGMENT_SIZE=${MC_GLOBAL_SEGMENT_SIZE:-4294967296}
MC_LOCAL_BUFFER_SIZE=${MC_LOCAL_BUFFER_SIZE:-4294967296}

ensure_container() {
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "container $NAME not running; starting via start_docker_va.sh"
    bash "$SCRIPT_DIR/start_docker_va.sh"
  fi
}

ensure_etcd() {
  if [ "$DISCOVERY" = "file" ]; then
    return 0
  fi
  bash "$SCRIPT_DIR/start_etcd.sh"
}

ensure_pth() {
  docker exec "$NAME" bash -lc "
    set -e
    sp='$SITE_PACKAGES'
    echo '$RUNTIME_SRC' > \"\$sp/dynamo_ascend_runtime.pth\"
    echo '$COMPONENTS_SRC' > \"\$sp/dynamo_ascend_components.pth\"
    python3 -c 'import dynamo._core, dynamo.vllm; print(\"dynamo ok\", dynamo._core.__file__)'
  "
}

stop_inside() {
  docker exec "$NAME" bash -lc '
    set +e
    pkill -f "python3 -m dynamo.frontend" 2>/dev/null || true
    pkill -f "python3 -m dynamo.vllm" 2>/dev/null || true
    pkill -f "mooncake_master" 2>/dev/null || true
    # orphans keep holding NPU after parent dies
    pkill -9 -f "VLLM::" 2>/dev/null || true
    sleep 2
    pgrep -af "dynamo.frontend|dynamo.vllm|VLLM::|mooncake" | grep -v pgrep || echo "stopped"
  ' || true
}

if [ "${1:-}" = "stop" ]; then
  ensure_container
  stop_inside
  exit 0
fi

if [ "$DISCOVERY" != "etcd" ] && [ "$DISCOVERY" != "file" ]; then
  echo "DISCOVERY must be etcd or file, got: $DISCOVERY" >&2
  exit 2
fi

ensure_container
ensure_etcd
ensure_pth

if [ "$RESTART" = "1" ]; then
  stop_inside
fi

mkdir -p "$STORE" "$LOGDIR"

# Generate mooncake_config.json inside the container so this script is self-contained.
MC_CONF_DIR=$(dirname "$MOONCAKE_CONFIG_PATH")
docker exec "$NAME" bash -lc "
set -e
mkdir -p '$MC_CONF_DIR'
cat > '$MOONCAKE_CONFIG_PATH' <<'EOF'
{
  \"protocol\": \"ascend\",
  \"use_ascend_direct\": true,
  \"master_server_address\": \"$MC_MASTER_ADDRESS:$MOONCAKE_RPC_PORT\",
  \"global_segment_size\": $MC_GLOBAL_SEGMENT_SIZE,
  \"local_buffer_size\": $MC_LOCAL_BUFFER_SIZE
}
EOF
echo \"generated mooncake config at '$MOONCAKE_CONFIG_PATH':\"
cat '$MOONCAKE_CONFIG_PATH'
"

: > "$LOGDIR/frontend.log"
: > "$LOGDIR/worker.log"
: > "$LOGDIR/mooncake.log"

if [ ! -d "$STORE" ]; then
  sysctl -w fs.inotify.max_user_watches=524288 >/dev/null 2>&1 || true
fi

if [ "$DISCOVERY" = "etcd" ]; then
  DISCOVERY_ENV="export DYN_DISCOVERY_BACKEND=etcd ETCD_ENDPOINTS='$ETCD_ENDPOINTS'"
else
  DISCOVERY_ENV="export DYN_DISCOVERY_BACKEND=file DYN_FILE_KV='$STORE'"
fi

# --- 核心改动：docker exec 的双引号已改回 ---
docker exec -d "$NAME" bash -lc "
set -e
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# --- Mooncake Integration Start ---
export MOONCAKE_CONFIG_PATH='$MOONCAKE_CONFIG_PATH'

if ! ss -tlnp | grep -q \":$MOONCAKE_RPC_PORT \"; then
  echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] Starting mooncake_master on port $MOONCAKE_RPC_PORT...\" >> '$LOGDIR/mooncake.log'
  nohup mooncake_master \
    --rpc_port $MOONCAKE_RPC_PORT \
    --eviction_high_watermark_ratio 0.9 \
    --rpc_thread_num 32 \
    >> '$LOGDIR/mooncake.log' 2>&1 &
  MC_PID=\$!
  echo \$MC_PID > '$LOGDIR/mooncake.pid'
  sleep 2
  if kill -0 \$MC_PID 2>/dev/null; then
    echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] mooncake_master started successfully (PID: \$MC_PID)\" >> '$LOGDIR/mooncake.log'
  else
    echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] ERROR: mooncake_master failed to start, check log above\" >> '$LOGDIR/mooncake.log'
    exit 1
  fi
else
  EXISTING_PID=\$(ss -tlnp | grep \":$MOONCAKE_RPC_PORT \" | grep -oP 'pid=\\K[0-9]+' | head -1)
  echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] mooncake_master already running on port $MOONCAKE_RPC_PORT (PID: \$EXISTING_PID), skipping startup\" >> '$LOGDIR/mooncake.log'
fi
# --- Mooncake Integration End ---

$DISCOVERY_ENV
cd /tmp

nohup python3 -m dynamo.frontend \
  --http-port $PORT \
  --discovery-backend $DISCOVERY \
  --request-plane tcp \
  --response-plane tcp \
  > '$LOGDIR/frontend.log' 2>&1 &
echo \$! > '$LOGDIR/frontend.pid'

sleep 2

nohup python3 -m dynamo.vllm \
  --model '$MODEL' \
  --served-model-name '$SERVED_NAME' \
  --tensor-parallel-size $TP \
  --data-parallel-size $DP \
  --discovery-backend $DISCOVERY \
  --request-plane tcp \
  --response-plane tcp \
  --disaggregation-mode agg \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --dyn-tool-call-parser qwen3_coder \
  --kv-transfer-config '{\"kv_connector\": \"MooncakeConnector\", \"kv_role\": \"kv_both\"}' \
  > '$LOGDIR/worker.log' 2>&1 &
echo \$! > '$LOGDIR/worker.pid'
"

echo "started inside $NAME (FE :$PORT, worker TP${TP}xDP${DP}, model=$SERVED_NAME, discovery=$DISCOVERY, mooncake=$MOONCAKE_RPC_PORT)"
if [ "$DISCOVERY" = "etcd" ]; then
  echo "ETCD_ENDPOINTS=$ETCD_ENDPOINTS"
fi
echo "waiting for model registry (worker load can take several minutes)..."

ok=0
for i in $(seq 1 180); do
  if curl -sf "localhost:$PORT/v1/models" 2>/dev/null | grep -q "\"$SERVED_NAME\""; then
    ok=1
    break
  fi
  if ! curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && ! curl -sf "localhost:$PORT/v1/models" >/dev/null 2>&1; then
    if [ "$i" -ge 6 ]; then
      echo "frontend not responding; last frontend log:"
      tail -40 "$LOGDIR/frontend.log" || true
      exit 1
    fi
  fi
  sleep 5
done

if [ "$ok" = "1" ]; then
  curl -s "localhost:$PORT/v1/models"; echo
  echo "OK  frontend.pid=$(cat "$LOGDIR/frontend.pid" 2>/dev/null) worker.pid=$(cat "$LOGDIR/worker.pid" 2>/dev/null) mooncake.pid=$(cat "$LOGDIR/mooncake.pid" 2>/dev/null)"
  echo "logs: $LOGDIR/"
else
  echo "TIMEOUT waiting for $SERVED_NAME on :$PORT — check $LOGDIR/worker.log"
  docker exec "$NAME" bash -lc "pgrep -af 'dynamo.frontend|dynamo.vllm' | grep -v pgrep || true"
  tail -40 "$LOGDIR/frontend.log" || true
  echo "--- mooncake log ---"
  tail -40 "$LOGDIR/mooncake.log" || true
  exit 1
fi
