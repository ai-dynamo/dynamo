#!/bin/bash
# LAYER 0 — GMS RW->RO weight commit/import in isolation (no shadow, no scratch-KV,
# no flock, no failover). Validates the weight-sharing half of failover:
#   A) engine A (--load-format gms) loads disk weights -> GMS -> commits -> serves 200
#   B) kill A (weights persist in GMS servers)
#   C) engine B (--load-format gms) connects RO -> materializes committed weights -> serves 200
# Needs only the meta-safety fix (RO meta-construction). Kimi-K2.6 TP8.
set -u
source /opt/dynamo/venv/bin/activate 2>/dev/null || true
export HF_HOME=/tmp/hf HF_HUB_OFFLINE=0; mkdir -p /tmp/hf
MODEL=/tmp/kimi-k2.6-nvfp4; SERVED=kimi-k2.6; TP=8; MML=4096; UTIL=0.8
NOAT=${DYN_NO_AUTOTUNE:-1}; EAGER=${EAGER:-1}
EAGER_FLAG=""; [ "$EAGER" = "1" ] && EAGER_FLAG="--enforce-eager"
OUT=/tmp/kimi_rwro; rm -rf "$OUT"; mkdir -p "$OUT/logs"
have(){ grep -aq "$2" "$1" 2>/dev/null; }
log(){ echo "[rwro $(date +%T)] $*"; }
mem(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | head -"$TP" | tr '\n' ' '; echo; }
cleanup(){ pkill -9 -f "[d]ynamo.vllm" 2>/dev/null; pkill -9 -f "[d]ynamo.frontend" 2>/dev/null
  pkill -9 -f "[E]ngineCore" 2>/dev/null; pkill -9 -f "[g]pu_memory_service" 2>/dev/null; sleep 2
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  rm -f /tmp/gms_*.sock; cp "$OUT"/logs/*.log "$OUT/" 2>/dev/null; }
trap cleanup EXIT
cleanup; sleep 4
infer(){ curl -s -o "$OUT/infer_$1.json" -w "%{http_code}" -X POST http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"prompt\":\"The capital of France is\",\"max_tokens\":12,\"temperature\":0}"; }
launch(){ DYN_NO_AUTOTUNE=$NOAT DYN_SYSTEM_PORT=$((8100+$1)) \
  VLLM_NIXL_SIDE_CHANNEL_PORT=$((5600+$1)) DYN_VLLM_KV_EVENT_PORT=$((20080+$1)) \
  nohup python3 -m dynamo.vllm --model "$MODEL" --served-model-name "$SERVED" -tp "$TP" \
  --trust-remote-code --max-model-len "$MML" --gpu-memory-utilization "$UTIL" $EAGER_FLAG \
  --load-format gms > "$2" 2>&1 & echo $!; }

log "GMS weights servers (TP=$TP)"
for d in $(seq 0 $((TP-1))); do python3 -m gpu_memory_service --device $d --tag weights > "$OUT/logs/gms_w$d.log" 2>&1 & done
for d in $(seq 0 $((TP-1))); do for i in $(seq 1 90); do have "$OUT/logs/gms_w$d.log" "Server started" && break; sleep 1; done; done
nohup python3 -m dynamo.frontend > "$OUT/logs/frontend.log" 2>&1 &

log "=== A) RW engine: disk -> GMS -> commit -> serve ==="
A=$(launch 0 "$OUT/logs/engineA.log"); log "engineA pid $A"
for i in $(seq 1 700); do have "$OUT/logs/engineA.log" "Registered endpoint" && { log "RW registered +${i}s"; break; }
  kill -0 $A 2>/dev/null || { log "engineA DIED"; grep -anE "Error|Traceback|Cannot copy|meta tensor|non-positive|routed through scratch" "$OUT/logs/engineA.log" | tail -10; exit 1; }; sleep 1; done
have "$OUT/logs/engineA.log" "Registered endpoint" || { log "RW never registered"; exit 1; }
for i in $(seq 1 60); do have "$OUT/logs/frontend.log" "Completions is ready" && break; sleep 1; done
log "RW mem: $(mem)"
log "RW infer: HTTP $(infer rw) :: $(head -c 120 "$OUT/infer_rw.json")"

log "=== kill A (committed weights must survive in GMS servers) ==="
pkill -9 -P "$A" 2>/dev/null; kill -9 "$A" 2>/dev/null; sleep 8
log "post-kill mem (weights held by GMS servers): $(mem)"

log "=== B) RO engine: import committed weights -> serve ==="
B=$(launch 1 "$OUT/logs/engineB.log"); log "engineB pid $B"
for i in $(seq 1 700); do have "$OUT/logs/engineB.log" "Registered endpoint" && { log "RO registered +${i}s"; break; }
  kill -0 $B 2>/dev/null || { log "engineB DIED"; grep -anE "Error|Traceback|Cannot copy|meta tensor|non-positive|Stale|routed through scratch" "$OUT/logs/engineB.log" | tail -12; exit 1; }; sleep 1; done
have "$OUT/logs/engineB.log" "Registered endpoint" || { log "RO never registered"; exit 1; }
sleep 3
log "RO mem: $(mem)"
log "RO infer: HTTP $(infer ro) :: $(head -c 120 "$OUT/infer_ro.json")"
echo "KIMI_RWRO_DONE (RW=$(cat $OUT/infer_rw.json 2>/dev/null | head -c 0)rw ro)"
