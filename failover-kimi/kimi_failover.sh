#!/bin/bash
# Kimi-K2.6 TP8 two-engine intra-pod shadow failover (single-block scratch + autotune off).
# engine0 ACTIVE (holds flock, serves) + engine1 SHADOW (scratch-KV, STANDBY), both TP8
# on the same 8 GPUs, sharing GMS weights. Kill engine0 (WHOLE engine) -> engine1 wakes,
# remaps real KV, promotes, serves. EAGER=1 first pass, EAGER=0 (cudagraphs) second.
set -u
source /opt/dynamo/venv/bin/activate 2>/dev/null || true
export HF_HOME=/tmp/hf HF_HUB_OFFLINE=0   # writable for Kimi trust-remote-code dynamic modules
mkdir -p /tmp/hf
MODEL=${MODEL:-/tmp/kimi-k2.6-nvfp4}; SERVED=${SERVED:-kimi-k2.6}
TP=${TP:-8}; MML=${MML:-4096}; UTIL=${UTIL:-0.8}
SINGLE=${DYN_GMS_SCRATCH_SINGLE_BLOCK:-1}; NOAT=${DYN_NO_AUTOTUNE:-1}; EAGER=${EAGER:-1}
ACT_TO=${ACT_TO:-900}; SHA_TO=${SHA_TO:-900}; PROMO_TO=${PROMO_TO:-400}
EAGER_FLAG=""; [ "$EAGER" = "1" ] && EAGER_FLAG="--enforce-eager"
TAG=${TAG:-$( [ "$EAGER" = "1" ] && echo eager || echo graphs )}
OUT=/tmp/kimi_failover-$TAG; rm -rf "$OUT"; mkdir -p "$OUT/logs"; LOCK=$OUT/failover.lock
have(){ grep -aq "$2" "$1" 2>/dev/null; }
log(){ echo "[kimi_failover $(date +%T)] $*"; }
phase(){ echo "$(date +%s.%N),$1" >> "$OUT/phases.csv"; }   # epoch-stamped phase boundaries for mem windowing
mem(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | head -"$TP" | tr '\n' ' '; echo; }
cleanup(){ log "cleanup"; [ "${SAMP:-0}" != "0" ] && kill -9 "$SAMP" 2>/dev/null
  pkill -9 -f "[d]ynamo.vllm" 2>/dev/null; pkill -9 -f "[d]ynamo.frontend" 2>/dev/null
  pkill -9 -f "[E]ngineCore" 2>/dev/null; pkill -9 -f "[g]pu_memory_service" 2>/dev/null; sleep 2
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  rm -f /tmp/gms_*.sock; cp "$OUT"/logs/*.log "$OUT/" 2>/dev/null; }
trap cleanup EXIT
cleanup; sleep 4

# verify clean baseline
for a in $(seq 1 10); do
  dirty=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2+0>512{print}')
  [ -z "$dirty" ] && break; log "GPUs not clean, retry $a"; pkill -9 -f "[E]ngineCore" 2>/dev/null; sleep 5
done
log "baseline: $(mem)"

echo "epoch,label" > "$OUT/phases.csv"; phase baseline
# 1s device sampler (all GPUs)
( echo "ts,gpu,mem_mib" > "$OUT/dev_mem.csv"
  while true; do ts=$(date +%s.%N)
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', ' -v t="$ts" '{print t","$1","$2}' >> "$OUT/dev_mem.csv"; sleep 1; done ) & SAMP=$!

infer(){ curl -s -o "$OUT/infer_$1.json" -w "%{http_code}" -X POST http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"prompt\":\"The capital of France is\",\"max_tokens\":12,\"temperature\":0}"; }

launch(){ DYN_GMS_SCRATCH_SINGLE_BLOCK=$SINGLE DYN_NO_AUTOTUNE=$NOAT DYN_GMS_KVROUTE_V1=1 \
  ENGINE_ID=$1 FAILOVER_LOCK_PATH=$LOCK DYN_SYSTEM_PORT=$((8100+$1)) \
  VLLM_NIXL_SIDE_CHANNEL_PORT=$((5600+$1)) DYN_VLLM_KV_EVENT_PORT=$((20080+$1)) \
  nohup python3 -m dynamo.vllm --model "$MODEL" --served-model-name "$SERVED" -tp "$TP" \
  --trust-remote-code --max-model-len "$MML" --gpu-memory-utilization "$UTIL" $EAGER_FLAG \
  --load-format gms --gms-shadow-mode > "$2" 2>&1 & echo $!; }

log "GMS weights+kv servers (TP=$TP)"
for d in $(seq 0 $((TP-1))); do
  python3 -m gpu_memory_service --device $d --tag weights  > "$OUT/logs/gms_w$d.log"  2>&1 &
  python3 -m gpu_memory_service --device $d --tag kv_cache > "$OUT/logs/gms_kv$d.log" 2>&1 &
done
for d in $(seq 0 $((TP-1))); do for i in $(seq 1 60); do have "$OUT/logs/gms_w$d.log" "Server started" && break; sleep 1; done; done
nohup python3 -m dynamo.frontend > "$OUT/logs/frontend.log" 2>&1 &

log "engine0 ACTIVE launching (single_block=$SINGLE no_autotune=$NOAT eager=$EAGER)"
phase active_launch
A=$(launch 0 "$OUT/logs/engine0.log"); log "engine0 pid $A"
for i in $(seq 1 $ACT_TO); do have "$OUT/logs/engine0.log" "Registered endpoint" && { log "ACTIVE registered +${i}s"; break; }
  kill -0 $A 2>/dev/null || { log "engine0 DIED"; grep -anE "Error|OutOfMemory|Traceback|CUDA error|non-positive" "$OUT/logs/engine0.log"|tail -10; exit 1; }; sleep 1; done
have "$OUT/logs/engine0.log" "Registered endpoint" || { log "ACTIVE never registered (timeout)"; exit 1; }
phase active_registered; sleep 3; phase active_resting
log "mem after ACTIVE: $(mem)"

log "engine1 SHADOW launching"
phase shadow_launch
B=$(launch 1 "$OUT/logs/engine1.log"); log "engine1 pid $B"
for i in $(seq 1 $SHA_TO); do have "$OUT/logs/engine1.log" "waiting for lock" && { log "SHADOW standby +${i}s"; break; }
  kill -0 $B 2>/dev/null || { log "engine1 DIED"; grep -anE "Error|OutOfMemory|Traceback|CUDA error|non-positive" "$OUT/logs/engine1.log"|tail -10; exit 1; }; sleep 1; done
have "$OUT/logs/engine1.log" "waiting for lock" || { log "SHADOW never reached standby (timeout)"; grep -anE "Error|OutOfMemory|Traceback" "$OUT/logs/engine1.log"|tail -10; exit 1; }
phase shadow_standby
echo "=== shadow [D3] scratch_map (expect 1 granule/GPU) ==="; grep -a "\[D3\] scratch_map" "$OUT/logs/engine1.log" | head -3
sleep 3; phase colocated_resting
log "MEM COLOCATED (active+shadow): $(mem)"

for i in $(seq 1 60); do have "$OUT/logs/frontend.log" "Completions is ready" && break; sleep 1; done
log "infer ACTIVE: HTTP $(infer active) :: $(head -c 120 "$OUT/infer_active.json")"

log "KILL engine0 (whole engine: children first, then parent) -> FAILOVER"
phase active_kill
pkill -9 -P "$A" 2>/dev/null; kill -9 "$A" 2>/dev/null
for i in $(seq 1 $PROMO_TO); do have "$OUT/logs/engine1.log" "Registered endpoint" && { log "SHADOW PROMOTED +${i}s"; break; }
  kill -0 $B 2>/dev/null || { log "engine1 DIED on promotion"; grep -anE "Error|OutOfMemory|Traceback|CUDA error" "$OUT/logs/engine1.log"|tail -10; break; }; sleep 1; done
phase shadow_promoted
log "MEM after failover: $(mem)"
# Post-failover inference: the frontend must evict the dead active's discovery
# entry and route to the promoted engine. Retry with backoff (connection-refused
# to the dead active is expected for the first few seconds).
PF=000; PFR=0
for r in $(seq 1 18); do
  sleep 5
  code=$(infer postfail)
  log "  post-failover retry $r (+$((r*5))s): HTTP $code :: $(head -c 90 "$OUT/infer_postfail.json")"
  [ "$code" = "200" ] && { PF=200; PFR=$((r*5)); log "POST-FAILOVER 200 at +${PFR}s after promotion"; break; }
done
phase final; log "MEM final: $(mem)"
echo "KIMI_FAILOVER_DONE tag=$TAG single_block=$SINGLE no_autotune=$NOAT eager=$EAGER postfail=$PF postfail_secs=$PFR"
