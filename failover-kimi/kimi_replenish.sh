#!/bin/bash
# POST-FAILOVER SHADOW REPLENISHMENT — fix the "fail-once" limitation.
# Sequence (TP8, deterministic serialized launches so roles are unambiguous):
#   P2  engine0 ACTIVE + engine1 SHADOW (parked at standby)         -> 1a+1s
#   P3  FAILOVER-1: kill engine0 -> engine1 PROMOTES -> serves 200  -> 1a (no shadow)
#   P4  REPLENISH: launch engineC (ENGINE_ID=0 reused) -> it races the flock, loses
#       to the live active (engine1), imports RO weights, sets up scratch-KV, PARKS
#       at standby -> restores 1a+1s WHILE engine1 keeps serving
#   P5  FAILOVER-2: kill engine1 -> engineC PROMOTES -> serves 200  -> proves a 2nd
#       failover works on the replenished shadow
# The untested crux = can a NEW shadow acquire a scratch-KV grant + RO weights from
# the still-alive GMS servers AFTER a promotion (active holds kv_cache RW).
set -u
source /opt/dynamo/venv/bin/activate 2>/dev/null || true
export HF_HOME=/tmp/hf HF_HUB_OFFLINE=0; mkdir -p /tmp/hf
MODEL=${MODEL:-/tmp/kimi-k2.6-nvfp4}; SERVED=${SERVED:-kimi-k2.6}
TP=${TP:-8}; MML=${MML:-4096}; UTIL=${UTIL:-0.8}
SINGLE=${DYN_GMS_SCRATCH_SINGLE_BLOCK:-1}; NOAT=${DYN_NO_AUTOTUNE:-1}; EAGER=${EAGER:-1}
ACT_TO=${ACT_TO:-900}; SHA_TO=${SHA_TO:-900}; PROMO_TO=${PROMO_TO:-400}
EAGER_FLAG=""; [ "$EAGER" = "1" ] && EAGER_FLAG="--enforce-eager"
TAG=${TAG:-replenish-$( [ "$EAGER" = "1" ] && echo eager || echo graphs )}
OUT=/tmp/kimi_failover-$TAG; rm -rf "$OUT"; mkdir -p "$OUT/logs"; LOCK=$OUT/failover.lock
have(){ grep -aq "$2" "$1" 2>/dev/null; }
log(){ echo "[replenish $(date +%T)] $*"; }
phase(){ echo "$(date +%s.%N),$1" >> "$OUT/phases.csv"; }
mem(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | head -"$TP" | tr '\n' ' '; echo; }
cleanup(){ log "cleanup"; [ "${SAMP:-0}" != "0" ] && kill -9 "$SAMP" 2>/dev/null
  pkill -9 -f "[d]ynamo.vllm" 2>/dev/null; pkill -9 -f "[d]ynamo.frontend" 2>/dev/null
  pkill -9 -f "[E]ngineCore" 2>/dev/null; pkill -9 -f "[g]pu_memory_service" 2>/dev/null; sleep 2
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  rm -f /tmp/gms_*.sock; cp "$OUT"/logs/*.log "$OUT/" 2>/dev/null; }
trap cleanup EXIT
cleanup; sleep 4
for a in $(seq 1 10); do
  dirty=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2+0>512{print}')
  [ -z "$dirty" ] && break; log "GPUs not clean, retry $a"; pkill -9 -f "[E]ngineCore" 2>/dev/null; sleep 5
done
echo "epoch,label" > "$OUT/phases.csv"; phase baseline
log "baseline: $(mem)"
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
# wait for a marker in a log, dying-engine aware. $1=log $2=marker $3=timeout $4=pid $5=label
wait_for(){ local L=$1 M=$2 TO=$3 PID=$4 LB=$5
  for i in $(seq 1 $TO); do have "$L" "$M" && { log "$LB +${i}s"; return 0; }
    kill -0 "$PID" 2>/dev/null || { log "$LB: ENGINE DIED"; grep -anE "Error|OutOfMemory|Traceback|CUDA error|not validated|non-positive|routed through scratch" "$L"|tail -12; return 1; }
    sleep 1; done; log "$LB: TIMEOUT waiting '$M'"; return 1; }
kill_engine(){ pkill -9 -P "$1" 2>/dev/null; kill -9 "$1" 2>/dev/null; }

log "GMS weights+kv servers (TP=$TP)"
for d in $(seq 0 $((TP-1))); do
  python3 -m gpu_memory_service --device $d --tag weights  > "$OUT/logs/gms_w$d.log"  2>&1 &
  python3 -m gpu_memory_service --device $d --tag kv_cache > "$OUT/logs/gms_kv$d.log" 2>&1 &
done
for d in $(seq 0 $((TP-1))); do for i in $(seq 1 60); do have "$OUT/logs/gms_w$d.log" "Server started" && break; sleep 1; done; done
nohup python3 -m dynamo.frontend > "$OUT/logs/frontend.log" 2>&1 &
for i in $(seq 1 60); do have "$OUT/logs/frontend.log" "Completions is ready" && break; sleep 1; done

# ---- P2: engine0 ACTIVE + engine1 SHADOW ----
log "P2 engine0 ACTIVE launching"; phase e0_launch
A=$(launch 0 "$OUT/logs/engine0.log")
wait_for "$OUT/logs/engine0.log" "Registered endpoint" $ACT_TO "$A" "ACTIVE(e0) registered" || exit 1
phase e0_active; log "mem after e0 ACTIVE: $(mem)"
log "P2 engine1 SHADOW launching"; phase e1_launch
B=$(launch 1 "$OUT/logs/engine1.log")
wait_for "$OUT/logs/engine1.log" "Engine sleeping" $SHA_TO "$B" "SHADOW(e1) standby" || exit 1
phase colocated_1; log "MEM COLOCATED (e0 active + e1 shadow): $(mem)"
log "infer (e0 active): HTTP $(infer p2) :: $(head -c 80 "$OUT/infer_p2.json")"

# ---- P3: FAILOVER-1 (kill e0 -> e1 promotes) ----
log "P3 FAILOVER-1: kill engine0"; phase failover1_kill
kill_engine "$A"
wait_for "$OUT/logs/engine1.log" "Registered endpoint" $PROMO_TO "$B" "e1 PROMOTED" || exit 1
phase e1_promoted; log "MEM after failover-1 (e1 sole active): $(mem)"
PF=000; for r in $(seq 1 12); do sleep 5; c=$(infer p3); log "  post-failover-1 retry $r: HTTP $c"; [ "$c" = "200" ] && { PF=200; break; }; done
[ "$PF" = "200" ] || { log "FAILOVER-1 never served 200"; exit 1; }

# ---- P4: REPLENISH (launch engineC as new shadow beside live e1) ----
log "P4 REPLENISH: launch engineC (ENGINE_ID=0 reused) beside live active e1"; phase replenish_launch
C=$(launch 0 "$OUT/logs/engine0b.log")
wait_for "$OUT/logs/engine0b.log" "Engine sleeping" $SHA_TO "$C" "REPLENISH shadow(eC) standby" || { log "REPLENISH FAILED to park"; exit 1; }
phase colocated_2; log "MEM COLOCATED (e1 active + eC shadow) — REPLENISHED: $(mem)"
log "infer (e1 still active, eC parked): HTTP $(infer p4) :: $(head -c 80 "$OUT/infer_p4.json")"

# ---- P5: FAILOVER-2 (kill e1 -> eC promotes) ----
log "P5 FAILOVER-2: kill engine1 (now active)"; phase failover2_kill
kill_engine "$B"
wait_for "$OUT/logs/engine0b.log" "Registered endpoint" $PROMO_TO "$C" "eC PROMOTED" || exit 1
phase eC_promoted; log "MEM after failover-2 (eC sole active): $(mem)"
PF2=000; for r in $(seq 1 12); do sleep 5; c=$(infer p5); log "  post-failover-2 retry $r: HTTP $c :: $(head -c 70 "$OUT/infer_p5.json")"; [ "$c" = "200" ] && { PF2=200; break; }; done
phase final
echo "KIMI_REPLENISH_DONE tag=$TAG failover1=$PF replenish_parked=yes failover2=$PF2"
