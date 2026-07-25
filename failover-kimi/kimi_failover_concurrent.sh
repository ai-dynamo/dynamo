#!/bin/bash
# CONCURRENT COLD-INIT — both engines launched at the SAME time (both shadow-mode).
# They race the flock: winner=ACTIVE (disk->GMS->commit->KV->warmup->register),
# loser=SHADOW (waits for committed weights -> RO import -> scratch-KV -> warmup ->
# standby). Unlike kimi_failover.sh (which serializes: active fully up THEN shadow),
# this measures the TRUE concurrent bring-up peak — the 0.23.0 risk case (each engine
# warmup-peaked ~178 GiB; two at once would blow the 183 ceiling). With autotune-off +
# single-block scratch, does concurrent cold-init now fit? Measure, don't assume.
set -u
source /opt/dynamo/venv/bin/activate 2>/dev/null || true
export HF_HOME=/tmp/hf HF_HUB_OFFLINE=0; mkdir -p /tmp/hf
MODEL=${MODEL:-/tmp/kimi-k2.6-nvfp4}; SERVED=${SERVED:-kimi-k2.6}
TP=${TP:-8}; MML=${MML:-4096}; UTIL=${UTIL:-0.8}
SINGLE=${DYN_GMS_SCRATCH_SINGLE_BLOCK:-1}; NOAT=${DYN_NO_AUTOTUNE:-1}; EAGER=${EAGER:-1}
CO_TO=${CO_TO:-1400}; PROMO_TO=${PROMO_TO:-400}
EAGER_FLAG=""; [ "$EAGER" = "1" ] && EAGER_FLAG="--enforce-eager"
TAG=${TAG:-concurrent-$( [ "$EAGER" = "1" ] && echo eager || echo graphs )}
OUT=/tmp/kimi_failover-$TAG; rm -rf "$OUT"; mkdir -p "$OUT/logs"; LOCK=$OUT/failover.lock
have(){ grep -aq "$2" "$1" 2>/dev/null; }
log(){ echo "[concurrent $(date +%T)] $*"; }
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

log "GMS weights+kv servers (TP=$TP)"
for d in $(seq 0 $((TP-1))); do
  python3 -m gpu_memory_service --device $d --tag weights  > "$OUT/logs/gms_w$d.log"  2>&1 &
  python3 -m gpu_memory_service --device $d --tag kv_cache > "$OUT/logs/gms_kv$d.log" 2>&1 &
done
for d in $(seq 0 $((TP-1))); do for i in $(seq 1 60); do have "$OUT/logs/gms_w$d.log" "Server started" && break; sleep 1; done; done
nohup python3 -m dynamo.frontend > "$OUT/logs/frontend.log" 2>&1 &

log "*** CONCURRENT LAUNCH: engine0 + engine1 at once (single_block=$SINGLE no_autotune=$NOAT eager=$EAGER) ***"
phase concurrent_launch
P0=$(launch 0 "$OUT/logs/engine0.log"); P1=$(launch 1 "$OUT/logs/engine1.log")
log "engine0 pid $P0 ; engine1 pid $P1 (racing flock)"

L0=$OUT/logs/engine0.log; L1=$OUT/logs/engine1.log
ACTIVE_LOG=""; SHADOW_LOG=""; ACTIVE_PID=""; SHADOW_PID=""
for i in $(seq 1 $CO_TO); do
  # resolve roles: the flock LOSER is the only one that sleeps ("Engine sleeping"),
  # so it is unambiguously the SHADOW; the other is ACTIVE. (Both transiently log
  # "waiting for lock" during the race, so that string cannot distinguish them.)
  if [ -z "$SHADOW_LOG" ]; then
    if have "$L0" "Engine sleeping"; then SHADOW_LOG=$L0; SHADOW_PID=$P0; ACTIVE_LOG=$L1; ACTIVE_PID=$P1; log "roles: engine1 ACTIVE, engine0 SHADOW (+${i}s)"
    elif have "$L1" "Engine sleeping"; then SHADOW_LOG=$L1; SHADOW_PID=$P1; ACTIVE_LOG=$L0; ACTIVE_PID=$P0; log "roles: engine0 ACTIVE, engine1 SHADOW (+${i}s)"; fi
  fi
  areg=0; sstb=0
  [ -n "$SHADOW_LOG" ] && sstb=1   # shadow has slept -> standby
  [ -n "$ACTIVE_LOG" ] && have "$ACTIVE_LOG" "Registered endpoint" && areg=1
  [ "$areg" = 1 ] && [ "$sstb" = 1 ] && { log "BOTH READY (active registered + shadow standby) +${i}s"; break; }
  kill -0 $P0 2>/dev/null || { log "engine0 DIED"; grep -anE "Error|OutOfMemory|Traceback|CUDA error|not validated|non-positive|routed through scratch" "$L0"|tail -12; exit 1; }
  kill -0 $P1 2>/dev/null || { log "engine1 DIED"; grep -anE "Error|OutOfMemory|Traceback|CUDA error|not validated|non-positive|routed through scratch" "$L1"|tail -12; exit 1; }
  sleep 1
done
phase concurrent_ready
[ -n "$ACTIVE_LOG" ] && have "$ACTIVE_LOG" "Registered endpoint" || { log "concurrent bring-up did not converge (timeout)"; exit 1; }
log "MEM after concurrent bring-up (active+shadow): $(mem)"
sleep 3; phase concurrent_settled; log "MEM settled: $(mem)"

for i in $(seq 1 60); do have "$OUT/logs/frontend.log" "Completions is ready" && break; sleep 1; done
log "infer ACTIVE: HTTP $(infer active) :: $(head -c 90 "$OUT/infer_active.json")"

# failover for completeness (kill whichever won active)
log "KILL active engine -> FAILOVER"; phase active_kill
pkill -9 -P "$ACTIVE_PID" 2>/dev/null; kill -9 "$ACTIVE_PID" 2>/dev/null
for i in $(seq 1 $PROMO_TO); do have "$SHADOW_LOG" "Registered endpoint" && { log "SHADOW PROMOTED +${i}s"; break; }
  kill -0 $SHADOW_PID 2>/dev/null || { log "shadow DIED on promotion"; grep -anE "Error|Traceback|CUDA error" "$SHADOW_LOG"|tail -10; break; }; sleep 1; done
phase shadow_promoted; log "MEM after failover: $(mem)"
PF=000; PFR=0
for r in $(seq 1 18); do sleep 5; code=$(infer postfail)
  log "  post-failover retry $r (+$((r*5))s): HTTP $code"; [ "$code" = "200" ] && { PF=200; PFR=$((r*5)); break; }; done
phase final; log "MEM final: $(mem)"
echo "KIMI_CONCURRENT_DONE tag=$TAG eager=$EAGER postfail=$PF postfail_secs=$PFR"
