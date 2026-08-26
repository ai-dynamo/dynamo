#!/usr/bin/env bash
# GMS V0 shadow-failover experiment. Same measurement shape as dynamo-snaptrack:
# open-loop synthetic load, SIGKILL the active engine, collect a bundle, then
# cutover.py / measure_replenish.py for promotion time and replenishment contention.
set -uo pipefail

NS=${NS:-schwinns-vcluster}
HOST_KC=${HOST_KC:-$HOME/.kube/config}
VC_KC=${VC_KC:-$HOME/.kube/vc-schwinns.yaml}
DGD=${DGD:-dsv4pro-vllm-gmsv0-fo}
FRONTEND_DGD=${FRONTEND_DGD:-$DGD}
MODEL=${MODEL:-nvidia/DeepSeek-V4-Pro-NVFP4}
TOKENIZER=${TOKENIZER:-nvidia/DeepSeek-V4-Pro-NVFP4}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-bench/runs/$RUN_ID}

LOAD=${LOAD:-1}
CONCURRENCY=${CONCURRENCY:-2048}
ISL=${ISL:-32000}
OSL=${OSL:-1000}
REQUEST_RATE=${REQUEST_RATE:-0.7}
RATE_MODE=${RATE_MODE:-poisson}
LOAD_DURATION=${LOAD_DURATION:-1200}
SOAK_S=${SOAK_S:-450}
MIN_RUNNING=${MIN_RUNNING:-4}
WATCH_S=${WATCH_S:-600}
BASELINE=${BASELINE:-0}

kh() { kubectl --kubeconfig "$HOST_KC" "$@"; }
kv() { kubectl --kubeconfig "$VC_KC" "$@"; }
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

ensure_vc() {
  if [ -f "$VC_KC" ]; then
    return 0
  fi
  log "writing vcluster kubeconfig to $VC_KC"
  vcluster connect schwinns-vcluster -n schwinns-vcluster --print > "$VC_KC"
}

ensure_vc
mkdir -p "$OUT"/{harness,aiperf,analysis,logs}

WORKER_GREP="$DGD-vllmdecodeworker"
W=$(kh get pods -n "$NS" --no-headers -o custom-columns=:metadata.name | grep "$WORKER_GREP" | grep -v Terminating | head -1)
WVC=$(kv get pods -n "$NS" --no-headers -o custom-columns=:metadata.name | grep "$WORKER_GREP" | head -1)
BENCH=$(kh get pods -n "$NS" --no-headers -o custom-columns=:metadata.name | grep '^bench-shell' | head -1)
[ -n "$W" ] || { echo "worker pod not found for $WORKER_GREP"; exit 1; }
WNODE=$(kh get pod -n "$NS" "$W" -o jsonpath='{.spec.nodeName}')
log "worker=$W node=$WNODE bench=${BENCH:-none}"

ACTIVE=""; STANDBY=""
if [ "$BASELINE" = 1 ]; then
  ACTIVE=main; STANDBY=main
  log "BASELINE=1 — no shadow: active=standby=main"
else
  for c in engine-0 engine-1; do
    last=$(kh logs -n "$NS" "$W" -c "$c" --tail=20000 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' \
           | grep -aoE 'failover_state engine=[01] -> [a-z]+' | tail -1)
    case "$last" in *active) ACTIVE=$c;; *standby) STANDBY=$c;; esac
  done
  [ -z "$ACTIVE" ] && [ -n "$STANDBY" ] && ACTIVE=$([ "$STANDBY" = engine-0 ] && echo engine-1 || echo engine-0)
  [ -z "$STANDBY" ] && [ -n "$ACTIVE" ] && STANDBY=$([ "$ACTIVE" = engine-0 ] && echo engine-1 || echo engine-0)
fi
[ -n "$ACTIVE" ] && [ -n "$STANDBY" ] || { echo "could not resolve roles"; exit 1; }
log "active=$ACTIVE standby=$STANDBY"
printf 'active=%s\nstandby=%s\n' "$ACTIVE" "$STANDBY" > "$OUT/harness/roles.txt"

read_running() {
  kh logs -n "$NS" "$W" -c "$ACTIVE" --tail=800 2>/dev/null \
    | grep -ao "Running: [0-9]* reqs" | tail -1 | grep -oE '[0-9]+'
}

AIPERF_REMOTE="/artifacts/runs/$(basename "$OUT")"
if [ "$LOAD" = 1 ]; then
  [ -n "$BENCH" ] || { echo "no bench-shell to drive load"; exit 1; }
  for _ in $(seq 1 60); do
    [ "$(kh get pod -n "$NS" "$BENCH" -o jsonpath='{.status.containerStatuses[?(@.name=="bench")].ready}' 2>/dev/null)" = "true" ] && break
    sleep 2
  done
  ROOT=$(cd "$(dirname "$0")/../.." && pwd)
  for f in aiperf_load.sh check_tokenizer.sh; do
    kh cp "$ROOT/bench/harness/$f" "$NS/$BENCH:/tmp/$f" -c bench >/dev/null 2>&1
  done
  URL_CHK="http://$FRONTEND_DGD-frontend.$NS.svc.cluster.local:8000"
  if ! kh exec -n "$NS" "$BENCH" -c bench -- bash -c \
        "MODEL='$MODEL' TOKENIZER='$TOKENIZER' URL='$URL_CHK' bash /tmp/check_tokenizer.sh" \
        > "$OUT/harness/tokenizer_check.txt" 2>&1; then
    log "!! tokenizer disagrees with the server — refusing to run"
    sed 's/^/    /' "$OUT/harness/tokenizer_check.txt" | tail -8
    exit 1
  fi
  URL="$URL_CHK"
  log "starting aiperf: rate=$REQUEST_RATE/s isl=$ISL osl=$OSL dur=${LOAD_DURATION}s"
  kh exec -n "$NS" "$BENCH" -c bench -- bash -c "
    mkdir -p $AIPERF_REMOTE
    MODEL='$MODEL' URL='$URL' TOKENIZER='$TOKENIZER' ARTIFACT_DIR='$AIPERF_REMOTE' \
    CONCURRENCY=$CONCURRENCY ISL=$ISL OSL=$OSL DURATION=$LOAD_DURATION \
    REQUEST_RATE='$REQUEST_RATE' RATE_MODE='$RATE_MODE' \
      nohup bash /tmp/aiperf_load.sh > $AIPERF_REMOTE/run.log 2>&1 &
    echo started" >/dev/null
  log "waiting for Running >= $MIN_RUNNING"
  ok=0
  for i in $(seq 1 90); do
    r=$(read_running)
    [ -n "$r" ] && log "  Running=$r"
    [ -n "$r" ] && [ "$r" -ge "$MIN_RUNNING" ] && { ok=1; echo "$r" > "$OUT/harness/running_gate.txt"; break; }
    sleep 5
  done
  [ "$ok" = 1 ] || { log "!! never reached Running>=$MIN_RUNNING"; exit 1; }
  if [ "${SOAK_S:-0}" -gt 0 ]; then
    log "soaking ${SOAK_S}s before kill"
    sleep "$SOAK_S"
  fi
fi

rk=$(read_running)
echo "${rk:-UNVERIFIED}" > "$OUT/harness/running_at_kill.txt"
log "verified Running=${rk:-UNVERIFIED} immediately before kill"

T0=$(date -u +%s%N)
if [ "${KILL:-1}" = 1 ]; then
  kh exec -n "$NS" "$W" -c "$ACTIVE" -- bash -c "pkill -9 -f 'dynamo.vllm|VLLM::' || kill -9 1" >/dev/null 2>&1 || true
  log "SIGKILL $ACTIVE"
  echo "$T0 $ACTIVE pkill dynamo.vllm" > "$OUT/harness/kill.txt"
else
  log "KILL=0"
  echo "$T0 $ACTIVE none none" > "$OUT/harness/kill.txt"
fi

log "watching ${WATCH_S}s"
sleep "$WATCH_S"

if [ "$LOAD" = 1 ] && [ -n "$BENCH" ]; then
  waitmax=$(( (LOAD_DURATION + 300) / 5 ))
  log "waiting for aiperf summary (up to $((waitmax*5))s)"
  for i in $(seq 1 "$waitmax"); do
    if kh exec -n "$NS" "$BENCH" -c bench -- \
         test -f "$AIPERF_REMOTE/profile_export_aiperf.json" 2>/dev/null; then
      log "  aiperf summary written after ~$((i*5))s"; break
    fi
    sleep 5
  done
fi

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT="$OUT" DGD="$DGD" NODE="$WNODE" AIPERF_REMOTE_DIR="$AIPERF_REMOTE" \
  bash "$ROOT/bench/harness/collect_artifacts.sh" >/dev/null 2>&1 || true
log "bundle at $OUT"
python3 "$ROOT/bench/harness/cutover.py" "$OUT" || true
python3 "$ROOT/bench/harness/measure_replenish.py" "$OUT" || true
