#!/usr/bin/env bash
# Restart timing: kill the engine process, wait for Ready, dump [dyn-phase] lines.
# Use on gms-only and no-gms DGDs after caches are warm.
set -euo pipefail
NS=${NS:-schwinns-vcluster}
HOST_KC=${HOST_KC:-$HOME/.kube/config}
DGD=${DGD:?set DGD}
CONTAINER=${CONTAINER:-main}
OUT=${OUT:-bench/runs/restart-$(date -u +%Y%m%dT%H%M%SZ)}
kh() { kubectl --kubeconfig "$HOST_KC" "$@"; }

W=$(kh get pods -n "$NS" --no-headers -o custom-columns=:metadata.name \
      | grep "$DGD-vllmdecodeworker" | grep -v Terminating | head -1)
[ -n "$W" ] || { echo "worker not found"; exit 1; }
mkdir -p "$OUT"
echo "worker=$W container=$CONTAINER" | tee "$OUT/meta.txt"
kh logs -n "$NS" "$W" -c "$CONTAINER" --tail=5000 2>/dev/null \
  | grep -E '\[dyn-phase\]|VllmWorker .* has been initialized|graph capturing' \
  > "$OUT/phases-before.txt" || true
T0=$(date -u +%s%N)
echo "$T0" > "$OUT/kill.txt"
kh exec -n "$NS" "$W" -c "$CONTAINER" -- bash -c "pkill -9 -f 'dynamo.vllm|VLLM::' || kill -9 1" \
  >/dev/null 2>&1 || true
echo "killed $CONTAINER at $T0; waiting for Ready"
for i in $(seq 1 600); do
  ready=$(kh get pod -n "$NS" "$W" --no-headers | awk '{print $2}')
  [ "${ready%%/*}" = "${ready##*/}" ] && [ "${ready%%/*}" != "0" ] && break
  sleep 5
done
T1=$(date -u +%s%N)
echo "$T1" > "$OUT/ready.txt"
python3 - <<PY
t0=int(open("$OUT/kill.txt").read())/1e9
t1=int(open("$OUT/ready.txt").read())/1e9
print(f"wall_ready_s={t1-t0:.1f}")
PY
kh logs -n "$NS" "$W" -c "$CONTAINER" --tail=8000 2>/dev/null \
  | grep -E '\[dyn-phase\]|VllmWorker .* has been initialized|graph capturing|Read mode: imported|Write mode:' \
  > "$OUT/phases-after.txt" || true
echo "wrote $OUT"
