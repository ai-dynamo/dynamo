#!/usr/bin/env bash
# Collect every artifact of one failover run into a self-describing bundle.
#
# The rule this enforces: a run must be reconstructible from its bundle alone,
# without the cluster still being in that state. Two things have already been
# lost by not doing this. Plain `kubectl logs` returns the CURRENT container, so
# for the engine we just killed it hands back the post-restart log and silently
# drops the pre-kill scheduler lines -- the only evidence load was flowing.
# And image tags are mutable, so a bundle that records a tag rather than a
# digest cannot prove which build produced its numbers.
#
# Layout:
#   cluster/   DGD, checkpoint CR, pod spec, events        (what was deployed)
#   logs/      engine + gms + frontend, each with .previous (what it did)
#   aiperf/    profile export, summary, run log            (what load was sent)
#   harness/   kill, roles, gpu-mem, proc-tree, timings    (what we did to it)
#   analysis/  derived timeline                            (what it means)
set -uo pipefail

NS=${NS:-schwinns-vcluster}
HOST_KC=${HOST_KC:-$HOME/.kube/config}
VC_KC=${VC_KC:-$HOME/.kube/vc-schwinns.yaml}
DGD=${DGD:-glm52-vllm-fo-snapshot}
WORKER_GREP=${WORKER_GREP:-$DGD-vllmdecodeworker}
BENCH_POD_GREP=${BENCH_POD_GREP:-^bench-shell}
AIPERF_REMOTE_DIR=${AIPERF_REMOTE_DIR:-}
NODE=${NODE:-cluster-0967a26d-pool-14bee067-prctr-tx5tk}
OUT=${OUT:?set OUT}

kh() { kubectl --kubeconfig "$HOST_KC" "$@"; }
kv() { kubectl --kubeconfig "$VC_KC" "$@"; }

mkdir -p "$OUT"/{cluster,logs,aiperf,harness,analysis,metrics}

W=$(kh get pods -n "$NS" --no-headers -o custom-columns=:metadata.name | grep "$WORKER_GREP" | head -1)
FE=$(kh get pods -n "$NS" --no-headers -o custom-columns=:metadata.name | grep "$DGD-frontend" | head -1)
AGENT=$(kh get pods -n "$NS" -o wide --no-headers | awk -v n="$NODE" '/forka/ && $7==n {print $1}' | head -1)
BENCH=$(kh get pods -n "$NS" --no-headers -o custom-columns=:metadata.name | grep -E "$BENCH_POD_GREP" | head -1)

# ---- cluster state -----------------------------------------------------------
kv get dynamographdeployment "$DGD" -n "$NS" -o yaml > "$OUT/cluster/dgd.yaml" 2>/dev/null
CK=$(kv get dynamographdeployment "$DGD" -n "$NS" \
     -o jsonpath='{.status.checkpoints.VllmDecodeWorker.checkpointID}' 2>/dev/null)
[ -n "$CK" ] && kv get dynamocheckpoint "checkpoint-$CK" -n "$NS" -o yaml > "$OUT/cluster/checkpoint.yaml" 2>/dev/null
[ -n "$W" ] && kh get pod -n "$NS" "$W" -o yaml > "$OUT/cluster/worker-pod.yaml" 2>/dev/null
kh get events -n "$NS" --sort-by=.lastTimestamp > "$OUT/cluster/events.txt" 2>/dev/null

# Who else is on this node, and what the node thinks it has to give away.
#
# Namespace-scoped events cannot show a co-tenant, because a co-tenant is by
# definition in someone else's namespace. That matters here: our pods hold these
# GPUs through a DRA claim and request no nvidia.com/gpu at all, so if the
# classic device plugin is still advertising them, every one of those GPUs looks
# free to the scheduler while we are using them. A pod from any other team can
# land on the same silicon and contend for SMs and NVLink without appearing
# anywhere in the artifacts we were collecting.
if [ -n "$NODE" ]; then
    kh get pods -A -o wide --field-selector "spec.nodeName=$NODE" \
        > "$OUT/cluster/node-pods.txt" 2>/dev/null
    kh get node "$NODE" -o yaml > "$OUT/cluster/node.yaml" 2>/dev/null
    # Anything actually holding device-plugin GPUs, as opposed to DRA.
    kh get pods -A -o json --field-selector "spec.nodeName=$NODE" 2>/dev/null \
      | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
for p in d.get("items",[]):
    for c in p["spec"]["containers"]:
        r=c.get("resources",{})
        n=(r.get("limits") or {}).get("nvidia.com/gpu") or (r.get("requests") or {}).get("nvidia.com/gpu")
        if n: print(f"{p[\"metadata\"][\"namespace\"]}/{p[\"metadata\"][\"name\"]} {c[\"name\"]} nvidia.com/gpu={n} phase={p[\"status\"].get(\"phase\")}")
' > "$OUT/cluster/node-device-plugin-holders.txt" 2>/dev/null
fi

# ---- logs, current AND previous ----------------------------------------------
# --previous is not optional here: the engine we kill restarts, so its pre-kill
# history exists nowhere else.
if [ -n "$W" ]; then
  # "main" is the baseline arm's only container (stock vLLM, no shadow); the
  # engine-*/gms-server names exist only on the failover arm. Ask for all of
  # them and drop whatever the pod does not have, so one collector serves both.
  for c in engine-0 engine-1 gms-server main; do
    kh logs -n "$NS" "$W" -c "$c" --tail=200000 > "$OUT/logs/$c.log" 2>/dev/null
    kh logs -n "$NS" "$W" -c "$c" --previous --tail=200000 > "$OUT/logs/$c.previous.log" 2>/dev/null
    [ -s "$OUT/logs/$c.log" ]          || rm -f "$OUT/logs/$c.log"
    [ -s "$OUT/logs/$c.previous.log" ] || rm -f "$OUT/logs/$c.previous.log"
  done
fi
[ -n "$FE" ] && kh logs -n "$NS" "$FE" --tail=100000 > "$OUT/logs/frontend.log" 2>/dev/null

# Router decisions, one line per request, extracted from the frontend.
#
# DYN_ROUTER_MODE=kv means the router scores workers by prefix overlap, so after
# a promotion it prefers the warm survivor over the cold shadow: measured 83/16
# in the first minute after a kill, rebalancing to ~50/50 by two minutes. That
# skew is part of what a failover costs and needs to be in the bundle rather
# than reconstructed later from a live pod whose logs have since rotated.
[ -n "$FE" ] && kh logs -n "$NS" "$FE" --tail=200000 2>/dev/null \
    | sed 's/\x1b\[[0-9;]*m//g' | grep "scheduling::selector" \
    > "$OUT/logs/router-decisions.log" 2>/dev/null
[ -s "$OUT/logs/router-decisions.log" ] || rm -f "$OUT/logs/router-decisions.log"
# The agent owns CRIU dump/restore; its log is where restore failures explain
# themselves and where checkpoint timing summaries live.
[ -n "$AGENT" ] && kh logs -n "$NS" "$AGENT" -c agent --tail=20000 > "$OUT/logs/snapshot-agent.log" 2>/dev/null

# Failover metrics, scraped per engine.
#
# The LD_PRELOAD runtime does not carry our checkpoint_restore/wake_up timers,
# but it exposes something better: a failover state machine whose
# last_state_duration_seconds{state="waking"} is documented as the wake/switch
# time. Scraping it makes cutover measurable on any runtime that publishes it,
# and costs nothing on runtimes that do not -- the file is simply empty.
#
# Both engines are scraped because the one that matters is whichever was
# promoted, and that is the opposite of whichever was active at the kill.
for i in 0 1; do
    port=$((9090 + i))
    kh exec -n "$NS" "$W" -c "engine-$i" -- \
        sh -c "curl -sS -m 8 localhost:$port/metrics 2>/dev/null | grep -E 'failover'" \
        > "$OUT/metrics/engine-$i.prom" 2>/dev/null
    [ -s "$OUT/metrics/engine-$i.prom" ] || rm -f "$OUT/metrics/engine-$i.prom"
done

# ---- aiperf artifacts --------------------------------------------------------
if [ -n "$BENCH" ] && [ -n "$AIPERF_REMOTE_DIR" ]; then
  # server_metrics_* are aiperf's scrape of the frontend /metrics endpoint --
  # an independent view of what the server thought it was doing, useful when the
  # client and engine disagree.
  for f in profile_export.jsonl profile_export_aiperf.json profile_export_aiperf.csv \
           server_metrics_export.json server_metrics_export.csv \
           inputs.json gpu_telemetry_export.jsonl run.log; do
    kh exec -n "$NS" "$BENCH" -c bench -- cat "$AIPERF_REMOTE_DIR/$f" > "$OUT/aiperf/$f" 2>/dev/null
    [ -s "$OUT/aiperf/$f" ] || rm -f "$OUT/aiperf/$f"
  done
  kh exec -n "$NS" "$BENCH" -c bench -- sh -c "cat $AIPERF_REMOTE_DIR/logs/aiperf.log" \
    > "$OUT/aiperf/aiperf.log" 2>/dev/null
  [ -s "$OUT/aiperf/aiperf.log" ] || rm -f "$OUT/aiperf/aiperf.log"
fi

# ---- index -------------------------------------------------------------------
{
  echo "# Run bundle"
  echo
  echo "collected:  $(date -u -Iseconds)"
  echo "dgd:        $DGD"
  echo "checkpoint: ${CK:-<none>}"
  echo "worker pod: ${W:-<none>}"
  echo
  echo "## Contents"
  find "$OUT" -mindepth 2 -type f -printf '  %-52p %10s bytes\n' 2>/dev/null | sed "s|$OUT/||" | sort
} > "$OUT/README.md"
echo "bundle -> $OUT"
du -sh "$OUT" 2>/dev/null | sed 's/^/  /'
