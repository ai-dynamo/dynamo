#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Bring a remote batch home and ingest it into the campaign cache.
#
#   fetch_ingest.sh --name NAME [--out RESULTS.jsonl] [--no-ingest] [--root DIR]
#
# 1. refreshes the batch's jobs in CR/facts/remote.json from sacct and prints their states;
# 2. rsyncs SSH_ALIAS:REMOTE_ROOT/returned/NAME/ to CR/runs/remote/returned/NAME/ (works while jobs are
#    still running: shards sync finished results every 5 minutes);
# 3. runs `lr-eval --ingest` on every shard directory (cache keys are recomputed from each record's
#    own fields; harness version and bindings build must match), appending the shards' result rows
#    to --out (default CR/runs/remote/returned/NAME/results.ingested.jsonl), and records a summary
#    under batches.NAME in CR/facts/remote.json. lr-train run directories arrive under
#    CR/runs/remote/returned/NAME/<job>/runs/train/<run>/.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

NAME="" OUT="" INGEST=1 ROOT="$CR"
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --no-ingest) INGEST=0; shift ;;
    --root) ROOT="$2"; shift 2 ;;
    *) die "unknown argument $1 (see the header of $0)" ;;
  esac
done
[ -n "$NAME" ] || die "--name is required"
JOBS_FILE="$LOCAL_RUNS/$NAME.jobs.json"
[ -f "$JOBS_FILE" ] || die "$JOBS_FILE not found (submit with submit_eval.sh or submit_train.sh)"
DEST="$LOCAL_RUNS/returned/$NAME"
OUT="${OUT:-$DEST/results.ingested.jsonl}"
mkdir -p "$DEST"

mapfile -t JOBS < <("$PY" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["jobs"]))' "$JOBS_FILE")
refresh=()
for job in "${JOBS[@]}"; do refresh+=(--job "$job"); done
"$PY" "$LANE_DIR/lane.py" alloc-refresh "${refresh[@]}" >/dev/null || echo "warning: sacct refresh failed" >&2
"$PY" - "$CR/facts/remote.json" "${JOBS[@]}" <<'EOF'
import json, sys
facts = json.load(open(sys.argv[1]))
for job in sys.argv[2:]:
    a = next((x for x in facts.get("allocations", []) if str(x["job_id"]) == job), {})
    print(f"job {job}: {a.get('state', '?')} node={a.get('node', '?')} elapsed={a.get('elapsed', '?')}")
EOF

rsync -a "$SSH_ALIAS:$REMOTE_ROOT/returned/$NAME/" "$DEST/"
[ "$INGEST" -eq 1 ] || { echo "fetched to $DEST (not ingested)"; exit 0; }

summaries=()
for shard in "$DEST"/*/; do
  shard="${shard%/}"
  [ -d "$shard/runs/cache/results" ] || { echo "skip $shard (no results yet)"; continue; }
  set +e
  "$LR_EVAL" --root "$ROOT" --ingest "$shard" --out "$OUT" > "$shard/ingest.json"
  rc=$?
  set -e
  echo "$(basename "$shard"): $(cat "$shard/ingest.json") (exit $rc)"
  summaries+=("$shard/ingest.json")
done
"$PY" - "$NAME" "$DEST" "$OUT" "$JOBS_FILE" "${summaries[@]}" <<'EOF' > "$DEST/batch_summary.json"
import json, sys, time
name, dest, out, jobs_file, *paths = sys.argv[1:]
shards = {}
for p in paths:
    s = json.load(open(p))
    shards[p.split("/")[-2]] = {k: s[k] for k in ("ingested", "already_cached", "results_appended", "e0_values_merged")} | {"rejected": len(s["rejected"])}
print(json.dumps({"name": name, "returned": dest, "results": out, "jobs": json.load(open(jobs_file))["jobs"],
                  "shards": shards, "fetched": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=1, sort_keys=True))
EOF
"$PY" - "$CR/facts/remote.json" "$DEST/batch_summary.json" "$LANE_DIR" <<'EOF'
import json, sys
sys.path.insert(0, sys.argv[3])
import lane
summary = json.load(open(sys.argv[2]))
with lane.facts_lock() as data:
    data.setdefault("batches", {})[summary["name"]] = summary
EOF
echo "ingested; summary $DEST/batch_summary.json"
