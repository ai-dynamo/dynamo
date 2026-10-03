# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Node-side helpers for node_eval.sh and node_train.sh (sourced on the Slurm CPU node).
# Expects BUNDLE_NFS, OUT_NFS and LOCAL to be set by the caller.

JOB="${SLURM_JOB_ID:-nojob}"
SYNC_S="${LR_SYNC_S:-300}"

# Copy the bundle node-local (NFS is slow for many small reads) and verify every trace SHA-256.
lane_stage() {
  mkdir -p "$OUT_NFS" "$LOCAL"
  local t0 t1
  t0=$(date +%s.%N)
  rsync -a --exclude runs/cache/results --exclude results.jsonl --exclude 'runs/train' \
    "$BUNDLE_NFS/" "$LOCAL/"
  t1=$(date +%s.%N)
  STAGE_S=$(awk "BEGIN {print $t1 - $t0}")
  if ! (cd "$LOCAL" && sha256sum -c --quiet traces.sha256) > "$OUT_NFS/trace_check.txt" 2>&1; then
    echo "[lane] trace SHA-256 check FAILED"
    cat "$OUT_NFS/trace_check.txt"
    exit 4
  fi
  echo "ok $(wc -l < "$LOCAL/traces.sha256") traces" >> "$OUT_NFS/trace_check.txt"
  PY="${PYTHON:-$(command -v python3.12 || true)}"
  [ -n "$PY" ] || { echo "[lane] python3.12 not found (set PYTHON)"; exit 5; }
  # Default: one replay per physical core. Replays are single-threaded; SMT siblings doubled each
  # replay's wall time and added under 10% node throughput (facts/remote.json throughput), and a
  # shorter per-replay time shortens lr-train's per-generation barrier.
  local tpc
  tpc=$(lscpu | awk -F: '/^Thread\(s\) per core/ {gsub(/ /, "", $2); print $2}')
  SLOTS="${LR_SLOTS:-$(( $(nproc) / ${tpc:-1} ))}"
}

lane_probe() {
  cat > "$OUT_NFS/node.json" <<EOF
{"job": "$JOB", "host": "$(hostname)", "nproc": $(nproc),
 "cpus_online": $(getconf _NPROCESSORS_ONLN), "slots": $SLOTS,
 "cpu_model": "$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')",
 "mem_total_kib": $(awk '/MemTotal/ {print $2}' /proc/meminfo),
 "glibc": "$(ldd --version | head -1 | awk '{print $NF}')", "python": "$($PY -V 2>&1)",
 "python_path": "$PY", "os": "$(. /etc/os-release && echo "$PRETTY_NAME")",
 "local_dir": "$LOCAL", "local_fs": "$(df -h --output=fstype,size,avail "$(dirname "$LOCAL")" | tail -1 | xargs)",
 "stage_s": $STAGE_S, "cgroup_mem_max": "$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo n/a)",
 "load_at_start": "$(cat /proc/loadavg)"}
EOF
}

# Memory/load sampler every 10 s -> OUT_NFS/mem.tsv:
# epoch, cgroup memory.current, cgroup memory.peak, RSS sum of this user's processes (KiB),
# largest single RSS (KiB), 1-min load average.
lane_sampler_start() {
  (
    while true; do
      echo -e "$(date +%s)\t$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo NA)\t$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo NA)\t$(ps -u "$USER" -o rss= | awk '{s+=$1} END {print s+0}')\t$(ps -u "$USER" -o rss= | sort -n | tail -1 | tr -d ' ')\t$(cut -d' ' -f1 /proc/loadavg)"
      sleep 10
    done
  ) > "$OUT_NFS/mem.tsv" 2>/dev/null &
  SAMPLER=$!
}

# Copy finished results (cache records + per-request rows), the E0 table, results.jsonl, logs and
# lr-train run directories back to OUT_NFS.
lane_sync_back() {
  mkdir -p "$OUT_NFS/runs/cache"
  [ -d "$LOCAL/runs/cache/results" ] && rsync -a "$LOCAL/runs/cache/results" "$OUT_NFS/runs/cache/" || true
  [ -d "$LOCAL/runs/cache/e0" ] && rsync -a "$LOCAL/runs/cache/e0" "$OUT_NFS/runs/cache/" || true
  [ -d "$LOCAL/runs/train" ] && rsync -a "$LOCAL/runs/train" "$OUT_NFS/runs/" || true
  [ -f "$LOCAL/results.jsonl" ] && cp "$LOCAL/results.jsonl" "$OUT_NFS/results.jsonl" || true
  [ -d "$LOCAL/logs" ] && rsync -a "$LOCAL/logs" "$OUT_NFS/" || true
}

lane_periodic_sync_start() {
  ( while true; do sleep "$SYNC_S"; lane_sync_back; done ) &
  SYNCER=$!
  trap 'kill $SAMPLER $SYNCER 2>/dev/null || true; lane_sync_back' EXIT
  trap 'exit 143' TERM
}

# Clamp a wall budget to the job's real end time. Slurm can start a job with less than the requested
# limit (a QoS with TimeMin, or a partition whose jobs must end before a maintenance cutoff),
# so the requested --time is not a safe deadline. MARGIN covers the longest replay plus the sync.
lane_clamp_wall() {
  local want="$1" margin="${2:-600}" left
  if [ -n "${SLURM_JOB_END_TIME:-}" ]; then
    left=$(( SLURM_JOB_END_TIME - $(date +%s) - margin ))
    [ "$left" -lt "$want" ] && want="$left"
  fi
  [ "$want" -lt 60 ] && want=60
  echo "$want"
}
