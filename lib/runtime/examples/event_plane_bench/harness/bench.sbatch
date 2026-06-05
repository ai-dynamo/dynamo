#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# DIS-2172 multi-node event-plane benchmark — PYXIS CONTAINER form.
# CPU-only (mocker; no GPU). Submit:  sbatch bench.sbatch
#
# Architecture:
#   * The orchestrator (bench.py) runs on node0 with the HOST python (stdlib only).
#   * It srun's every task (mocker / frontend / loadgen / bench_sub) INTO the
#     instrumented dynamo container via pyxis (--container-image=$DYN_BENCH_IMAGE),
#     using in-container paths (DYN_BENCH_PY / DYN_BENCH_SUB_BIN).
#   * etcd + nats run on node0 from the SAME image.
#
# Prereq: build + push the image once with build_image.sh, then:
#   export DYN_BENCH_IMAGE=<registry>:<tag>   (sbatch inherits the env)
#
#SBATCH --job-name=dis2172-eventplane
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --exclusive
# Set --partition for your cluster, e.g.:  #SBATCH --partition=lego-c2
# NOTE: no --gres (mocker is CPU-only).

set -euo pipefail
BENCH_DIR="${BENCH_DIR:-$PWD}"
NODE0=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)

# ============================================================================
# CONTAINER + PATHS
# ============================================================================
: "${DYN_BENCH_IMAGE:?set DYN_BENCH_IMAGE to the pushed image (see build_image.sh)}"
# In-container paths (override if your image differs — build_image.sh prints them):
export DYN_BENCH_IMAGE
export DYN_BENCH_PY="${DYN_BENCH_PY:-/opt/dynamo/venv/bin/python}"
export DYN_BENCH_SUB_BIN="${DYN_BENCH_SUB_BIN:-/usr/local/bin/event_plane_bench_sub}"
export DYN_BENCH_MODEL="${DYN_BENCH_MODEL:-Qwen/Qwen2.5-0.5B}"
# Mount the harness dir at the SAME path in-container so bench.py's loadgen.py /
# zmq_broker.py / results paths resolve identically inside and out.
export DYN_BENCH_MOUNTS="${DYN_BENCH_MOUNTS:-$BENCH_DIR:$BENCH_DIR}"

# Pick the host python that runs the orchestrator (stdlib only; NOT the container).
HOST_PY="${HOST_PY:-python3}"

# ============================================================================
# CLOCK SYNC (critical for cross-node latency): multi-node uses CLOCK_REALTIME
# for the one-way delta, so nodes MUST be PTP/NTP-synced. computelab/dlcluster
# are NTP (no PTP) -> treat cross-node *latency* as N/A; the no-PTP primary
# metrics are throughput (events/s) + drop_rate (seq-gap), which bench.py/report.py
# already emit. Still record the offset here for the record.
# ============================================================================
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --overlap bash -lc \
     'echo "$(hostname): $(chronyc tracking 2>/dev/null | grep -i "system time" || echo no-chrony)"' \
     2>/dev/null | sort > "$BENCH_DIR/clocksync.txt" || true

# ============================================================================
# etcd + nats on node0, from the container image (dynamo dev image bundles them).
# Every task connects here (bench.py sets ETCD_ENDPOINTS / NATS_SERVER to NODE0).
# ============================================================================
srun --nodelist="$NODE0" --overlap --ntasks=1 --container-image="$DYN_BENCH_IMAGE" \
     --output="$BENCH_DIR/nats.log" -- \
     nats-server -a 0.0.0.0 -p 4222 &
srun --nodelist="$NODE0" --overlap --ntasks=1 --container-image="$DYN_BENCH_IMAGE" \
     --output="$BENCH_DIR/etcd.log" -- \
     etcd --data-dir "/tmp/etcd-${SLURM_JOB_ID}" \
       --listen-client-urls "http://0.0.0.0:2379" \
       --advertise-client-urls "http://${NODE0}:2379" &
sleep 10   # let infra bind before workers start

OUT="${OUT:-$BENCH_DIR/results/nvl72_${SLURM_JOB_ID}}"

# One command — same sweep on single- or multi-node, only --launcher slurm differs.
# NVL72: 72 workers across 8 nodes -> --gpus-per-node 9 (one mocker worker per GPU);
# sweep p up to nodes*9 = 72.  --speedup-ratio 0 = unthrottled (push to saturation).
"$HOST_PY" "$BENCH_DIR/bench.py" \
  --launcher      slurm \
  --gpus-per-node "${GPUS_PER_NODE:-9}" \
  --workers       "${WORKERS:-8,16,32,72}" \
  --subs          "${SUBS:-1,2,4}" \
  --transports    "${TRANSPORTS:-nats,zmq,zmq-broker}" \
  --topics        "${TOPICS:-kv-events,forward-pass-metrics,kv_metrics}" \
  --trials        "${TRIALS:-3}" \
  --speedup-ratio "${SPEEDUP_RATIO:-0}" \
  --infra-addr    "$NODE0" \
  --out           "$OUT"

"$HOST_PY" "$BENCH_DIR/report.py" --in "$OUT"
echo "[sbatch] done -> $OUT/report.md  (clocksync -> $BENCH_DIR/clocksync.txt)"
