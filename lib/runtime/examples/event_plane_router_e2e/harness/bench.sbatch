#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# DIS-2172 REAL end-to-end event-plane THROUGHPUT + LOSS bench — PYXIS form.
# CPU-only (mocker + routers + FPM receiver; no GPU). Submit: sbatch bench.sbatch
#
# Architecture:
#   * orchestrator (bench.py) runs on node0 with the HOST python (stdlib only).
#   * it srun's every task (frontend kv-routers / mocker / recv_fpm / loadgen)
#     INTO the instrumented dynamo container via pyxis.
#   * etcd + nats run on node0 from the SAME image.
#   * the three real consumers emit per-window dis2172_recv JSON to their logs;
#     bench.py scrapes those.
#
# Prereq: build + push the image once with build_image.sh, then:
#   export DYN_BENCH_IMAGE=<registry>:<tag>
#
#SBATCH --job-name=dis2172-router-e2e
#SBATCH --nodes=7
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --exclusive

set -euo pipefail
BENCH_DIR="${BENCH_DIR:-$PWD}"
NODE0=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)

export DYN_BENCH_IMAGE="${DYN_BENCH_IMAGE:?set DYN_BENCH_IMAGE to the pushed image}"
export DYN_BENCH_PY="${DYN_BENCH_PY:-/opt/dynamo/venv/bin/python}"
export DYN_BENCH_MODEL="${DYN_BENCH_MODEL:-Qwen/Qwen2.5-0.5B}"
export DYN_BENCH_MOUNTS="${DYN_BENCH_MOUNTS:-$BENCH_DIR:$BENCH_DIR}"
export DYN_BENCH_CONTAINER_NAME="${DYN_BENCH_CONTAINER_NAME:-dis2172e2e}"
export DYN_BENCH_MODEL_WAIT="${DYN_BENCH_MODEL_WAIT:-600}"
export DYN_BENCH_EXTRACT_PAD="${DYN_BENCH_EXTRACT_PAD:-220}"
export DYN_BENCH_WINDOW_SECS="${DYN_BENCH_WINDOW_SECS:-10}"

HOST_PY="${HOST_PY:-python3}"

# etcd + nats on node0 from the container image.
srun --nodelist="$NODE0" --overlap --ntasks=1 --container-image="$DYN_BENCH_IMAGE" \
     --output="$BENCH_DIR/nats.log" -- \
     nats-server -a 0.0.0.0 -p 4222 &
srun --nodelist="$NODE0" --overlap --ntasks=1 --container-image="$DYN_BENCH_IMAGE" \
     --output="$BENCH_DIR/etcd.log" -- \
     etcd --data-dir "/tmp/etcd-${SLURM_JOB_ID}" \
       --listen-client-urls "http://0.0.0.0:2379" \
       --advertise-client-urls "http://${NODE0}:2379" &
sleep 10

OUT="${OUT:-$BENCH_DIR/results/e2e_${SLURM_JOB_ID}}"

"$HOST_PY" "$BENCH_DIR/bench.py" \
  --launcher      slurm \
  --gpus-per-node "${GPUS_PER_NODE:-18}" \
  --workers       "${WORKERS:-8,16,32,72}" \
  --subs          "${SUBS:-1,2,4,8}" \
  --transports    "${TRANSPORTS:-nats,zmq}" \
  --trials        "${TRIALS:-3}" \
  --duration      "${DURATION:-40}" \
  --concurrency   "${CONCURRENCY:-256}" \
  --loadgen-procs "${LOADGEN_PROCS:-4}" \
  --speedup-ratio "${SPEEDUP_RATIO:-10}" \
  --infra-addr    "$NODE0" \
  --out           "$OUT"
