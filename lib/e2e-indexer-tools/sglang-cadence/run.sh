#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Usage: run.sh RUN LANES DURATION_S MAX_REQUESTS MAX_TOTAL_TOKENS [CHAIN_OFFSET] [EXTRA_SGLANG_ARGS...]
# Env: CADENCE_SHARED (shared dir holding these scripts in scripts/, workload-s8.jsonl, and out/),
#      CADENCE_LOCAL (node-local work dir with a venv that has aiohttp, msgpack, pyzmq),
#      SGLANG_IMAGE (default lmsysorg/sglang:v0.5.21). Run inside a GPU allocation with Docker.
set -uo pipefail
RUN=$1; LANES=$2; DUR=$3; MAXREQ=$4; KVTOK=$5; OFF=${6:-0}; shift 6 || shift $#
EXTRA=("$@")
S=${CADENCE_SHARED:?}
L=${CADENCE_LOCAL:?}
O=$L/out/$RUN
mkdir -p $O $L/hf $L/sglang-cache $L/xdg
PY=$L/venv/bin/python  # host venv with aiohttp, msgpack, pyzmq for the driver and recorder
PORT=30000
IMAGE=${SGLANG_IMAGE:-lmsysorg/sglang:v0.5.21}
cleanup() {
  [ -n "${REC:-}" ] && kill -TERM $REC 2>/dev/null && wait $REC 2>/dev/null
  [ -n "${CNAME:-}" ] && docker stop -t 30 $CNAME >/dev/null 2>&1
  mkdir -p $S/out/$RUN && cp $O/*.jsonl $O/*.log $O/*.json $S/out/$RUN/ 2>/dev/null
  echo "RUN_DONE $RUN"
}
trap cleanup EXIT
echo "{\"run\":\"$RUN\",\"lanes\":$LANES,\"duration_s\":$DUR,\"max_requests\":$MAXREQ,\"max_total_tokens\":$KVTOK,\"chain_offset\":$OFF,\"extra\":\"${EXTRA[*]}\",\"host\":\"$(hostname)\",\"start\":$(date +%s)}" > $O/config.json
CNAME=sgcad-$RUN-$$
mkdir -p $L/home
docker run -d --rm --name $CNAME --gpus all --network host --ipc host --shm-size 16g \
  --user $(id -u):$(id -g) -e HOME=$L/home -e HF_HOME=$L/hf -e XDG_CACHE_HOME=$L/xdg \
  -e SGLANG_CACHE_DIR=$L/sglang-cache -e TORCHINDUCTOR_CACHE_DIR=$L/xdg/inductor -e TRITON_CACHE_DIR=$L/xdg/triton \
  -v $L:$L $IMAGE \
  python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --host 127.0.0.1 --port $PORT \
  --page-size 1 --max-total-tokens $KVTOK --chunked-prefill-size 8192 --max-prefill-tokens 16384 \
  --context-length 40960 --mem-fraction-static 0.85 \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}' \
  "${EXTRA[@]}" > $O/docker-id.txt
echo "container $CNAME $(cat $O/docker-id.txt)"
docker logs -f $CNAME > $O/server.log 2>&1 &

for i in $(seq 1 600); do
  curl -sf http://127.0.0.1:$PORT/health_generate >/dev/null 2>&1 && break
  docker inspect -f "{{.State.Running}}" $CNAME 2>/dev/null | grep -q true || { echo "server died"; tail -50 $O/server.log; exit 1; }
  sleep 2
done
echo "server ready after ~$((i*2)) s"
# Recorder starts after warm-up health checks so their events are excluded; give SUB time to connect.
$PY $S/scripts/record_events.py --out $O/events.msgpack > $O/recorder.log 2>&1 &
REC=$!
sleep 3
$PY $S/scripts/drive.py --workload $S/workload-s8.jsonl --lanes $LANES --duration-s $DUR \
  --max-requests $MAXREQ --chain-offset $OFF --out $O/requests.jsonl | tee $O/driver.json
sleep 5
kill -TERM $REC; wait $REC; REC=
cat $O/recorder.log
ls -la $O
mkdir -p $S/out/$RUN && cp $O/events.msgpack $S/out/$RUN/
