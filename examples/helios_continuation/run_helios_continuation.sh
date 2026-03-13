#!/bin/bash
# Run Helios-Distilled with latent cache continuation via Dynamo runtime.
#
# Architecture:
#   etcd (discovery) -> Worker (sglang pipeline on GPU) -> HTTP Server (no GPU)
#
# The worker loads the sglang Helios pipeline directly (bypassing DiffGenerator's
# multiprocessing) and patches the denoising stage to support latent caching,
# enabling continuous video generation across multiple requests.
#
# Usage:
#   ./run_helios_continuation.sh [MODEL_PATH]
#
# Test:
#   # Generate first segment:
#   curl -X POST http://localhost:8090/v1/videos/generate \
#     -H 'Content-Type: application/json' \
#     -d '{"prompt": "A cat walking on grass", "num_frames": 33, "height": 384, "width": 640}'
#
#   # Continue from cache:
#   curl -X POST http://localhost:8090/v1/videos/continue \
#     -H 'Content-Type: application/json' \
#     -d '{"cache_id": "<cache_id>", "num_frames": 33}'
#
#   # List / delete caches:
#   curl http://localhost:8090/v1/caches
#   curl -X DELETE http://localhost:8090/v1/caches/<cache_id>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python}"
ETCD="${ETCD:-etcd}"

MODEL_PATH="${1:-BestWishYsh/Helios-Distilled}"
export MODEL_PATH
export HTTP_PORT="${HTTP_PORT:-8090}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"
export CACHE_DIR="${CACHE_DIR:-${SCRIPT_DIR}/latent_caches}"

LOGDIR="${SCRIPT_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo "  Helios Continuation via Dynamo + sglang"
echo "  Model:    ${MODEL_PATH}"
echo "  HTTP:     http://localhost:${HTTP_PORT}"
echo "  Output:   ${OUTPUT_DIR}"
echo "  Caches:   ${CACHE_DIR}"
echo "  Log:      ${LOGDIR}/continuation_${TIMESTAMP}.log"
echo "============================================"

mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"

PIDS=()
cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    pkill -f "etcd --data-dir /tmp/etcd_helios_cont" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

# 1. Start etcd
echo "[1/3] Starting etcd..."
rm -rf /tmp/etcd_helios_cont
$ETCD --data-dir /tmp/etcd_helios_cont \
     --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://localhost:2379 \
     > /tmp/etcd_helios_cont.log 2>&1 &
PIDS+=($!)
sleep 2

# 2. Start Helios Worker (GPU) with pipeline-direct backend
echo "[2/3] Starting Helios Continuation Worker (GPU ${CUDA_VISIBLE_DEVICES:-0})..."
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" MODE=worker \
    $PYTHON "${SCRIPT_DIR}/helios_continuation_dynamo.py" \
    2>&1 | tee -a "${LOGDIR}/continuation_${TIMESTAMP}.log" | sed 's/^/  [worker] /' &
PIDS+=($!)

# Wait for pipeline loading (~40s on cached model)
echo "Waiting for pipeline to load (~50s)..."
sleep 50

# 3. Start HTTP Server (no GPU)
echo "[3/3] Starting HTTP Server on port ${HTTP_PORT}..."
CUDA_VISIBLE_DEVICES="" MODE=server \
    $PYTHON "${SCRIPT_DIR}/helios_continuation_dynamo.py" \
    2>&1 | tee -a "${LOGDIR}/continuation_${TIMESTAMP}.log" | sed 's/^/  [server] /'

# Server runs in foreground; cleanup on exit
