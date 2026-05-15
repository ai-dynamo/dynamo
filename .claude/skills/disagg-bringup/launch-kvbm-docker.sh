#!/bin/bash
# KVBM disagg bringup for Docker-based nodes (dlcluster b100_preprod, 4u4g, etc.)
#
# Hard-won lessons from 2026-05-15 session:
#
# 1. RUSTUP_HOME and CARGO_HOME must be passed via `docker exec -e`, NOT via
#    `export` inside bash -c. The Docker image sets RUSTUP_HOME=/home/dynamo/.rustup
#    as an ENV, which overrides any shell export when running as a non-dynamo user.
#
# 2. The kvbm Python extension (lib_core.so) is built with `cargo rustc --lib
#    --crate-type cdylib --features v1,v2`. The default features include v1 which
#    enables dynamo-llm/block-manager, but sometimes explicit --features is needed.
#    Without this the linker produces a 0-byte empty lib_core.so with no error.
#
# 3. maturin develop fails on NFS with "Malformed entity: Object is too small" when
#    it tries to parse lib_core.so. Bypass maturin by:
#      a) cargo rustc --lib --crate-type cdylib --features v1,v2  (produces ~1.4GB debug .so)
#      b) cp debug/lib_core.so python/kvbm/_core.abi3.so
#      c) cp debug/deps/libkvbm_kernels.so python/kvbm/libkvbm_kernels.so
#      d) Set PYTHONPATH=/workspace/lib/bindings/kvbm/python
#
# 4. Use CARGO_TARGET_DIR on NFS (/scratch/cargo-target-kvbm) to survive docker restarts.
#    Artifacts in /tmp are lost on docker restart.
#
# 5. The hub must be started BEFORE the vLLM workers. Use start-hub.sh which sets
#    the correct --prefill-vllm-url pointing to the prefill API port.
#
# Usage:
#   CONTAINER=kvbm-b100 MODEL=Qwen/Qwen3-8B bash launch-kvbm-docker.sh
#
# Prereqs:
#   - Docker container running: docker run -d --name $CONTAINER --gpus '"device=0,1"' \
#       --net=host -v /home/scratch.mkosec_hw/worktrees/dynamo-pfaas:/workspace \
#       -v /home/scratch.mkosec_hw:/scratch \
#       -e HOME=/tmp nvcr.io/nvidian/dynamo-dev/vllm-dev:mkosec-a3a41eb9975 sleep infinity
#   - kvbm built (see step 1 below)

set -euo pipefail

CONTAINER="${CONTAINER:-kvbm-b100}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
CARGO_TARGET="/scratch/cargo-target-kvbm"
WORKSPACE="/workspace"
VENV="/opt/dynamo/venv"
PYTHONPATH_KVBM="${WORKSPACE}/lib/bindings/kvbm/python"
HUB_BIN="${CARGO_TARGET}/debug/kvbm_hub"
HUB_PORT=1337
PREFILL_PORT=8001
DECODE_PORT=8000

# Env vars that MUST be passed via -e (not export inside bash -c)
DOCKER_ENV="-e RUSTUP_HOME=/scratch/.rustup -e CARGO_HOME=/scratch/.cargo -e CARGO_TARGET_DIR=${CARGO_TARGET}"

dexec() {
  docker exec ${DOCKER_ENV} -e PYTHONPATH="${PYTHONPATH_KVBM}" "${CONTAINER}" bash -c "$@"
}

dexec_env() {
  local extra_env="$1"; shift
  docker exec ${DOCKER_ENV} ${extra_env} -e PYTHONPATH="${PYTHONPATH_KVBM}" "${CONTAINER}" bash -c "$@"
}

echo "=== Step 1: Build kvbm Python extension and hub ==="
echo "Building cdylib with explicit features (v1,v2 to enable dynamo-llm/block-manager)..."
dexec "mkdir -p /scratch/.cargo && source ${VENV}/bin/activate && cd ${WORKSPACE}/lib/bindings/kvbm && cargo rustc --manifest-path Cargo.toml --lib --crate-type cdylib --features v1,v2 2>&1 | tail -3"

echo "Copying extension files..."
# lib_core.so is the compiled Python extension (~1.4GB debug build)
cp -fv "${CARGO_TARGET}/debug/lib_core.so" \
       "${WORKSPACE}/lib/bindings/kvbm/python/kvbm/_core.abi3.so"
cp -fv "${CARGO_TARGET}/debug/deps/libkvbm_kernels.so" \
       "${WORKSPACE}/lib/bindings/kvbm/python/kvbm/libkvbm_kernels.so"

echo "Verifying import..."
dexec "source ${VENV}/bin/activate && python3 -c 'import kvbm; print(\"kvbm OK:\", kvbm.__file__)'"

echo "Building kvbm_hub binary..."
dexec "cd ${WORKSPACE} && cargo build -p kvbm-hub --bin kvbm_hub 2>&1 | tail -3"
echo "Hub binary: $(ls -lh ${CARGO_TARGET}/debug/kvbm_hub 2>/dev/null || echo 'MISSING')"

echo ""
echo "=== Step 2: Launch hub ==="
docker exec -d \
  ${DOCKER_ENV} \
  -e PYTHONPATH="${PYTHONPATH_KVBM}" \
  -e HF_HOME=/scratch/hf_cache \
  -e KVBM_HUB_BIN="${HUB_BIN}" \
  -e KVBM_HUB_MODEL="${MODEL}" \
  -e KVBM_HUB_PREFILL_URL="http://127.0.0.1:${PREFILL_PORT}" \
  "${CONTAINER}" bash "${WORKSPACE}/.claude/skills/disagg-bringup/start-hub.sh" /tmp/hub.log
echo "Hub started on port ${HUB_PORT}"
sleep 3

echo "=== Step 3: Launch prefill (GPU 0, port ${PREFILL_PORT}) ==="
KVBM_PREFILL_CFG="{\"kv_connector\":\"DynamoConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_module_path\":\"kvbm.v2.vllm.schedulers.connector\",\"kv_connector_extra_config\":{\"leader\":{\"disagg\":{\"hub_url\":\"http://127.0.0.1:${HUB_PORT}\",\"role\":\"prefill\"},\"cache\":{\"host\":{\"cache_size_gb\":4.0}},\"tokio\":{\"worker_threads\":4}},\"worker\":{\"nixl\":{\"backends\":{\"POSIX\":{}}},\"tokio\":{\"worker_threads\":4}}}}"
docker exec -d \
  -e PYTHONPATH="${PYTHONPATH_KVBM}" \
  -e HF_HOME=/scratch/hf_cache \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e CUDA_VISIBLE_DEVICES=0 \
  "${CONTAINER}" bash -c "source ${VENV}/bin/activate && ${VENV}/bin/python3 -m vllm.entrypoints.openai.api_server --model ${MODEL} --max-model-len 16384 --gpu-memory-utilization 0.45 --port ${PREFILL_PORT} --kv-transfer-config '${KVBM_PREFILL_CFG}' > /tmp/prefill.log 2>&1"
echo "Prefill started on GPU 0, port ${PREFILL_PORT}"
sleep 3

echo "=== Step 4: Launch decode (GPU 1, port ${DECODE_PORT}) ==="
KVBM_DECODE_CFG="{\"kv_connector\":\"DynamoConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_module_path\":\"kvbm.v2.vllm.schedulers.connector\",\"kv_connector_extra_config\":{\"leader\":{\"disagg\":{\"hub_url\":\"http://127.0.0.1:${HUB_PORT}\",\"role\":\"decode\"},\"cache\":{\"host\":{\"cache_size_gb\":4.0}},\"tokio\":{\"worker_threads\":4}},\"worker\":{\"nixl\":{\"backends\":{\"POSIX\":{}}},\"tokio\":{\"worker_threads\":4}}}}"
docker exec -d \
  -e PYTHONPATH="${PYTHONPATH_KVBM}" \
  -e HF_HOME=/scratch/hf_cache \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e CUDA_VISIBLE_DEVICES=1 \
  "${CONTAINER}" bash -c "source ${VENV}/bin/activate && ${VENV}/bin/python3 -m vllm.entrypoints.openai.api_server --model ${MODEL} --max-model-len 16384 --gpu-memory-utilization 0.45 --port ${DECODE_PORT} --kv-transfer-config '${KVBM_DECODE_CFG}' > /tmp/decode.log 2>&1"
echo "Decode started on GPU 1, port ${DECODE_PORT}"

echo ""
echo "=== Waiting for decode health on :${DECODE_PORT} ==="
until curl -s "http://localhost:${DECODE_PORT}/health" | grep -q healthy; do
  sleep 5; echo -n "."
done
echo ""
echo "READY — run AIPerf against http://$(hostname -I | awk '{print $1}'):${DECODE_PORT}"
