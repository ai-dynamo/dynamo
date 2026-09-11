/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

// Image and command from inference-engine-instances at 334110c9676479224544a3879ebf8723d04ab3c7,
// kustomize/overlays/hou2-prod1/deployments/gpt-oss-20b-lp20-b300.yaml.
// The command is /bin/bash -lc; this is its complete, unmodified argument.
const pinnedCyborgImage = "us-docker.pkg.dev/nv-hwlpu-20260103060253/cyborg/cyborg:0.157.0@sha256:e13b373f9a08ab9dc1d7d4c618180815aa3bbac471f0ca25426da79e8437f908"

const pinnedCyborgLauncher = `set -euo pipefail

if [[ -z "${GBUILD_MANIFEST_PATH:-}" ]]; then
  echo "GBUILD_MANIFEST_PATH was not provided by the Dynamo operator" >&2
  exit 1
fi
if [[ -z "${CYBORG_BATCH_SIZE:-}" ]]; then
  echo "CYBORG_BATCH_SIZE was not provided by the Dynamo operator" >&2
  exit 1
fi
MODEL_RUNTIME_DIR="${GBUILD_MANIFEST_PATH%/*}"
. /usr/local/bin/k8s_gpu_detect.sh
set_cuda_visible_devices_from_nvidia_mounts

trap 'exit 0' TERM INT

ulimit -l unlimited

LPU_RDMA_ADDR_FILE="$(mktemp)"
/usr/local/bin/render_grove_lpu_addr_files \
  --server-hosts-file "${SERVER_HOSTS_FILE}" \
  --rdma-addr-file "${LPU_RDMA_ADDR_FILE}"

cyborg_cmd=(
  /usr/local/bin/cyborg
  --lpu-rdma-addr-file "${LPU_RDMA_ADDR_FILE}" \
  --gpu 0 \
  --model-family gpt-oss \
  --max-inflight-tasks 1 \
  --num-layers 12 \
  --num-heads-q 64 \
  --head-dim-qk 64 \
  --head-dim-v 64 \
  --num-heads-kv 8 \
  --seqlen-kv 131072 \
  --tokens-per-page 32 \
  --hidden-size 2880 \
  --stop-tokens 200002 \
  --swa-block-size 128 \
  --swa-padding-size 128 \
  --lpu-tail-top-k 40 \
  --standalone-post-decoder \
  --embedding-path "${MODEL_RUNTIME_DIR}/runtime/text_embeddings.npz" \
  --attention-sinks-path "${MODEL_RUNTIME_DIR}/cuda/attention_sinks.npz" \
  --batch-size "${CYBORG_BATCH_SIZE}" \
  --nic auto \
  --dtype bf16 \
  --backend fpga_gpi \
  --dynamo-listen-port 9124 \
  --dynamo-readiness-port 9090 \
  --dynamo-model-name "${MODEL_NAME}" \
  --tokenizer-dir "${TOKENIZER_DIR}" \
  --dynamo-namespace "${DYN_NAMESPACE:-${DYN_PARENT_DGD_K8S_NAMESPACE:-inference-engine}}" \
  --dynamo-component "${DYN_COMPONENT:-decode}" \
  --dynamo-tool-call-parser harmony \
  --dynamo-reasoning-parser gpt_oss \
  --openai-http-port "${OPENAI_HTTP_PORT}" \
  --openai-http-head-host "${POD_IP}"
)

cyborg_nsys_enable="${CYBORG_NSYS_ENABLE:-false}"
if [[ "${cyborg_nsys_enable,,}" == "true" || "${cyborg_nsys_enable}" == "1" ]]; then
  nsys_session_name="${NSYS_SESSION_NAME:-cyborg}"
  nsys_profile_dir="${NSYS_PROFILE_DIR:-/profiles}"
  nsys_trace="${NSYS_TRACE:-cuda-sw,nvtx}"
  nsys_cuda_graph_trace="${NSYS_CUDA_GRAPH_TRACE:-node}"
  command -v nsys >/dev/null
  mkdir -p "${nsys_profile_dir}"
  exec nsys launch \
    --session-new "${nsys_session_name}" \
    --trace="${nsys_trace}" \
    --cuda-graph-trace="${nsys_cuda_graph_trace}" \
    --resolve-symbols=false \
    "${cyborg_cmd[@]}"
fi

exec "${cyborg_cmd[@]}"
`
