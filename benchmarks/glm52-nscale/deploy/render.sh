#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
set -a
source "${ROOT_DIR}/campaign.env"
glm52_require_cluster_env NAMESPACE NODE_NAME EXPECTED_GPU_UUIDS
set +a

OUT_DIR="${SCRIPT_DIR}/rendered"
mkdir -p "${OUT_DIR}"

for template in "${SCRIPT_DIR}"/templates/*.yaml; do
  output="${OUT_DIR}/$(basename "${template}")"
  envsubst '${NAMESPACE} ${NODE_NAME} ${EXPECTED_GPU_UUIDS} ${MODEL_PATH} ${MODEL_REVISION} ${SERVED_MODEL_NAME} ${TP_SIZE} ${MAX_MODEL_LEN} ${MAX_RUNNING_REQUESTS} ${VLLM_IMAGE} ${SGLANG_IMAGE}' \
    < "${template}" > "${output}"
done

echo "Rendered manifests in ${OUT_DIR}"
