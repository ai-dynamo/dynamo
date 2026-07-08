#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly EXPECTED_FLASHINFER_SHA=330cc8e1a09f59c1241084459f3df3204b9b8327
readonly EXPECTED_VLLM_GIT_URL=https://github.com/galletas1712/vllm.git
readonly EXPECTED_FLASHINFER_GIT_URL=https://github.com/galletas1712/flashinfer.git
readonly EXPECTED_FLASHINFER_GIT_REF=schwinns/checkpoint-collectives-integration
readonly PROVENANCE_FILE=/opt/dynamo/source-provenance.txt
readonly OVERLAY_PROVENANCE_FILE=/opt/dynamo/vllm-overlay-provenance.txt
readonly FLASHINFER_SHA_FILE=/opt/dynamo/flashinfer-source-sha.txt

readonly -a OVERLAY_PATHS=(
    vllm/distributed/device_communicators/all2all.py
    vllm/distributed/device_communicators/base_device_communicator.py
    vllm/distributed/device_communicators/cuda_communicator.py
    vllm/distributed/device_communicators/flashinfer_all_reduce.py
    vllm/distributed/parallel_state.py
    vllm/envs.py
    vllm/model_executor/warmup/kernel_warmup.py
    vllm/utils/mem_utils.py
    vllm/v1/attention/backends/flashinfer.py
    vllm/v1/engine/core.py
    vllm/v1/worker/gpu/attn_utils.py
    vllm/v1/worker/gpu_model_runner.py
    vllm/v1/worker/gpu_worker.py
)

readonly -a CURRENT_EXPECTED_DIFF=(
    "A	tests/utils/test_checkpoint_memory_cleanup.py"
    "A	tests/v1/attention/test_flashinfer_workspace_experiment.py"
    "A	tests/v1/worker/test_checkpoint_memory_cleanup.py"
    "M	tests/distributed/test_weight_transfer.py"
    "M	vllm/distributed/device_communicators/all2all.py"
    "M	vllm/distributed/device_communicators/base_device_communicator.py"
    "M	vllm/distributed/device_communicators/cuda_communicator.py"
    "M	vllm/distributed/device_communicators/flashinfer_all_reduce.py"
    "M	vllm/distributed/parallel_state.py"
    "M	vllm/envs.py"
    "M	vllm/model_executor/warmup/kernel_warmup.py"
    "M	vllm/utils/mem_utils.py"
    "M	vllm/v1/attention/backends/flashinfer.py"
    "M	vllm/v1/engine/core.py"
    "M	vllm/v1/worker/gpu/attn_utils.py"
    "M	vllm/v1/worker/gpu_model_runner.py"
    "M	vllm/v1/worker/gpu_worker.py"
)

readonly -a CROSSOVER_EXPECTED_DIFF=(
    "A	tests/utils/test_checkpoint_memory_cleanup.py"
    "A	tests/v1/attention/test_flashinfer_workspace_experiment.py"
    "A	tests/v1/worker/test_checkpoint_memory_cleanup.py"
    "M	vllm/distributed/device_communicators/all2all.py"
    "M	vllm/distributed/device_communicators/base_device_communicator.py"
    "M	vllm/distributed/device_communicators/cuda_communicator.py"
    "M	vllm/distributed/device_communicators/flashinfer_all_reduce.py"
    "M	vllm/distributed/parallel_state.py"
    "M	vllm/envs.py"
    "M	vllm/model_executor/warmup/kernel_warmup.py"
    "M	vllm/utils/mem_utils.py"
    "M	vllm/v1/attention/backends/flashinfer.py"
    "M	vllm/v1/engine/core.py"
    "M	vllm/v1/worker/gpu/attn_utils.py"
    "M	vllm/v1/worker/gpu_model_runner.py"
    "M	vllm/v1/worker/gpu_worker.py"
)

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_exact_sha() {
    local name=$1
    local actual=$2
    local expected=$3
    [[ "${actual}" == "${expected}" ]] ||
        die "${name} is ${actual:-<empty>}, expected ${expected}"
}

select_tuple() {
    local base_image=${VLLM_RUNTIME_BASE_IMAGE:-}
    base_image=${base_image#docker.io/}

    case "${base_image}|${VLLM_GIT_SHA:-}" in
        vllm/vllm-openai@sha256:184914ac7c32e4aa7789bb686bfaa0817dd56dbdc8ee05fc0ec671aa0b1792f0\|17355f6f668857d9b85e0e7714529b42757e0730)
            OVERLAY_TUPLE=current-697158
            EXPECTED_BASE_INDEX_DIGEST=sha256:184914ac7c32e4aa7789bb686bfaa0817dd56dbdc8ee05fc0ec671aa0b1792f0
            EXPECTED_AMD64_DIGEST=sha256:1fd4323d0aafe8d92b4a4b568ad33661ecaf3bfc7f40860c95d09fed4e6ccd58
            EXPECTED_BASE_COMMIT=69715823df89b11ee684b84066390cbb9092d5c1
            EXPECTED_VLLM_HEAD=17355f6f668857d9b85e0e7714529b42757e0730
            EXPECTED_VLLM_GIT_REF=schwinns/exp-cuda-zero-page-234
            EXPECTED_OVERLAY_FILES=13
            EXPECTED_BASELINE_SBOM=vllm-openai@184914ac
            EXPECTED_DIFF=("${CURRENT_EXPECTED_DIFF[@]}")
            ;;
        vllm/vllm-openai@sha256:5da1eb79b49d3edb3b3601a116273f019adb7cab403e86790f61130f8596810a\|7e48076f13710677c223daf6e4e1af039c0f016e)
            OVERLAY_TUPLE=crossover-93d8
            EXPECTED_BASE_INDEX_DIGEST=sha256:7c5a10e9a8b3c8642f4d0463a41215176c0dd834b4f0967287c7e3e517cf1be9
            EXPECTED_AMD64_DIGEST=sha256:5da1eb79b49d3edb3b3601a116273f019adb7cab403e86790f61130f8596810a
            EXPECTED_BASE_COMMIT=93d8f834dd8acf33eb0e2a75b2711b628cb6e226
            EXPECTED_VLLM_HEAD=7e48076f13710677c223daf6e4e1af039c0f016e
            EXPECTED_VLLM_GIT_REF=schwinns/exp-93d8-current-overlay-zero-regression-20260708t082747z
            EXPECTED_OVERLAY_FILES=13
            EXPECTED_BASELINE_SBOM=vllm-openai@7c5a10e9
            EXPECTED_DIFF=("${CROSSOVER_EXPECTED_DIFF[@]}")
            ;;
        *)
            die "unknown nightly overlay tuple: ${VLLM_RUNTIME_BASE_IMAGE:-<empty>} / ${VLLM_GIT_SHA:-<empty>}"
            ;;
    esac
    require_exact_sha VLLM_GIT_URL "${VLLM_GIT_URL:-}" \
        "${EXPECTED_VLLM_GIT_URL}"
    require_exact_sha VLLM_GIT_REF "${VLLM_GIT_REF:-}" \
        "${EXPECTED_VLLM_GIT_REF}"
    require_exact_sha FLASHINFER_GIT_URL "${FLASHINFER_GIT_URL:-}" \
        "${EXPECTED_FLASHINFER_GIT_URL}"
    require_exact_sha FLASHINFER_GIT_REF "${FLASHINFER_GIT_REF:-}" \
        "${EXPECTED_FLASHINFER_GIT_REF}"
    require_exact_sha FLASHINFER_GIT_SHA "${FLASHINFER_GIT_SHA:-}" \
        "${EXPECTED_FLASHINFER_SHA}"
}

emit_tuple() {
    printf '%s\n' \
        "vllm_overlay_tuple=${OVERLAY_TUPLE}" \
        "vllm_base_index_digest=${EXPECTED_BASE_INDEX_DIGEST}" \
        "vllm_amd64_digest=${EXPECTED_AMD64_DIGEST}" \
        "vllm_base_commit=${EXPECTED_BASE_COMMIT}" \
        "vllm_head=${EXPECTED_VLLM_HEAD}" \
        "vllm_overlay_files=${EXPECTED_OVERLAY_FILES}" \
        "vllm_baseline_sbom=${EXPECTED_BASELINE_SBOM}"
}

clone_source() {
    local url=$1
    local ref=$2
    local sha=$3
    local destination=$4
    local with_submodules=${5:-0}

    [[ -n "${url}" ]] || die "source URL is required"
    [[ -n "${ref}" ]] || die "source ref is required"
    [[ "${sha}" =~ ^[0-9a-f]{40}$ ]] || die "source SHA must be 40 lowercase hex"

    rm -rf "${destination}"
    git clone --branch "${ref}" "${url}" "${destination}"
    require_exact_sha "remote branch ${ref}" \
        "$(git -C "${destination}" rev-parse "refs/remotes/origin/${ref}")" "${sha}"
    git -C "${destination}" checkout --detach "${sha}"
    if [[ "${with_submodules}" == 1 ]]; then
        git -C "${destination}" submodule update --init --recursive
    fi
}

validate_overlay_source() {
    local source=$1
    local -a actual_diff expected_diff

    require_exact_sha VLLM_GIT_SHA "$(git -C "${source}" rev-parse HEAD)" \
        "${EXPECTED_VLLM_HEAD}"
    git -C "${source}" cat-file -e "${EXPECTED_BASE_COMMIT}^{commit}"
    git -C "${source}" merge-base --is-ancestor "${EXPECTED_BASE_COMMIT}" HEAD
    [[ "$(git -C "${source}" rev-list --count "${EXPECTED_BASE_COMMIT}..HEAD")" \
        == 4 ]] || die "vLLM overlay must contain exactly four commits"

    mapfile -t actual_diff < <(
        git -C "${source}" diff --no-renames --name-status \
            "${EXPECTED_BASE_COMMIT}..HEAD" | LC_ALL=C sort
    )
    mapfile -t expected_diff < <(printf '%s\n' "${EXPECTED_DIFF[@]}" | LC_ALL=C sort)
    [[ "$(printf '%s\n' "${actual_diff[@]}")" == \
       "$(printf '%s\n' "${expected_diff[@]}")" ]] ||
        die "unexpected vLLM overlay delta"
}

install_overlay() {
    local source=$1
    local package_dir=$2
    local path

    : > "${OVERLAY_PROVENANCE_FILE}"
    for path in "${OVERLAY_PATHS[@]}"; do
        [[ -f "${source}/${path}" ]] || die "missing overlay source file: ${path}"
        [[ -f "${package_dir}/${path#vllm/}" ]] ||
            die "nightly package is missing overlay destination: ${path}"
        install -m 0644 "${source}/${path}" "${package_dir}/${path#vllm/}"
        (cd "${package_dir}/.." && sha256sum "${path}") \
            >> "${OVERLAY_PROVENANCE_FILE}"
    done
}

select_tuple
if [[ "${1:-}" == "select" ]]; then
    emit_tuple
    exit 0
fi

[[ "${VLLM_INSTALL_MODE:-}" == "python-overlay" ]] ||
    die "installer requires VLLM_INSTALL_MODE=python-overlay"
[[ "$(uname -m)" == "x86_64" ]] || die "nightly overlay is x86_64-only"
require_exact_sha VLLM_GIT_SHA "${VLLM_GIT_SHA:-}" "${EXPECTED_VLLM_HEAD}"
require_exact_sha VLLM_BUILD_COMMIT "${VLLM_BUILD_COMMIT:-}" \
    "${EXPECTED_BASE_COMMIT}"
require_exact_sha VLLM_EXPECTED_BASE_INDEX_DIGEST \
    "${VLLM_EXPECTED_BASE_INDEX_DIGEST:-}" "${EXPECTED_BASE_INDEX_DIGEST}"
require_exact_sha VLLM_EXPECTED_AMD64_DIGEST \
    "${VLLM_EXPECTED_AMD64_DIGEST:-}" "${EXPECTED_AMD64_DIGEST}"
require_exact_sha VLLM_EXPECTED_BASE_COMMIT \
    "${VLLM_EXPECTED_BASE_COMMIT:-}" "${EXPECTED_BASE_COMMIT}"
require_exact_sha VLLM_EXPECTED_VLLM_HEAD \
    "${VLLM_EXPECTED_VLLM_HEAD:-}" "${EXPECTED_VLLM_HEAD}"
require_exact_sha VLLM_EXPECTED_OVERLAY_FILES \
    "${VLLM_EXPECTED_OVERLAY_FILES:-}" "${EXPECTED_OVERLAY_FILES}"
require_exact_sha VLLM_EXPECTED_BASELINE_SBOM \
    "${VLLM_EXPECTED_BASELINE_SBOM:-}" "${EXPECTED_BASELINE_SBOM}"
require_exact_sha BASELINE_SBOM_FILE "${BASELINE_SBOM_FILE:-}" \
    "${EXPECTED_BASELINE_SBOM}"

[[ -z "${VLLM_NCCL_SO_PATH:-}" ]] ||
    die "nightly overlay must use the stock NCCL selection"
[[ -z "${LD_PRELOAD:-}" ]] || die "nightly overlay must not set LD_PRELOAD"

mkdir -p /opt/dynamo
python3 /usr/local/lib/validate_nightly_overlay.py capture

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates cuda-nvrtc-dev-13-0 git
rm -rf /var/lib/apt/lists/*

clone_source "${VLLM_GIT_URL:-}" "${VLLM_GIT_REF:-}" "${VLLM_GIT_SHA:-}" \
    /tmp/vllm-overlay-src
validate_overlay_source /tmp/vllm-overlay-src
vllm_package_dir="$(
    python3 - <<'PY'
import importlib.metadata as metadata

print(metadata.distribution("vllm").locate_file("vllm").resolve())
PY
)"
install_overlay /tmp/vllm-overlay-src "${vllm_package_dir}"
rm -rf /tmp/vllm-overlay-src

clone_source \
    "${FLASHINFER_GIT_URL:-}" \
    "${FLASHINFER_GIT_REF:-}" \
    "${FLASHINFER_GIT_SHA:-}" \
    /tmp/flashinfer-src \
    1
cd /tmp/flashinfer-src
flashinfer_source_version="$(tr -d '[:space:]' < version.txt)"
[[ "${flashinfer_source_version}" == "0.6.14" ]] ||
    die "FlashInfer source is ${flashinfer_source_version}, expected 0.6.14"
printf '%s\n' "${EXPECTED_FLASHINFER_SHA}" > "${FLASHINFER_SHA_FILE}"
BUILD_NVEP=0 BUILD_NCCL_EP=0 BUILD_NIXL_EP=0 \
    uv pip install --system --force-reinstall --no-build-isolation --no-deps .
BUILD_NVEP=0 BUILD_NCCL_EP=0 BUILD_NIXL_EP=0 \
    uv pip install --system --force-reinstall --no-build-isolation --no-deps \
        ./flashinfer-cubin
uv pip uninstall --system flashinfer-jit-cache || true
cd /
rm -rf /tmp/flashinfer-src

cat > "${PROVENANCE_FILE}" <<EOF
install_mode=python-overlay
vllm_overlay_tuple=${OVERLAY_TUPLE}
vllm_runtime_base_image=${VLLM_RUNTIME_BASE_IMAGE}
vllm_runtime_base_index_digest=${EXPECTED_BASE_INDEX_DIGEST}
vllm_runtime_amd64_digest=${EXPECTED_AMD64_DIGEST}
vllm_base_commit=${EXPECTED_BASE_COMMIT}
vllm_git_url=${EXPECTED_VLLM_GIT_URL}
vllm_git_ref=${EXPECTED_VLLM_GIT_REF}
vllm_source_sha=${EXPECTED_VLLM_HEAD}
vllm_overlay_files=${EXPECTED_OVERLAY_FILES}
compliance_baseline_sbom=${EXPECTED_BASELINE_SBOM}
flashinfer_git_url=${EXPECTED_FLASHINFER_GIT_URL}
flashinfer_git_ref=${EXPECTED_FLASHINFER_GIT_REF}
flashinfer_source_sha=${EXPECTED_FLASHINFER_SHA}
flashinfer_source_version=${flashinfer_source_version}
EOF
