#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly EXPECTED_BASE_COMMIT=2c17d33f4291a55b447317640c81eb61077b1b00
readonly EXPECTED_BASE_DIGEST=sha256:5bda7078b1bb17f74d369e3ded63115a77d5ea5eeb9eab6ca9a52d108f9a262d
readonly EXPECTED_AMD64_DIGEST=sha256:1ebb205a272a55abb60b09ecbf2adc63831ef2377910afd527478de720788cd8
readonly EXPECTED_VLLM_HEAD=b198491a7163c636d58ee1796c433d83a9ec1b0d
readonly EXPECTED_FLASHINFER_SHA=f2f9646ec388d9f178b2fbda6ae0ec4246d8e7dc
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
    vllm/utils/mem_utils.py
    vllm/v1/attention/backends/flashinfer.py
    vllm/v1/engine/core.py
    vllm/v1/worker/gpu/attn_utils.py
    vllm/v1/worker/gpu_model_runner.py
    vllm/v1/worker/gpu_worker.py
)

readonly -a EXPECTED_DIFF=(
    "A	tests/utils/test_checkpoint_memory_cleanup.py"
    "A	tests/v1/attention/test_flashinfer_workspace_experiment.py"
    "A	tests/v1/worker/test_checkpoint_memory_cleanup.py"
    "M	vllm/distributed/device_communicators/all2all.py"
    "M	vllm/distributed/device_communicators/base_device_communicator.py"
    "M	vllm/distributed/device_communicators/cuda_communicator.py"
    "M	vllm/distributed/device_communicators/flashinfer_all_reduce.py"
    "M	vllm/distributed/parallel_state.py"
    "M	vllm/envs.py"
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
        == 3 ]] || die "vLLM overlay must contain exactly three commits"

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

[[ "${VLLM_INSTALL_MODE:-}" == "python-overlay" ]] ||
    die "installer requires VLLM_INSTALL_MODE=python-overlay"
[[ "$(uname -m)" == "x86_64" ]] || die "nightly overlay is x86_64-only"
[[ "${VLLM_RUNTIME_BASE_IMAGE##*@}" == "${EXPECTED_BASE_DIGEST}" ]] ||
    die "nightly overlay requires base manifest ${EXPECTED_BASE_DIGEST}"
require_exact_sha VLLM_GIT_SHA "${VLLM_GIT_SHA:-}" "${EXPECTED_VLLM_HEAD}"
require_exact_sha FLASHINFER_GIT_SHA "${FLASHINFER_GIT_SHA:-}" \
    "${EXPECTED_FLASHINFER_SHA}"

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
[[ "${flashinfer_source_version}" == "0.6.15" ]] ||
    die "FlashInfer source is ${flashinfer_source_version}, expected 0.6.15"
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
vllm_runtime_base_image=${VLLM_RUNTIME_BASE_IMAGE}
vllm_runtime_amd64_digest=${EXPECTED_AMD64_DIGEST}
vllm_base_commit=${EXPECTED_BASE_COMMIT}
vllm_source_sha=${EXPECTED_VLLM_HEAD}
vllm_overlay_files=12
flashinfer_source_sha=${EXPECTED_FLASHINFER_SHA}
flashinfer_source_version=${flashinfer_source_version}
EOF
