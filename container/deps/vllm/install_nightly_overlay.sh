#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly EXPECTED_BASE_COMMIT=93d8f834dd8acf33eb0e2a75b2711b628cb6e226
readonly EXPECTED_BASE_DIGEST=sha256:5da1eb79b49d3edb3b3601a116273f019adb7cab403e86790f61130f8596810a
readonly EXPECTED_VLLM_HEAD=7cd9fdf23860d2ab3e34de4b34bbfa553ada4207
readonly EXPECTED_FLASHINFER_SHA=330cc8e1a09f59c1241084459f3df3204b9b8327
readonly PROVENANCE_FILE=/opt/dynamo/source-provenance.txt
readonly OVERLAY_PROVENANCE_FILE=/opt/dynamo/vllm-overlay-provenance.txt
readonly FLASHINFER_VERSION_FILE=/opt/dynamo/flashinfer-source-version.txt
readonly FLASHINFER_SHA_FILE=/opt/dynamo/flashinfer-source-sha.txt

readonly -a OVERLAY_PATHS=(
    vllm/distributed/device_communicators/all2all.py
    vllm/distributed/device_communicators/base_device_communicator.py
    vllm/distributed/device_communicators/cuda_communicator.py
    vllm/distributed/device_communicators/flashinfer_all_reduce.py
    vllm/distributed/parallel_state.py
    vllm/model_executor/warmup/kernel_warmup.py
    vllm/v1/worker/gpu_model_runner.py
)

readonly -a EXPECTED_DIFF=(
    "A	tests/model_executor/warmup/test_kernel_warmup.py"
    "M	vllm/distributed/device_communicators/all2all.py"
    "M	vllm/distributed/device_communicators/base_device_communicator.py"
    "M	vllm/distributed/device_communicators/cuda_communicator.py"
    "M	vllm/distributed/device_communicators/flashinfer_all_reduce.py"
    "M	vllm/distributed/parallel_state.py"
    "M	vllm/model_executor/warmup/kernel_warmup.py"
    "M	vllm/v1/worker/gpu_model_runner.py"
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
    require_exact_sha "resolved source SHA" \
        "$(git -C "${destination}" rev-parse HEAD)" "${sha}"
    if [[ "${with_submodules}" == 1 ]]; then
        git -C "${destination}" submodule update --init --recursive
    fi
}

validate_overlay_source() {
    local source=$1
    local head
    local -a actual_diff

    head="$(git -C "${source}" rev-parse HEAD)"
    require_exact_sha VLLM_GIT_SHA "${head}" "${EXPECTED_VLLM_HEAD}"
    git -C "${source}" cat-file -e "${EXPECTED_BASE_COMMIT}^{commit}"
    git -C "${source}" merge-base --is-ancestor "${EXPECTED_BASE_COMMIT}" "${head}"
    [[ "$(git -C "${source}" rev-list --count "${EXPECTED_BASE_COMMIT}..${head}")" \
        == 3 ]] || die "vLLM overlay must contain exactly three feature commits"

    mapfile -t actual_diff < <(
        git -C "${source}" diff --no-renames --name-status \
            "${EXPECTED_BASE_COMMIT}..${head}" | LC_ALL=C sort
    )
    mapfile -t expected_diff < <(printf '%s\n' "${EXPECTED_DIFF[@]}" | LC_ALL=C sort)
    if [[ "$(printf '%s\n' "${actual_diff[@]}")" != \
          "$(printf '%s\n' "${expected_diff[@]}")" ]]; then
        echo "Unexpected vLLM overlay delta:" >&2
        printf '  %s\n' "${actual_diff[@]}" >&2
        exit 1
    fi

    if git -C "${source}" grep -n -E \
        'VLLM_ALLGATHER_USE_FLASHINFER|_flashinfer_all_gather|fi_ag_workspaces|SymmetricAllGatherWorkspace' \
        -- "${OVERLAY_PATHS[@]}"; then
        die "symmetric FlashInfer all-gather is forbidden in this overlay"
    fi
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
        (
            cd "${package_dir}/.."
            sha256sum "${path}"
        ) >> "${OVERLAY_PROVENANCE_FILE}"
    done
    [[ "$(wc -l < "${OVERLAY_PROVENANCE_FILE}")" == 7 ]] ||
        die "overlay provenance must contain seven files"
}

[[ "${VLLM_INSTALL_MODE:-}" == "python-overlay" ]] ||
    die "install_nightly_overlay requires VLLM_INSTALL_MODE=python-overlay"
[[ "$(uname -m)" == "x86_64" ]] || die "nightly overlay is x86_64-only"
[[ "${VLLM_RUNTIME_BASE_IMAGE##*@}" == "${EXPECTED_BASE_DIGEST}" ]] ||
    die "nightly overlay requires base manifest ${EXPECTED_BASE_DIGEST}"
require_exact_sha VLLM_BUILD_COMMIT "${VLLM_BUILD_COMMIT:-}" \
    "${EXPECTED_BASE_COMMIT}"
require_exact_sha VLLM_GIT_SHA "${VLLM_GIT_SHA:-}" "${EXPECTED_VLLM_HEAD}"
require_exact_sha FLASHINFER_GIT_SHA "${FLASHINFER_GIT_SHA:-}" \
    "${EXPECTED_FLASHINFER_SHA}"

[[ -z "${VLLM_PRECOMPILED_WHEEL_COMMIT:-}" ]] ||
    die "python-overlay does not accept precompiled replacement wheels"
[[ -z "${VLLM_TORCH_VERSION:-}" ]] ||
    die "python-overlay does not accept a Torch replacement"
[[ -z "${VLLM_TORCHVISION_VERSION:-}" ]] ||
    die "python-overlay does not accept a torchvision replacement"
[[ -z "${VLLM_NCCL_VERSION:-}" ]] ||
    die "python-overlay does not accept an NCCL replacement"
[[ -z "${NCCL_CHECKPOINT_VERSION:-}" ]] ||
    die "python-overlay does not accept an NCCL checkpoint shim"
[[ -z "${VLLM_NCCL_SO_PATH:-}" ]] ||
    die "python-overlay must use the nightly's stock NCCL selection"
[[ -z "${LD_PRELOAD:-}" ]] || die "python-overlay must not set LD_PRELOAD"

mkdir -p /opt/dynamo
python3 /usr/local/lib/validate_nightly_overlay.py capture

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates git
rm -rf /var/lib/apt/lists/*

clone_source \
    "${VLLM_GIT_URL:-}" \
    "${VLLM_GIT_REF:-}" \
    "${VLLM_GIT_SHA:-}" \
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
printf '%s\n' "${flashinfer_source_version}" > "${FLASHINFER_VERSION_FILE}"

BUILD_NVEP=0 BUILD_NCCL_EP=0 BUILD_NIXL_EP=0 \
    uv pip install --system --force-reinstall --no-build-isolation --no-deps .
BUILD_NVEP=0 BUILD_NCCL_EP=0 BUILD_NIXL_EP=0 \
    uv pip install --system --force-reinstall --no-build-isolation --no-deps \
        ./flashinfer-cubin
uv pip uninstall --system flashinfer-jit-cache

cd /
rm -rf /tmp/flashinfer-src

cat > "${PROVENANCE_FILE}" <<EOF
install_mode=python-overlay
vllm_runtime_base_image=${VLLM_RUNTIME_BASE_IMAGE}
vllm_base_commit=${EXPECTED_BASE_COMMIT}
vllm_source_sha=${EXPECTED_VLLM_HEAD}
vllm_overlay_files=7
flashinfer_source_sha=${EXPECTED_FLASHINFER_SHA}
flashinfer_source_version=${flashinfer_source_version}
EOF

echo "Installed nightly overlay provenance:"
cat "${PROVENANCE_FILE}"
cat "${OVERLAY_PROVENANCE_FILE}"
