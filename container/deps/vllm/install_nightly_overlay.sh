#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly EXPECTED_BASE_COMMIT=dcfebf93f4eccf30f71872283331eee757915daf
readonly EXPECTED_BASE_DIGEST=sha256:7f2bc168366c77fbd8329368f00310d208531c14ece6c2de31a6611ef99f6ec8
readonly EXPECTED_AMD64_DIGEST=sha256:99e7dd3cf74c489af0615671f3fdbde182de2930f1195a0ee39e914e38033a88
readonly EXPECTED_VLLM_URL=https://github.com/vllm-project/vllm.git
readonly EXPECTED_VLLM_REF=refs/pull/46877/head
readonly EXPECTED_VLLM_HEAD=af259f998ff7301504829d2551c746502afe2f0a
readonly EXPECTED_VLLM_HEAD_TREE=bf3b2afdc9082606129662909c5a417df9a8d533
readonly EXPECTED_MERGE_BASE=c4f5cd60dae386d106c9b8a12dbab24e2e9dda0b
readonly EXPECTED_COMPOSED_TREE=72e7896e8bd04ba92d9ee6c446875c3745fc2668
readonly EXPECTED_FLASHINFER_URL=https://github.com/galletas1712/flashinfer.git
readonly EXPECTED_FLASHINFER_REF=refs/heads/experiment/flashinfer-v0.6.15-pr3950-provenance-20260720
readonly EXPECTED_FLASHINFER_SHA=12a51a30ce011b08eb673cb4387db6d9f67945b1
readonly EXPECTED_FLASHINFER_RELEASE=8eccd0c1352165302840c0e19066bc42d36dbd7a
readonly EXPECTED_FLASHINFER_VERSION=0.6.15
readonly PROVENANCE_FILE=/opt/dynamo/source-provenance.txt
readonly OVERLAY_PROVENANCE_FILE=/opt/dynamo/vllm-overlay-provenance.txt
readonly FLASHINFER_SHA_FILE=/opt/dynamo/flashinfer-source-sha.txt

readonly -a OVERLAY_PATHS=(
    vllm/distributed/device_communicators/all2all.py
    vllm/distributed/device_communicators/base_device_communicator.py
    vllm/distributed/device_communicators/cuda_communicator.py
    vllm/distributed/device_communicators/flashinfer_all_reduce.py
    vllm/distributed/parallel_state.py
    vllm/v1/worker/gpu_worker.py
)

readonly -a EXPECTED_DIFF=(
    "M	vllm/distributed/device_communicators/all2all.py"
    "M	vllm/distributed/device_communicators/base_device_communicator.py"
    "M	vllm/distributed/device_communicators/cuda_communicator.py"
    "M	vllm/distributed/device_communicators/flashinfer_all_reduce.py"
    "M	vllm/distributed/parallel_state.py"
    "M	vllm/v1/worker/gpu_worker.py"
)

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_exact() {
    local name=$1
    local actual=$2
    local expected=$3
    [[ "${actual}" == "${expected}" ]] ||
        die "${name} is ${actual:-<empty>}, expected ${expected}"
}

fetch_exact_ref() {
    local repository=$1
    local ref=$2
    local sha=$3
    local destination=$4
    local depth=${5:-256}
    local remote_ref_sha

    [[ "${sha}" =~ ^[0-9a-f]{40}$ ]] ||
        die "source SHA must be 40 lowercase hex"
    rm -rf "${destination}"
    git init --quiet "${destination}"
    git -C "${destination}" remote add origin "${repository}"
    if [[ "${ref}" != "${sha}" ]]; then
        remote_ref_sha="$(
            git -C "${destination}" ls-remote --exit-code origin "${ref}" \
                | awk 'NR == 1 { print $1 } END { if (NR != 1) exit 1 }'
        )"
        require_exact "remote ref ${ref}" "${remote_ref_sha}" "${sha}"
    fi
    git -C "${destination}" fetch --quiet --no-tags --depth "${depth}" origin "${sha}"
    require_exact "fetched source" \
        "$(git -C "${destination}" rev-parse FETCH_HEAD)" "${sha}"
}

validate_and_compose_vllm() {
    local source=$1
    local actual_tree
    local -a actual_diff expected_diff

    require_exact "vLLM PR tree" \
        "$(git -C "${source}" rev-parse "${EXPECTED_VLLM_HEAD}^{tree}")" \
        "${EXPECTED_VLLM_HEAD_TREE}"
    require_exact "vLLM PR merge base" \
        "$(git -C "${source}" merge-base \
            "${EXPECTED_BASE_COMMIT}" "${EXPECTED_VLLM_HEAD}")" \
        "${EXPECTED_MERGE_BASE}"
    require_exact "vLLM PR commit count" \
        "$(git -C "${source}" rev-list --count \
            "${EXPECTED_MERGE_BASE}..${EXPECTED_VLLM_HEAD}")" \
        5

    git -C "${source}" checkout --quiet --detach "${EXPECTED_BASE_COMMIT}"
    if ! git -C "${source}" \
        -c user.name=Dynamo \
        -c user.email=dynamo@nvidia.com \
        merge --quiet --no-commit --no-ff "${EXPECTED_VLLM_HEAD}"; then
        git -C "${source}" merge --abort || true
        die "vLLM nightly and PR #46877 do not merge cleanly"
    fi
    actual_tree="$(git -C "${source}" write-tree)"
    git -C "${source}" merge --abort
    require_exact "composed vLLM tree" "${actual_tree}" "${EXPECTED_COMPOSED_TREE}"

    mapfile -t actual_diff < <(
        git -C "${source}" diff --no-renames --name-status \
            "${EXPECTED_BASE_COMMIT}" "${EXPECTED_COMPOSED_TREE}" | LC_ALL=C sort
    )
    mapfile -t expected_diff < <(
        printf '%s\n' "${EXPECTED_DIFF[@]}" | LC_ALL=C sort
    )
    [[ "$(printf '%s\n' "${actual_diff[@]}")" == \
       "$(printf '%s\n' "${expected_diff[@]}")" ]] ||
        die "unexpected composed vLLM delta"
}

install_overlay() {
    local source=$1
    local package_dir=$2
    local path destination

    : > "${OVERLAY_PROVENANCE_FILE}"
    for path in "${OVERLAY_PATHS[@]}"; do
        destination="${package_dir}/${path#vllm/}"
        [[ -f "${destination}" ]] ||
            die "nightly package is missing overlay destination: ${path}"
        git -C "${source}" show "${EXPECTED_BASE_COMMIT}:${path}" \
            | cmp --silent - "${destination}" ||
            die "installed nightly file does not match ${EXPECTED_BASE_COMMIT}: ${path}"
        git -C "${source}" show "${EXPECTED_COMPOSED_TREE}:${path}" \
            > "${destination}.dynamo-overlay"
        install -m 0644 "${destination}.dynamo-overlay" "${destination}"
        rm -f "${destination}.dynamo-overlay"
        (cd "${package_dir}/.." && sha256sum "${path}") \
            >> "${OVERLAY_PROVENANCE_FILE}"
    done
}

validate_flashinfer_source() {
    local source=$1
    local path

    require_exact "FlashInfer source" \
        "$(git -C "${source}" rev-parse "${EXPECTED_FLASHINFER_SHA}")" \
        "${EXPECTED_FLASHINFER_SHA}"
    require_exact "FlashInfer release parent" \
        "$(git -C "${source}" rev-parse "${EXPECTED_FLASHINFER_SHA}^")" \
        "${EXPECTED_FLASHINFER_RELEASE}"
    require_exact "FlashInfer provenance tree" \
        "$(git -C "${source}" rev-parse "${EXPECTED_FLASHINFER_SHA}^{tree}")" \
        "$(git -C "${source}" rev-parse "${EXPECTED_FLASHINFER_RELEASE}^{tree}")"
    require_exact "FlashInfer #3950 implementation blob" \
        "$(git -C "${source}" rev-parse \
            "${EXPECTED_FLASHINFER_SHA}:flashinfer/comm/trtllm_mnnvl_ar.py")" \
        3ab1d3258545837b66d7beed340311f8f86317f2
    require_exact "FlashInfer #3950 test blob" \
        "$(git -C "${source}" rev-parse \
            "${EXPECTED_FLASHINFER_SHA}:tests/comm/test_trtllm_allreduce_checkpoint.py")" \
        54bd6a96935b4773e5a691bff90790b6c746e6b6
    path="$(git -C "${source}" rev-parse \
        "${EXPECTED_FLASHINFER_SHA}:version.txt")"
    [[ -n "${path}" ]] || die "FlashInfer version.txt is missing"
}

[[ "${VLLM_INSTALL_MODE:-}" == "python-overlay" ]] ||
    die "installer requires VLLM_INSTALL_MODE=python-overlay"
[[ "$(uname -m)" == "x86_64" ]] || die "nightly overlay is x86_64-only"
require_exact VLLM_RUNTIME_BASE_IMAGE "${VLLM_RUNTIME_BASE_IMAGE:-}" \
    "vllm/vllm-openai@${EXPECTED_BASE_DIGEST}"
require_exact VLLM_GIT_URL "${VLLM_GIT_URL:-}" "${EXPECTED_VLLM_URL}"
require_exact VLLM_GIT_REF "${VLLM_GIT_REF:-}" "${EXPECTED_VLLM_REF}"
require_exact VLLM_GIT_SHA "${VLLM_GIT_SHA:-}" "${EXPECTED_VLLM_HEAD}"
require_exact FLASHINFER_GIT_URL "${FLASHINFER_GIT_URL:-}" \
    "${EXPECTED_FLASHINFER_URL}"
require_exact FLASHINFER_GIT_REF "${FLASHINFER_GIT_REF:-}" \
    "${EXPECTED_FLASHINFER_REF}"
require_exact FLASHINFER_GIT_SHA "${FLASHINFER_GIT_SHA:-}" \
    "${EXPECTED_FLASHINFER_SHA}"
require_exact GMS_SNAPSHOT_PROFILE "${GMS_SNAPSHOT_PROFILE:-}" 1
[[ "${DYNAMO_COMMIT_SHA:-}" =~ ^[0-9a-f]{40}$ ]] ||
    die "DYNAMO_COMMIT_SHA must be 40 lowercase hex"

[[ -z "${VLLM_NCCL_SO_PATH:-}" ]] ||
    die "nightly overlay must use the stock NCCL selection"
[[ -z "${NCCL_CHECKPOINT_SHIM:-}" ]] ||
    die "nightly overlay must not use an NCCL checkpoint shim"
[[ -z "${LD_PRELOAD:-}" ]] || die "nightly overlay must not set LD_PRELOAD"

mkdir -p /opt/dynamo
python3 /usr/local/lib/validate_nightly_overlay.py capture

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates cuda-nvrtc-dev-13-0 git
rm -rf /var/lib/apt/lists/*
nvrtc_package_version="$(dpkg-query -W -f='${Version}' cuda-nvrtc-dev-13-0)"
[[ -n "${nvrtc_package_version}" ]] ||
    die "cuda-nvrtc-dev-13-0 package version is empty"

fetch_exact_ref \
    "${VLLM_GIT_URL}" \
    "${VLLM_GIT_REF}" \
    "${VLLM_GIT_SHA}" \
    /tmp/vllm-overlay-src
git -C /tmp/vllm-overlay-src fetch \
    --quiet --no-tags --depth 256 origin "${EXPECTED_BASE_COMMIT}"
require_exact "fetched nightly source" \
    "$(git -C /tmp/vllm-overlay-src rev-parse FETCH_HEAD)" \
    "${EXPECTED_BASE_COMMIT}"
validate_and_compose_vllm /tmp/vllm-overlay-src
vllm_package_dir="$(
    python3 - <<'PY'
import importlib.metadata as metadata

print(metadata.distribution("vllm").locate_file("vllm").resolve())
PY
)"
install_overlay /tmp/vllm-overlay-src "${vllm_package_dir}"
rm -rf /tmp/vllm-overlay-src

fetch_exact_ref \
    "${FLASHINFER_GIT_URL}" \
    "${FLASHINFER_GIT_REF}" \
    "${FLASHINFER_GIT_SHA}" \
    /tmp/flashinfer-src \
    2
git -C /tmp/flashinfer-src checkout --quiet --detach "${FLASHINFER_GIT_SHA}"
git -C /tmp/flashinfer-src submodule update --init --recursive
validate_flashinfer_source /tmp/flashinfer-src
cd /tmp/flashinfer-src
flashinfer_source_version="$(tr -d '[:space:]' < version.txt)"
require_exact "FlashInfer source version" "${flashinfer_source_version}" \
    "${EXPECTED_FLASHINFER_VERSION}"
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
dynamo_source_sha=${DYNAMO_COMMIT_SHA}
snapshot_profile=1
install_mode=python-overlay
vllm_runtime_base_image=${VLLM_RUNTIME_BASE_IMAGE}
vllm_runtime_index_digest=${EXPECTED_BASE_DIGEST}
vllm_runtime_amd64_digest=${EXPECTED_AMD64_DIGEST}
vllm_base_commit=${EXPECTED_BASE_COMMIT}
vllm_source_url=${EXPECTED_VLLM_URL}
vllm_source_ref=${EXPECTED_VLLM_REF}
vllm_source_sha=${EXPECTED_VLLM_HEAD}
vllm_source_tree=${EXPECTED_VLLM_HEAD_TREE}
vllm_merge_base=${EXPECTED_MERGE_BASE}
vllm_composed_tree=${EXPECTED_COMPOSED_TREE}
vllm_pr_commits=5
vllm_overlay_files=${#OVERLAY_PATHS[@]}
flashinfer_source_url=${EXPECTED_FLASHINFER_URL}
flashinfer_source_ref=${EXPECTED_FLASHINFER_REF}
flashinfer_source_sha=${EXPECTED_FLASHINFER_SHA}
flashinfer_release_commit=${EXPECTED_FLASHINFER_RELEASE}
flashinfer_source_version=${flashinfer_source_version}
flashinfer_pr3950_head=243d56c12cf1c724bdb128a4575ebf6ce1e8a1a9
flashinfer_pr3950_squash=28b51d807a87b1f8f2ed09b77cef976e737991c4
flashinfer_equivalent_fix=d9d0d175741afad0230860c0f56a6307085e9186
nvrtc_package_version=${nvrtc_package_version}
EOF
