#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${CUDA_VERSION:?CUDA_VERSION must be set}"
: "${FLASHINF_REF:?FLASHINF_REF must be set}"

VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_PRECOMPILED_WHEEL_COMMIT:-}"
VLLM_PRECOMPILED_WHEEL_VARIANT="${VLLM_PRECOMPILED_WHEEL_VARIANT:-}"
PROVENANCE_FILE=/opt/dynamo/source-provenance.txt

clone_source() {
    local url=$1
    local ref=$2
    local sha=$3
    local destination=$4

    rm -rf "${destination}"
    if [[ -n "${ref}" ]]; then
        git clone --recurse-submodules --branch "${ref}" "${url}" "${destination}"
    else
        git clone --recurse-submodules "${url}" "${destination}"
    fi

    cd "${destination}"
    if [[ -n "${sha}" ]]; then
        RESOLVED_SOURCE_SHA="$(git rev-parse "${sha}^{commit}")"
        git checkout --detach "${RESOLVED_SOURCE_SHA}"
        [[ "$(git rev-parse HEAD)" == "${RESOLVED_SOURCE_SHA}" ]]
        git submodule update --init --recursive
    else
        RESOLVED_SOURCE_SHA="$(git rev-parse HEAD)"
    fi
}

require_git_selector() {
    local name=$1
    local ref=$2
    local sha=$3

    if [[ -z "${ref}" && -z "${sha}" ]]; then
        echo "${name}_GIT_URL requires ${name}_GIT_REF or ${name}_GIT_SHA" >&2
        exit 1
    fi
}

validate_exact_native_mode() {
    if [[ ! "${VLLM_PRECOMPILED_WHEEL_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]]; then
        echo "VLLM_PRECOMPILED_WHEEL_COMMIT must be a full 40-character SHA" >&2
        exit 1
    fi
    if [[ -z "${VLLM_PRECOMPILED_WHEEL_VARIANT}" ]]; then
        echo "VLLM_PRECOMPILED_WHEEL_VARIANT must be set in exact-native mode" >&2
        exit 1
    fi
    if [[ ! "${VLLM_PRECOMPILED_WHEEL_VARIANT}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
        echo "VLLM_PRECOMPILED_WHEEL_VARIANT contains invalid characters" >&2
        exit 1
    fi
    if [[ "$(uname -m)" != "x86_64" ]]; then
        echo "Exact vLLM native wheels are only supported on x86_64" >&2
        exit 1
    fi

    VLLM_PRECOMPILED_WHEEL_COMMIT="$(
        printf '%s' "${VLLM_PRECOMPILED_WHEEL_COMMIT}" | tr '[:upper:]' '[:lower:]'
    )"
    local metadata_url
    metadata_url="https://wheels.vllm.ai/${VLLM_PRECOMPILED_WHEEL_COMMIT}/${VLLM_PRECOMPILED_WHEEL_VARIANT}/vllm/metadata.json"
    echo "Validating exact native wheel metadata: ${metadata_url}"
    VLLM_PRECOMPILED_WHEEL_LOCATION="$(
        python3 - "${metadata_url}" <<'PY'
import json
import sys
from urllib.parse import urljoin
from urllib.request import urlopen

url = sys.argv[1]
with urlopen(url, timeout=30) as response:
    metadata = json.load(response)

matches = [
    wheel
    for wheel in metadata
    if wheel.get("package_name") == "vllm"
    and wheel.get("platform_tag") == "manylinux_2_28_x86_64"
    and wheel.get("path")
]
if not matches:
    raise SystemExit(
        f"{url} has no x86_64 manylinux_2_28 vLLM wheel: {metadata!r}"
    )
wheel = matches[0]
print(f"Validated native wheel: {wheel['filename']}", file=sys.stderr)
print(urljoin(url, wheel["path"]))
PY
    )"
    export VLLM_PRECOMPILED_WHEEL_LOCATION
}

verify_vllm_install() {
    REQUIRE_CHECKPOINT_HOOKS="${1:-}" python3 <<'PY'
import os
import importlib.metadata as metadata
import importlib.util
from pathlib import Path

from packaging.utils import canonicalize_name

distributions = [
    dist
    for dist in metadata.distributions()
    if canonicalize_name(dist.metadata["Name"]) == "vllm"
]
if len(distributions) != 1:
    locations = [str(dist.locate_file("").resolve()) for dist in distributions]
    raise SystemExit(f"Expected one vllm distribution, found {locations!r}")

distribution = distributions[0]
package_dir = Path(distribution.locate_file("vllm")).resolve()
extensions = sorted(package_dir.rglob("*.so"))
if not any(path.name.startswith("_C.") for path in extensions):
    raise SystemExit(f"vllm._C extension file is missing under {package_dir}")
extension_spec = importlib.util.find_spec("vllm._C")
if extension_spec is None or extension_spec.origin is None:
    raise SystemExit("vllm._C extension spec is missing")

print(f"Installed vLLM version: {distribution.version}")
print(f"Installed vLLM package: {package_dir}")
print("Installed vLLM extensions:")
for extension in extensions:
    print(f"  {extension}")

if os.environ["REQUIRE_CHECKPOINT_HOOKS"]:
    from vllm.v1.worker.gpu_worker import Worker

    assert hasattr(Worker, "checkpoint_prepare")
    assert hasattr(Worker, "checkpoint_restore")
    print("Verified vLLM checkpoint_prepare/checkpoint_restore worker methods")
PY
}

mkdir -p /opt/dynamo
: > "${PROVENANCE_FILE}"

if [[ -n "${VLLM_PRECOMPILED_WHEEL_COMMIT}" && -z "${VLLM_GIT_URL:-}" ]]; then
    echo "VLLM_PRECOMPILED_WHEEL_COMMIT requires VLLM_GIT_URL" >&2
    exit 1
fi
if [[ -n "${VLLM_GIT_URL:-}" || -n "${FLASHINFER_GIT_URL:-}" ]]; then
    apt-get update
    packages=(ca-certificates git)
    if [[ -n "${FLASHINFER_GIT_URL:-}" ]]; then
        cuda_version_rest="${CUDA_VERSION#*.}"
        if [[ "${cuda_version_rest}" == "${CUDA_VERSION}" ]]; then
            echo "CUDA_VERSION must include major.minor: ${CUDA_VERSION}" >&2
            exit 1
        fi
        cuda_minor_dash="${CUDA_VERSION%%.*}-${cuda_version_rest%%.*}"
        packages+=("cuda-nvrtc-dev-${cuda_minor_dash}")
    fi
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        "${packages[@]}"
    rm -rf /var/lib/apt/lists/*
fi

if [[ -n "${VLLM_PRECOMPILED_WHEEL_COMMIT}" ]]; then
    validate_exact_native_mode
fi

if [[ -n "${VLLM_GIT_URL:-}" ]]; then
    require_git_selector VLLM "${VLLM_GIT_REF:-}" "${VLLM_GIT_SHA:-}"
    clone_source \
        "${VLLM_GIT_URL}" \
        "${VLLM_GIT_REF:-}" \
        "${VLLM_GIT_SHA:-}" \
        /tmp/vllm-src
    vllm_source_sha="${RESOLVED_SOURCE_SHA}"

    if [[ -n "${VLLM_PRECOMPILED_WHEEL_COMMIT}" ]]; then
        if ! git cat-file -e "${VLLM_PRECOMPILED_WHEEL_COMMIT}^{commit}"; then
            echo "Native wheel commit is absent from the custom vLLM clone: ${VLLM_PRECOMPILED_WHEEL_COMMIT}" >&2
            exit 1
        fi
        if ! git merge-base --is-ancestor \
            "${VLLM_PRECOMPILED_WHEEL_COMMIT}" "${vllm_source_sha}"; then
            echo "Native wheel commit is not an ancestor of custom vLLM HEAD" >&2
            exit 1
        fi

        offending_paths="$(
            git diff --name-only \
                "${VLLM_PRECOMPILED_WHEEL_COMMIT}..${vllm_source_sha}" |
                sed -n '\|^vllm/.*\.py$|!p'
        )"
        if [[ -n "${offending_paths}" ]]; then
            echo "Exact-native mode only permits Python changes under vllm/:" >&2
            printf '%s\n' "${offending_paths}" >&2
            exit 1
        fi

        uv pip uninstall --system vllm
        VLLM_USE_PRECOMPILED=1 \
        VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_PRECOMPILED_WHEEL_COMMIT}" \
        VLLM_PRECOMPILED_WHEEL_VARIANT="${VLLM_PRECOMPILED_WHEEL_VARIANT}" \
        VLLM_PRECOMPILED_WHEEL_LOCATION="${VLLM_PRECOMPILED_WHEEL_LOCATION}" \
        VLLM_DOCKER_BUILD_CONTEXT=1 \
            uv pip install --system '.[flashinfer,runai,otel]' \
                --torch-backend=auto
        install_mode=exact-native-source
        native_wheel_commit="${VLLM_PRECOMPILED_WHEEL_COMMIT}"
        native_wheel_variant="${VLLM_PRECOMPILED_WHEEL_VARIANT}"
    else
        vllm_package_dir="$(
            cd /
            python3 -c 'import importlib.metadata as m; print(m.distribution("vllm").locate_file("vllm").resolve())'
        )"
        cp -a /tmp/vllm-src/vllm/. "${vllm_package_dir}/"
        install_mode=python-overlay
        native_wheel_commit=
        native_wheel_variant=
    fi
    cd /
    rm -rf /tmp/vllm-src
else
    install_mode=upstream-runtime
    vllm_source_sha=upstream-runtime
    native_wheel_commit=
    native_wheel_variant=
fi

{
    echo "install_mode=${install_mode}"
    echo "vllm_source_sha=${vllm_source_sha}"
    echo "vllm_native_wheel_commit=${native_wheel_commit}"
    echo "vllm_native_wheel_variant=${native_wheel_variant}"
} >> "${PROVENANCE_FILE}"

# This override intentionally runs after vLLM's dependency solve. In
# exact-native mode that prevents the source install's stock FlashInfer
# requirement from replacing the checkpoint-aware fork.
if [[ -n "${FLASHINFER_GIT_URL:-}" ]]; then
    require_git_selector \
        FLASHINFER "${FLASHINFER_GIT_REF:-}" "${FLASHINFER_GIT_SHA:-}"
    clone_source \
        "${FLASHINFER_GIT_URL}" \
        "${FLASHINFER_GIT_REF:-}" \
        "${FLASHINFER_GIT_SHA:-}" \
        /tmp/flashinfer-src
    flashinfer_source_sha="${RESOLVED_SOURCE_SHA}"
    uv pip install --system --force-reinstall --no-deps .
    if [[ -d ./flashinfer-cubin ]]; then
        uv pip install --system --force-reinstall --no-deps ./flashinfer-cubin
    fi
    uv pip uninstall --system flashinfer-jit-cache || true
    cd /
    rm -rf /tmp/flashinfer-src
else
    flashinfer_source_sha=upstream-runtime
    echo "Using FlashInfer from the vLLM runtime/dependency solve (${FLASHINF_REF})."
fi

verify_vllm_install "${VLLM_GIT_URL:-}"
python3 <<'PY'
import importlib.metadata as metadata

distribution = metadata.distribution("flashinfer-python")
print(f"Installed FlashInfer version: {distribution.version}")
print(f"Installed FlashInfer location: {distribution.locate_file('').resolve()}")
PY
echo "flashinfer_source_sha=${flashinfer_source_sha}" >> "${PROVENANCE_FILE}"

echo "Installed source/native provenance:"
cat "${PROVENANCE_FILE}"
