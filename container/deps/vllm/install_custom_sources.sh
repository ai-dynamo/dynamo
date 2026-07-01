#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${CUDA_VERSION:?CUDA_VERSION must be set}"
: "${FLASHINF_REF:?FLASHINF_REF must be set}"

VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_PRECOMPILED_WHEEL_COMMIT:-}"
VLLM_PRECOMPILED_WHEEL_VARIANT="${VLLM_PRECOMPILED_WHEEL_VARIANT:-}"
VLLM_INSTALL_MODE="${VLLM_INSTALL_MODE:-auto}"
VLLM_TORCH_VERSION="${VLLM_TORCH_VERSION:-}"
VLLM_TORCHVISION_VERSION="${VLLM_TORCHVISION_VERSION:-}"
VLLM_TORCH_BACKEND="${VLLM_TORCH_BACKEND:-}"
VLLM_TORCH_CUDA_ARCH_LIST="${VLLM_TORCH_CUDA_ARCH_LIST:-}"
VLLM_NCCL_VERSION="${VLLM_NCCL_VERSION:-${NCCL_CHECKPOINT_VERSION:-}}"
VLLM_EXPECTED_TORCH_LOCAL_VERSION=
PROVENANCE_FILE=/opt/dynamo/source-provenance.txt
FLASHINFER_VERSION_FILE=/opt/dynamo/flashinfer-source-version.txt
FLASHINFER_SHA_FILE=/opt/dynamo/flashinfer-source-sha.txt
VLLM_CONSTRAINTS_FILE=/tmp/vllm-full-source-constraints.txt
NCCL_DSO_LINK=/opt/dynamo/nccl/libnccl.so.2

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

configure_pip_nccl_dso() {
    mkdir -p "$(dirname "${NCCL_DSO_LINK}")"
    local nccl_dso
    nccl_dso="$(
        python3 - <<'PY'
import importlib.metadata as metadata
from pathlib import Path

distribution = metadata.distribution("nvidia-nccl-cu13")
candidate = Path(
    distribution.locate_file("nvidia/nccl/lib/libnccl.so.2")
).resolve()
if not candidate.is_file():
    raise SystemExit(f"nvidia-nccl-cu13 DSO is missing: {candidate}")
print(candidate)
PY
    )"
    ln -sfn "${nccl_dso}" "${NCCL_DSO_LINK}"
    test "$(readlink -f "${NCCL_DSO_LINK}")" = "${nccl_dso}"
    echo "Resolved pip NCCL DSO: ${nccl_dso}"
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

require_full_sha() {
    local name=$1
    local value=$2

    if [[ ! "${value}" =~ ^[0-9a-fA-F]{40}$ ]]; then
        echo "${name} must be a full 40-character SHA in full-source mode" >&2
        exit 1
    fi
}

validate_full_source_mode() {
    if [[ -n "${VLLM_PRECOMPILED_WHEEL_COMMIT}" ]]; then
        echo "full-source mode does not accept precompiled vLLM artifacts" >&2
        exit 1
    fi
    if [[ "$(uname -m)" != "x86_64" ]]; then
        echo "Full-source vLLM CUDA builds are only supported on x86_64" >&2
        exit 1
    fi

    require_full_sha VLLM_GIT_SHA "${VLLM_GIT_SHA:-}"
    require_full_sha FLASHINFER_GIT_SHA "${FLASHINFER_GIT_SHA:-}"
    if [[ "${VLLM_TORCH_VERSION}" != "2.12.0" ||
          "${VLLM_TORCHVISION_VERSION}" != "0.27.0" ||
          "${VLLM_TORCH_BACKEND}" != "cu130" ]]; then
        cat >&2 <<EOF
full-source mode requires the approved official PyTorch stack:
  VLLM_TORCH_VERSION=2.12.0
  VLLM_TORCHVISION_VERSION=0.27.0
  VLLM_TORCH_BACKEND=cu130
EOF
        exit 1
    fi
    if [[ "${VLLM_NCCL_VERSION}" != "2.29.7" ]]; then
        echo "full-source mode requires VLLM_NCCL_VERSION=2.29.7" >&2
        exit 1
    fi
    if [[ -z "${VLLM_TORCH_CUDA_ARCH_LIST}" ]]; then
        echo "VLLM_TORCH_CUDA_ARCH_LIST is required in full-source mode" >&2
        exit 1
    fi
    if [[ ! "${VLLM_RUNTIME_BASE_IMAGE:-}" =~ @sha256:[0-9a-f]{64}$ ]]; then
        echo "full-source mode requires an immutable VLLM_RUNTIME_BASE_IMAGE digest" >&2
        exit 1
    fi

    cat > "${VLLM_CONSTRAINTS_FILE}" <<EOF
torch==${VLLM_TORCH_VERSION}+${VLLM_TORCH_BACKEND}
torchvision==${VLLM_TORCHVISION_VERSION}+${VLLM_TORCH_BACKEND}
torchaudio==0
nvidia-nccl-cu13==${VLLM_NCCL_VERSION}
EOF

    VLLM_EXPECTED_TORCH_LOCAL_VERSION="${VLLM_TORCH_BACKEND}"
}

build_full_source_vllm() {
    if ! grep -qx 'torch==2.12.0' requirements/cuda.txt ||
       ! grep -qx 'torchvision==0.27.0 .*' requirements/cuda.txt ||
       grep -q '^torchaudio' requirements/cuda.txt ||
       ! grep -qx 'torch==2.12.0' requirements/build/cuda.txt ||
       ! grep -qx '    "torch == 2.12.0",' pyproject.toml ||
       ! grep -qx 'set(TORCH_SUPPORTED_VERSION_CUDA "2.12.0")' \
           CMakeLists.txt; then
        echo "Custom vLLM source does not contain the approved Torch 2.12 CUDA build metadata" >&2
        exit 1
    fi

    uv pip uninstall --system \
        vllm torch torchvision torchaudio nvidia-nccl-cu13
    uv pip install --system \
        --constraints "${VLLM_CONSTRAINTS_FILE}" \
        --torch-backend="${VLLM_TORCH_BACKEND}" \
        "torch==${VLLM_TORCH_VERSION}" \
        "torchvision==${VLLM_TORCHVISION_VERSION}" \
        "nvidia-nccl-cu13==${VLLM_NCCL_VERSION}"

    python3 - <<'PY'
import importlib.metadata as metadata

expected = {
    "torch": "2.12.0+cu130",
    "torchvision": "0.27.0+cu130",
    "nvidia-nccl-cu13": "2.29.7",
}
for name, expected_version in expected.items():
    actual = metadata.version(name)
    if actual != expected_version:
        raise SystemExit(f"{name} is {actual}, expected {expected_version}")
try:
    metadata.distribution("torchaudio")
except metadata.PackageNotFoundError:
    pass
else:
    raise SystemExit("torchaudio must be absent from the full-source runtime")
PY
    configure_pip_nccl_dso

    uv pip install --system \
        --constraints "${VLLM_CONSTRAINTS_FILE}" \
        --torch-backend="${VLLM_TORCH_BACKEND}" \
        -r requirements/build/cuda.txt
    uv pip install --system \
        --constraints "${VLLM_CONSTRAINTS_FILE}" \
        --torch-backend="${VLLM_TORCH_BACKEND}" \
        -r requirements/cuda.txt \
        "runai-model-streamer[s3,gcs,azure]>=0.15.7"

    ./tools/install_protoc.sh
    VLLM_RS_TARGET_PATH=/tmp/vllm-src/vllm/vllm-rs \
        bash ./build_rust.sh
    VLLM_TARGET_DEVICE=cuda \
    VLLM_USE_PRECOMPILED=0 \
    VLLM_USE_PRECOMPILED_RUST=0 \
    VLLM_REQUIRE_RUST_FRONTEND=1 \
    VLLM_DOCKER_BUILD_CONTEXT=1 \
    TORCH_CUDA_ARCH_LIST="${VLLM_TORCH_CUDA_ARCH_LIST}" \
    MAX_JOBS="${MAX_JOBS:-10}" \
    CMAKE_BUILD_TYPE=Release \
        uv pip install --system --no-build-isolation --no-deps .
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
    case "${VLLM_PRECOMPILED_WHEEL_VARIANT}" in
        cu130)
            VLLM_TORCH_BACKEND=cu130
            VLLM_EXPECTED_TORCH_LOCAL_VERSION=cu130
            ;;
        *)
            echo "Unsupported exact-native vLLM wheel variant: ${VLLM_PRECOMPILED_WHEEL_VARIANT}" \
                >&2
            echo "Supported CUDA variants: cu130" >&2
            exit 1
            ;;
    esac
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
    REQUIRE_CHECKPOINT_HOOKS="${1:-}" \
    REQUIRE_EXACT_NATIVE="${2:-}" \
    EXACT_NATIVE_VARIANT="${3:-}" \
    EXPECTED_TORCH_LOCAL_VERSION="${4:-}" \
    REQUIRE_FULL_SOURCE="${5:-}" \
    EXPECTED_TORCH_VERSION="${6:-}" \
        python3 <<'PY'
import ast
import importlib.metadata as metadata
import os
from importlib.machinery import PathFinder
from pathlib import Path


def canonicalize_name(name):
    return name.lower().replace("_", "-").replace(".", "-")


def require_distribution(name):
    distributions = [
        dist
        for dist in metadata.distributions()
        if canonicalize_name(dist.metadata["Name"]) == canonicalize_name(name)
    ]
    if len(distributions) != 1:
        locations = [str(dist.locate_file("").resolve()) for dist in distributions]
        raise SystemExit(
            f"Expected one {name} distribution, found {len(distributions)} "
            f"at {locations!r}"
        )
    return distributions[0]


distribution = require_distribution("vllm")
package_dir = Path(distribution.locate_file("vllm")).resolve()
extensions = sorted(package_dir.rglob("*.so"))
distribution_files = {str(path) for path in distribution.files or ()}
if os.environ["REQUIRE_EXACT_NATIVE"] or os.environ["REQUIRE_FULL_SOURCE"]:
    variant = os.environ["EXACT_NATIVE_VARIANT"]
    expected_torch_local = os.environ["EXPECTED_TORCH_LOCAL_VERSION"]
    if not variant or not expected_torch_local:
        raise SystemExit(
            "Native verification requires a CUDA variant and expected Torch "
            "local version"
        )

    torch_distribution = require_distribution("torch")
    torch_version = torch_distribution.version
    expected_torch_version = os.environ["EXPECTED_TORCH_VERSION"]
    if expected_torch_version and torch_version != expected_torch_version:
        raise SystemExit(
            f"torch version is {torch_version}, expected {expected_torch_version}"
        )
    torch_local_version = torch_version.partition("+")[2].lower()
    if torch_local_version != expected_torch_local:
        raise SystemExit(
            f"Exact-native variant {variant} requires torch local version "
            f"+{expected_torch_local}, found torch {torch_version}. "
            f"Install with uv --torch-backend={variant}."
        )

    if os.environ["REQUIRE_FULL_SOURCE"]:
        vllm_local_version = distribution.version.partition("+")[2].lower()
        vllm_local_tags = set(
            vllm_local_version.replace("-", ".").replace("_", ".").split(".")
        )
        if "cpu" in vllm_local_tags or "precompiled" in vllm_local_tags:
            raise SystemExit(
                "Full-source mode requires a locally built CUDA vLLM "
                f"distribution, found vllm {distribution.version}"
            )
        required_extensions = (
            "vllm/_C_stable_libtorch.abi3.so",
            "vllm/_moe_C_stable_libtorch.abi3.so",
            "vllm/cumem_allocator.abi3.so",
            "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so",
        )
    else:
        vllm_local_version = distribution.version.partition("+")[2].lower()
        vllm_local_tags = set(
            vllm_local_version.replace("-", ".").replace("_", ".").split(".")
        )
        if "cpu" in vllm_local_tags or "precompiled" not in vllm_local_tags:
            raise SystemExit(
                "Exact-native mode requires a CUDA precompiled vLLM "
                f"distribution, found vllm {distribution.version}. Ensure "
                "VLLM_TARGET_DEVICE=cuda and VLLM_USE_PRECOMPILED=1 during "
                "installation."
            )
        required_extensions = (
            "vllm/_C_stable_libtorch.abi3.so",
            "vllm/_moe_C_stable_libtorch.abi3.so",
        )

    # Do not import native modules during image builds because libcuda is
    # unavailable. Stable-libtorch modules still require libcuda at GPU runtime.
    for relative_path in required_extensions:
        if relative_path not in distribution_files:
            raise SystemExit(
                f"Exact-native vLLM distribution does not own {relative_path}"
            )
        extension = Path(distribution.locate_file(relative_path)).resolve()
        if not extension.is_file():
            raise SystemExit(f"Exact-native extension file is missing: {extension}")
else:
    if not any(path.name.startswith("_C.") for path in extensions):
        raise SystemExit(f"vllm._C extension file is missing under {package_dir}")
    extension_spec = PathFinder.find_spec("vllm._C", [str(package_dir)])
    if extension_spec is None or extension_spec.origin is None:
        raise SystemExit("vllm._C extension spec is missing")

print(f"Installed vLLM version: {distribution.version}")
print(f"Installed vLLM package: {package_dir}")
print("Installed vLLM extensions:")
for extension in extensions:
    print(f"  {extension}")

if os.environ["REQUIRE_CHECKPOINT_HOOKS"]:
    if os.environ["REQUIRE_EXACT_NATIVE"] or os.environ["REQUIRE_FULL_SOURCE"]:
        relative_path = "vllm/distributed/parallel_state.py"
        if relative_path not in distribution_files:
            raise SystemExit(
                f"Exact-native vLLM distribution does not own {relative_path}"
            )
        lifecycle_path = Path(distribution.locate_file(relative_path)).resolve()
        tree = ast.parse(lifecycle_path.read_text(), filename=str(lifecycle_path))
        functions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        missing_functions = {
            "checkpoint_prepare_distributed_state",
            "checkpoint_restore_distributed_state",
        } - functions
        if missing_functions:
            raise SystemExit(
                f"{lifecycle_path} is missing functions: "
                f"{sorted(missing_functions)!r}"
            )

        worker_relative_path = "vllm/v1/worker/gpu_worker.py"
        worker_path = Path(distribution.locate_file(worker_relative_path)).resolve()
        worker_tree = ast.parse(worker_path.read_text(), filename=str(worker_path))
        worker_classes = [
            node
            for node in worker_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "GPUWorker"
        ]
        forbidden_methods = {
            node.name
            for worker_class in worker_classes
            for node in worker_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        } & {
            "checkpoint_prepare",
            "checkpoint_restore",
        }
        if forbidden_methods:
            raise SystemExit(
                f"{worker_path} must not define checkpoint worker methods: "
                f"{sorted(forbidden_methods)!r}"
            )
        print(
            "Verified generic vLLM checkpoint lifecycle functions and no "
            "GPUWorker checkpoint override without importing vLLM"
        )
    else:
        from vllm.distributed.parallel_state import (
            checkpoint_prepare_distributed_state,
            checkpoint_restore_distributed_state,
        )
        from vllm.v1.worker.gpu_worker import GPUWorker

        assert callable(checkpoint_prepare_distributed_state)
        assert callable(checkpoint_restore_distributed_state)
        assert not hasattr(GPUWorker, "checkpoint_prepare")
        assert not hasattr(GPUWorker, "checkpoint_restore")
        print("Verified generic vLLM checkpoint lifecycle functions")
PY
}

validate_exact_native_changes() {
    local base=$1
    local head=$2
    local offending_paths

    offending_paths="$(
        git diff --no-renames --name-only "${base}..${head}" |
            sed -n \
                -e '\|^vllm/.*\.py$|b' \
                -e '\|^tests/.*\.py$|b' \
                -e '\|^requirements/cuda\.txt$|b' \
                -e p
    )"
    if [[ -n "${offending_paths}" ]]; then
        echo "Exact-native mode only permits Python changes under vllm/ or tests/, or requirements/cuda.txt:" >&2
        printf '%s\n' "${offending_paths}" >&2
        return 1
    fi
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 0
fi

mkdir -p /opt/dynamo
: > "${PROVENANCE_FILE}"

case "${VLLM_INSTALL_MODE}" in
    auto)
        ;;
    full-source)
        validate_full_source_mode
        ;;
    *)
        echo "Unsupported VLLM_INSTALL_MODE: ${VLLM_INSTALL_MODE}" >&2
        echo "Supported modes: auto, full-source" >&2
        exit 1
        ;;
esac

if [[ -n "${VLLM_PRECOMPILED_WHEEL_COMMIT}" && -z "${VLLM_GIT_URL:-}" ]]; then
    echo "VLLM_PRECOMPILED_WHEEL_COMMIT requires VLLM_GIT_URL" >&2
    exit 1
fi
if [[ "${VLLM_INSTALL_MODE}" == "full-source" &&
      ( -z "${VLLM_GIT_URL:-}" || -z "${FLASHINFER_GIT_URL:-}" ) ]]; then
    echo "full-source mode requires vLLM and FlashInfer git URLs" >&2
    exit 1
fi
if [[ -n "${VLLM_GIT_URL:-}" || -n "${FLASHINFER_GIT_URL:-}" ]]; then
    apt-get update
    packages=(ca-certificates git)
    if [[ "${VLLM_INSTALL_MODE}" == "full-source" ]]; then
        packages+=(build-essential curl libibverbs-dev)
    fi
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

if [[ "${VLLM_INSTALL_MODE}" == "auto" &&
      -n "${VLLM_PRECOMPILED_WHEEL_COMMIT}" ]]; then
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

    if [[ "${VLLM_INSTALL_MODE}" == "full-source" ]]; then
        build_full_source_vllm
        install_mode=full-native-source
        native_wheel_commit=
        native_wheel_variant=
    elif [[ -n "${VLLM_PRECOMPILED_WHEEL_COMMIT}" ]]; then
        if ! git cat-file -e "${VLLM_PRECOMPILED_WHEEL_COMMIT}^{commit}"; then
            echo "Native wheel commit is absent from the custom vLLM clone: ${VLLM_PRECOMPILED_WHEEL_COMMIT}" >&2
            exit 1
        fi
        if ! git merge-base --is-ancestor \
            "${VLLM_PRECOMPILED_WHEEL_COMMIT}" "${vllm_source_sha}"; then
            echo "Native wheel commit is not an ancestor of custom vLLM HEAD" >&2
            exit 1
        fi

        validate_exact_native_changes \
            "${VLLM_PRECOMPILED_WHEEL_COMMIT}" "${vllm_source_sha}"

        uv pip uninstall --system vllm
        VLLM_TARGET_DEVICE=cuda \
        VLLM_USE_PRECOMPILED=1 \
        VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_PRECOMPILED_WHEEL_COMMIT}" \
        VLLM_PRECOMPILED_WHEEL_VARIANT="${VLLM_PRECOMPILED_WHEEL_VARIANT}" \
        VLLM_PRECOMPILED_WHEEL_LOCATION="${VLLM_PRECOMPILED_WHEEL_LOCATION}" \
        VLLM_DOCKER_BUILD_CONTEXT=1 \
            uv pip install --system '.[flashinfer,runai,otel]' \
                --torch-backend="${VLLM_TORCH_BACKEND}"
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
    echo "vllm_runtime_base_image=${VLLM_RUNTIME_BASE_IMAGE:-default}"
    echo "vllm_source_sha=${vllm_source_sha}"
    echo "vllm_native_wheel_commit=${native_wheel_commit}"
    echo "vllm_native_wheel_variant=${native_wheel_variant}"
} >> "${PROVENANCE_FILE}"

# This override intentionally runs after the custom vLLM metadata has selected
# the exact stock FlashInfer version. Resolve the custom checkout's requirements
# before replacing that same-version package with checkpoint-aware source and
# cubin builds using --no-deps.
if [[ -n "${FLASHINFER_GIT_URL:-}" ]]; then
    require_git_selector \
        FLASHINFER "${FLASHINFER_GIT_REF:-}" "${FLASHINFER_GIT_SHA:-}"
    clone_source \
        "${FLASHINFER_GIT_URL}" \
        "${FLASHINFER_GIT_REF:-}" \
        "${FLASHINFER_GIT_SHA:-}" \
        /tmp/flashinfer-src
    flashinfer_source_sha="${RESOLVED_SOURCE_SHA}"
    printf '%s\n' "${flashinfer_source_sha}" > "${FLASHINFER_SHA_FILE}"
    if [[ ! -s version.txt ]]; then
        echo "Custom FlashInfer source is missing version.txt" >&2
        exit 1
    fi
    flashinfer_source_version="$(tr -d '[:space:]' < version.txt)"
    if [[ -z "${flashinfer_source_version}" ]]; then
        echo "Custom FlashInfer source has an empty version.txt" >&2
        exit 1
    fi
    printf '%s\n' "${flashinfer_source_version}" > "${FLASHINFER_VERSION_FILE}"
    if [[ "${VLLM_INSTALL_MODE}" == "full-source" ]]; then
        uv pip install --system \
            --constraints "${VLLM_CONSTRAINTS_FILE}" \
            --torch-backend="${VLLM_TORCH_BACKEND}" \
            -r requirements.txt
    else
        uv pip install --system -r requirements.txt
    fi
    uv pip install --system --force-reinstall --no-build-isolation --no-deps .
    if [[ -d ./flashinfer-cubin ]]; then
        uv pip install --system --force-reinstall --no-build-isolation \
            --no-deps ./flashinfer-cubin
    fi
    uv pip uninstall --system flashinfer-jit-cache || true
    cd /
    rm -rf /tmp/flashinfer-src
else
    flashinfer_source_sha=upstream-runtime
    rm -f "${FLASHINFER_VERSION_FILE}"
    rm -f "${FLASHINFER_SHA_FILE}"
    echo "Using FlashInfer from the vLLM runtime/dependency solve (${FLASHINF_REF})."
fi

configure_pip_nccl_dso
verify_vllm_install \
    "${VLLM_GIT_URL:-}" \
    "${native_wheel_commit}" \
    "${native_wheel_variant:-${VLLM_TORCH_BACKEND}}" \
    "${VLLM_EXPECTED_TORCH_LOCAL_VERSION}" \
    "$([[ "${VLLM_INSTALL_MODE}" == "full-source" ]] && echo full-source)" \
    "$([[ "${VLLM_INSTALL_MODE}" == "full-source" ]] &&
        echo "${VLLM_TORCH_VERSION}+${VLLM_TORCH_BACKEND}")"
python3 <<'PY'
import importlib.metadata as metadata

distribution = metadata.distribution("flashinfer-python")
print(f"Installed FlashInfer version: {distribution.version}")
print(f"Installed FlashInfer location: {distribution.locate_file('').resolve()}")
PY
uv pip check --system
if [[ "${VLLM_INSTALL_MODE}" == "full-source" ]]; then
    python3 - <<'PY'
import importlib.metadata as metadata

try:
    metadata.distribution("torchaudio")
except metadata.PackageNotFoundError:
    print("Verified torchaudio is omitted")
else:
    raise SystemExit("torchaudio was reinstalled into the full-source runtime")
PY
fi
echo "flashinfer_source_sha=${flashinfer_source_sha}" >> "${PROVENANCE_FILE}"
if [[ -n "${flashinfer_source_version:-}" ]]; then
    echo "flashinfer_source_version=${flashinfer_source_version}" \
        >> "${PROVENANCE_FILE}"
fi

echo "Installed source/native provenance:"
cat "${PROVENANCE_FILE}"
