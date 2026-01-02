#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SOURCE_DIR}/.." && pwd)"

ORIG_ARGS=("$@")

RUN_PREFIX=()
DRY_RUN="false"

FRAMEWORK="vllm"
PLATFORM="linux/amd64"
LOCAL_DEV_TAG=""
RUNTIME_IMAGE_INPUT=""
CUSTOM_UID=""
CUSTOM_GID=""
OUTPUT="dev"
TARGET_GIVEN="false"
OUTPUT_GIVEN="false"

BUILD_SH_ARGS=()

_gen_temp_dockerfile() {
    # Create a temporary Dockerfile that is a literal concatenation of:
    #   1) the framework Dockerfile chosen by build.sh (DEV_BUILD_DOCKERFILE)
    #   2) container/dev/Dockerfile.dev
    #
    # Dockerfile.dev intentionally depends on stages defined in (1): `runtime`, `dynamo_base`, `wheel_builder`.
    local fw_df out
    fw_df="${DEV_BUILD_DOCKERFILE:-}"
    if [[ -z "${fw_df}" ]]; then
        echo "ERROR: Missing DEV_BUILD_DOCKERFILE from build.sh." >&2
        exit 2
    fi
    if [[ ! -f "${fw_df}" ]]; then
        echo "ERROR: DEV_BUILD_DOCKERFILE does not exist: ${fw_df}" >&2
        exit 2
    fi
    out="$(mktemp -t dynamo-dev-combined.XXXXXX.Dockerfile)"

    if ! python3 - "$fw_df" "${SOURCE_DIR}/dev/Dockerfile.dev" "$out" <<'PY'
import sys
from pathlib import Path

fw_path = Path(sys.argv[1])
dev_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])

fw = fw_path.read_text(encoding="utf-8").splitlines()
dev = dev_path.read_text(encoding="utf-8").splitlines()

# Keep exactly one syntax directive (from the framework file if present; otherwise from dev file if present).
out_lines = []
if fw and fw[0].startswith("# syntax="):
    out_lines.append(fw[0])
    fw = fw[1:]
if dev and dev[0].startswith("# syntax="):
    dev = dev[1:]

out_lines.extend(fw)
out_lines.append("")
out_lines.append("# ----------------------------------------------------------------------")
out_lines.append("# CONCAT: container/dev/Dockerfile.dev")
out_lines.append("# ----------------------------------------------------------------------")
out_lines.append("")
out_lines.extend(dev)
out_lines.append("")

out_path.write_text("\n".join(out_lines), encoding="utf-8")
PY
    then
        echo "ERROR: Failed to generate temp Dockerfile: ${out}" >&2
        rm -f "${out}"
        exit 1
    fi

    if [[ ! -s "${out}" ]]; then
        echo "ERROR: Temp Dockerfile was generated but is empty: ${out}" >&2
        rm -f "${out}"
        exit 1
    fi

    echo "$out"
}

print_cmd() {
    # Print a shell-escaped command line (so it's copy/pasteable).
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
}

run_cmd() {
    # Always print the actual command; run it unless we're in dry-run mode.
    print_cmd "$@"
    if [[ "${DRY_RUN}" == "true" ]]; then
        return 0
    fi
    "$@"
}

usage() {
    cat <<'EOF'
usage: container/dev/dev_build.sh [options]

Build a development image on top of a runtime image.

Behavior:
  1) Generate a temp Dockerfile by concatenating the framework Dockerfile + container/dev/Dockerfile.dev
  2) Build either target from that temp Dockerfile:
     - local-dev (non-root, UID/GID remapped)
     - dev      (root-based)

Options:
  --framework <vllm|trtllm|sglang|none>   (default: vllm)
  --platform <linux/amd64|linux/arm64>   (default: linux/amd64)
  --runtime-image <image:tag>            REQUIRED. Existing runtime image to build on top of
  --output <local-dev|dev>               Output image type (default: local-dev)
  --tag <image:tag>                      Tag for the output image (default: derived from runtime tag)
  --uid <uid>                            UID for dynamo user inside local-dev (default: current user)
  --gid <gid>                            GID for dynamo user inside local-dev (default: current user)
  --no-tag-latest                        Do not add latest-* tags for the output image
  --dry-run                              Print commands without executing them
  -h, --help                             Show this help
EOF
}

while :; do
    case "${1:-}" in
    -h | --help)
        usage
        exit 0
        ;;
    --framework)
        FRAMEWORK="${2:-}"; shift
        ;;
    --platform)
        PLATFORM="${2:-}"; shift
        ;;
    --runtime-image)
        RUNTIME_IMAGE_INPUT="${2:-}"; shift
        ;;
    --target)
        # build.sh compatibility: map --target dev/local-dev to --output
        TARGET_GIVEN="true"
        case "${2:-}" in
          dev) OUTPUT="dev" ;;
          local-dev) OUTPUT="local-dev" ;;
          "" ) ;;
          *)
            # If build.sh forwards something else, keep it for runtime build.sh invocation
            BUILD_SH_ARGS+=("$1" "${2:-}")
            ;;
        esac
        shift
        ;;
    --output)
        OUTPUT="${2:-}"; shift
        OUTPUT_GIVEN="true"
        ;;
    --tag)
        LOCAL_DEV_TAG="${2:-}"; shift
        ;;
    --uid)
        CUSTOM_UID="${2:-}"; shift
        ;;
    --gid)
        CUSTOM_GID="${2:-}"; shift
        ;;
    --dry-run)
        DRY_RUN="true"
        ;;
    --no-tag-latest)
        NO_TAG_LATEST="true"
        ;;
    "")
        break
        ;;
    *)
        # Forward unknown options to build.sh (runtime build) for compatibility with container/build.sh flags
        BUILD_SH_ARGS+=("${1}")
        ;;
    esac
    shift
done

if [[ "${OUTPUT}" != "local-dev" && "${OUTPUT}" != "dev" ]]; then
    echo "ERROR: --output must be one of: local-dev, dev (got: ${OUTPUT})" >&2
    usage
    exit 2
fi

NO_TAG_LATEST="${NO_TAG_LATEST:-false}"

# Default behavior matches build.sh: no --target implies dev.

# Determine base tag from either explicit --tag (build.sh style) or build.sh-derived default tag.
BASE_TAG=""
if [[ -n "${LOCAL_DEV_TAG}" ]]; then
    BASE_TAG="${LOCAL_DEV_TAG}"
elif [[ -n "${DEV_BUILD_DEFAULT_TAG:-}" ]]; then
    BASE_TAG="${DEV_BUILD_DEFAULT_TAG}"
fi

if [[ -z "${BASE_TAG}" ]]; then
    echo "ERROR: must be invoked via build.sh (so a default tag can be derived) or provide --tag." >&2
    usage
    exit 2
fi

# Strip any dev/local-dev/runtime suffix to get the base, then derive dev/local-dev tag.
base="${BASE_TAG%-runtime}"
base="${base%-local-dev}"
base="${base%-dev}"

if [[ -z "${LOCAL_DEV_TAG}" ]]; then
    if [[ "${OUTPUT}" == "local-dev" ]]; then
        LOCAL_DEV_TAG="${base}-local-dev"
    else
        LOCAL_DEV_TAG="${base}-dev"
    fi
fi

if [[ -z "${CUSTOM_UID}" ]]; then
    CUSTOM_UID="$(id -u)"
fi
if [[ -z "${CUSTOM_GID}" ]]; then
    CUSTOM_GID="$(id -g)"
fi

ARCH="amd64"
ARCH_ALT="x86_64"
if [[ "${PLATFORM}" == *"linux/arm64"* ]]; then
    ARCH="arm64"
    ARCH_ALT="aarch64"
fi

# Generate a temp concatenated Dockerfile.
TEMP_DOCKERFILE=""
# Always use the temp concatenated Dockerfile; this script assumes build.sh is the caller.
DYNAMO_USE_TEMP_DOCKERFILE="${DYNAMO_USE_TEMP_DOCKERFILE:-1}"
if [[ -n "${DYNAMO_USE_TEMP_DOCKERFILE:-}" ]]; then
    if [[ -z "${DEV_BUILD_DOCKER_BUILD_ARGS:-}" ]]; then
        echo "ERROR: Missing DEV_BUILD_DOCKER_BUILD_ARGS from build.sh." >&2
        exit 2
    fi
    TEMP_DOCKERFILE="$(_gen_temp_dockerfile)"
fi

_parse_build_arg() {
    # Extract the last occurrence of: --build-arg NAME=value
    # from the build.sh-provided DEV_BUILD_DOCKER_BUILD_ARGS string.
    local name="$1"
    echo "${DEV_BUILD_DOCKER_BUILD_ARGS:-}" | sed -n "s/.*--build-arg ${name}=\\([^ ]*\\).*/\\1/p"
}

_validate_required_build_args() {
    # build.sh is the only supported caller; validate the final build-args we will pass through.
    if [[ -z "${DEV_BUILD_DOCKER_BUILD_ARGS:-}" ]]; then
        echo "ERROR: Missing DEV_BUILD_DOCKER_BUILD_ARGS from build.sh." >&2
        exit 2
    fi

    local base_image base_image_tag python_version
    base_image="$(_parse_build_arg BASE_IMAGE)"
    base_image_tag="$(_parse_build_arg BASE_IMAGE_TAG)"
    python_version="$(_parse_build_arg PYTHON_VERSION)"

    if [[ -z "${base_image}" ]]; then
        echo "ERROR: Missing required build arg: BASE_IMAGE" >&2
        exit 2
    fi
    if [[ -z "${base_image_tag}" ]]; then
        echo "ERROR: Missing required build arg: BASE_IMAGE_TAG" >&2
        exit 2
    fi
    if [[ -z "${python_version}" ]]; then
        echo "ERROR: Missing required build arg: PYTHON_VERSION" >&2
        exit 2
    fi

    # vLLM arm64 requires an explicit runtime tag, and the default vLLM x86 tag is not usable on arm64.
    if [[ "${FRAMEWORK}" == "vllm" && "${PLATFORM}" == *"linux/arm64"* ]]; then
        local runtime_image_tag
        runtime_image_tag="$(_parse_build_arg RUNTIME_IMAGE_TAG)"
        if [[ -z "${runtime_image_tag}" ]]; then
            echo "ERROR: vLLM arm64 requires --build-arg RUNTIME_IMAGE_TAG=... (e.g. 12.9.0-runtime-ubuntu24.04)." >&2
            exit 2
        fi
        # Known-bad default for vLLM arm64; require explicit override.
        if [[ "${base_image_tag}" == "25.04-cuda12.9-devel-ubuntu24.04" ]]; then
            echo "ERROR: vLLM arm64 requires an arm64-compatible BASE_IMAGE_TAG (current is default: ${base_image_tag})." >&2
            exit 2
        fi
    fi

    # SGLang requires CUDA_VERSION for wheel_builder.
    if [[ "${FRAMEWORK}" == "sglang" ]]; then
        local cuda_version
        cuda_version="$(_parse_build_arg CUDA_VERSION)"
        if [[ -z "${cuda_version}" ]]; then
            echo "ERROR: sglang requires --build-arg CUDA_VERSION=..." >&2
            exit 2
        fi
    fi
}

echo ""
echo "Building ${OUTPUT} image:"
echo "  runtime:    (from concatenated framework Dockerfile)"
echo "  output:     ${LOCAL_DEV_TAG}"
if [[ "${OUTPUT}" == "local-dev" ]]; then
    echo "  uid/gid:    ${CUSTOM_UID}/${CUSTOM_GID}"
fi
echo ""

# Build args common to both dev and local-dev
df_dev="${TEMP_DOCKERFILE:-${SOURCE_DIR}/dev/Dockerfile.dev}"
passthru_args=()
if [[ -n "${TEMP_DOCKERFILE}" ]]; then
    # build.sh constructs the full set of `--build-arg ...` (and related) flags; pass through verbatim.
    # Shell-splitting is OK here because build.sh constructs BUILD_ARGS as a simple, space-delimited flag list.
    read -r -a passthru_args <<< "${DEV_BUILD_DOCKER_BUILD_ARGS}"
fi
build_context_args=()
if [[ -n "${DEV_BUILD_DOCKER_BUILD_CONTEXT_ARGS:-}" ]]; then
    read -r -a build_context_args <<< "${DEV_BUILD_DOCKER_BUILD_CONTEXT_ARGS}"
fi
_validate_required_build_args
build_args=(
    --file "${df_dev}"
    --target "${OUTPUT}"
    --build-arg "FRAMEWORK=${FRAMEWORK}"
)

# Add local-dev specific args
if [[ "${OUTPUT}" == "local-dev" ]]; then
    build_args+=(
        --build-arg "USER_UID=${CUSTOM_UID}"
        --build-arg "USER_GID=${CUSTOM_GID}"
    )
fi

# wheel_builder stage is provided by the concatenated framework Dockerfile, so no external wheel_builder image arg is needed.

run_cmd docker build --progress=plain "${build_args[@]}" "${passthru_args[@]}" "${build_context_args[@]}" --tag "${LOCAL_DEV_TAG}" "${REPO_ROOT}"

echo ""
echo "Successfully built: ${LOCAL_DEV_TAG}"
echo "Run with:"
echo "  ${SOURCE_DIR}/run.sh --image ${LOCAL_DEV_TAG} --mount-workspace -it ..."

# Apply latest tags for dev/local-dev output (docker build itself only tags ${LOCAL_DEV_TAG}).
if [[ "${NO_TAG_LATEST}" != "true" ]]; then
    latest_out=""
    if [[ "${OUTPUT}" == "dev" ]]; then
        latest_out="dynamo:latest-${FRAMEWORK}"
    else
        latest_out="dynamo:latest-${FRAMEWORK}-local-dev"
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "+ docker tag ${LOCAL_DEV_TAG} ${latest_out}"
    else
        docker tag "${LOCAL_DEV_TAG}" "${latest_out}"
    fi
fi


