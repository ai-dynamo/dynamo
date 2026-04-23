#!/bin/bash -e
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build the TRT-LLM wheel.

# This script builds the TRT-LLM base image for Dynamo with TensorRT-LLM.

# BUILD_DRIVER selects how the wheel is built.
#   auto   (default): pick 'buildx' if a local docker daemon is not reachable,
#                      otherwise 'make'. This keeps the legacy local-dev
#                      experience (`make -C docker wheel_build`) working while
#                      letting the CI runner, which only has buildx against a
#                      remote BuildKit, produce the same wheel.
#   buildx : always use `docker buildx build` against the current builder.
#   make   : always use the TensorRT-LLM docker/Makefile flow (requires a
#            local docker daemon with containerd/docker CLI).
BUILD_DRIVER="auto"

while getopts "c:o:a:n:u:d:" opt; do
  case ${opt} in
    c) TRTLLM_COMMIT=$OPTARG ;;
    o) OUTPUT_DIR=$OPTARG ;;
    a) ARCH=$OPTARG ;;
    n) NIXL_COMMIT=$OPTARG ;;
    u) TRTLLM_GIT_URL=$OPTARG ;;
    d) BUILD_DRIVER=$OPTARG ;;
    *) echo "Usage: $(basename $0) [-c commit] [-o output_dir] [-a arch] [-n nixl_commit] [-u git_url] [-d build_driver]"
       echo "  -c: TensorRT-LLM commit to build"
       echo "  -o: Output directory for wheel files"
       echo "  -a: Architecture (amd64 or arm64)"
       echo "  -n: NIXL commit"
       echo "  -u: TensorRT-LLM git URL"
       echo "  -d: Build driver (auto|buildx|make, default auto)"
       exit 1 ;;
  esac
done

# Set default output directory if not specified
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="/tmp/trtllm_wheel"
fi

# Set default TensorRT-LLM git URL if not specified.
# Default points at the galletas1712 fork because the Kimi K2.5 + Eagle3
# shadow-failover proof pins the runtime image to a fork-only commit
# (see container/context.yaml::trtllm.github_trtllm_commit).
if [ -z "$TRTLLM_GIT_URL" ]; then
    TRTLLM_GIT_URL="https://github.com/galletas1712/TensorRT-LLM.git"
fi

# Store directory where script is being launched from
MAIN_DIR=$(dirname "$(readlink -f "$0")")

(cd /tmp && \
# Clone the TensorRT-LLM repository.
# Use a bare fetch-and-checkout flow so that ${TRTLLM_COMMIT} does not need
# to be reachable from ${TRTLLM_GIT_URL}'s default branch. The Kimi K2.5 +
# Eagle3 proof pins a SHA that lives only on a feature branch of the fork
# (see container/context.yaml::trtllm.github_trtllm_commit), so the previous
# 'checkout main; pull; checkout <sha>' flow would fail on any SHA that is
# not yet merged to main on the selected remote.
if [ ! -d "TensorRT-LLM" ]; then
  git init TensorRT-LLM
  (cd TensorRT-LLM && git remote add origin "${TRTLLM_GIT_URL}")
fi

cd TensorRT-LLM

# Make sure the remote URL matches the requested URL in case the directory was
# reused from a previous build that targeted a different fork.
git remote set-url origin "${TRTLLM_GIT_URL}"

# Fetch the exact commit directly. This works for any reachable SHA on the
# remote (branches, tags, or PR heads) without having to enumerate refs or
# pull the entire history of main first.
git fetch --no-tags --prune --depth=1 origin "${TRTLLM_COMMIT}"
git checkout --detach FETCH_HEAD

# Update the submodules.
git submodule update --init --recursive
git lfs pull

VERSION_FILE="tensorrt_llm/version.py"

# Check if file exists
if [ ! -f "$VERSION_FILE" ]; then
    echo "Error: $VERSION_FILE not found"
    exit 1
fi

# Create a backup of the original version file
cp $VERSION_FILE ${VERSION_FILE}.bak

# Check if version line exists
if ! grep -q "^__version__" "$VERSION_FILE"; then
    echo "Error: __version__ not found in $VERSION_FILE"
    exit 1
fi

# Append suffix to version
COMMIT_VERSION=$(git rev-parse --short HEAD)
sed -i "s/__version__ = \"\(.*\)\"/__version__ = \"\1+dev${COMMIT_VERSION}\"/" "$VERSION_FILE"

echo "Updated version:"
grep "__version__" "$VERSION_FILE"

# Maximize parallelism for the wheel build. TRTLLM's wheel build_wheel.py
# reads JOBS from the environment, and cmake/make rules accept MAKEFLAGS.
#
# nproc inside the runner pod reports the host CPU count, which is NOT the
# cgroup CPU limit the remote BuildKit pod actually has. Pushing `-j` up to
# the host count over-subscribes the pod's memory cgroup and OOM-kills the
# BuildKit worker mid-compile (observed in run 24855398208 with -j 12 and a
# 29Gi limit). Clamp JOBS to something that comfortably fits the BuildKit
# pod configured for the caller unless the caller has overridden it.
#
# With ~2-3 GiB peak per CUDA compile, 8 concurrent compiles use ~16-24 GiB
# which comfortably fits inside the 48Gi BuildKit pod configured by
# .github/workflows/build-on-demand.yml trtllm-pipeline (fallback node pool
# capped the pod at 48Gi -- see comment there). Callers that want a
# different parallelism should set JOBS in the environment before invoking
# this script.
if [ -z "${JOBS:-}" ]; then
    HOST_CPUS="$(nproc 2>/dev/null || echo 16)"
    if [ "${HOST_CPUS}" -gt 8 ]; then
        export JOBS=8
    else
        export JOBS="${HOST_CPUS}"
    fi
fi
if [ -z "${MAKEFLAGS:-}" ]; then
    export MAKEFLAGS="-j${JOBS}"
fi

mkdir -p "$OUTPUT_DIR"

# Auto-detect the build driver. The CI builder only has `docker buildx`
# pointed at a remote BuildKit; it has no local docker daemon. In that case
# the legacy `make -C docker wheel_build` flow (which uses `docker pull`,
# `docker create`, and `docker cp` through a local socket) is unusable.
if [ "${BUILD_DRIVER}" = "auto" ]; then
    if docker info >/dev/null 2>&1; then
        BUILD_DRIVER="make"
    else
        BUILD_DRIVER="buildx"
    fi
fi
echo "Using BUILD_DRIVER=${BUILD_DRIVER}"

if [ "${BUILD_DRIVER}" = "make" ]; then
    if [ "$ARCH" = "amd64" ]; then
        # Need to build in the Triton Devel Image for NIXL support.
        make -C docker tritondevel_build
        make -C docker wheel_build DEVEL_IMAGE=tritondevel BUILD_WHEEL_OPTS="--extra-cmake-vars NIXL_ROOT=/opt/nvidia/nvda_nixl --job_count ${JOBS}"
    else
        # NIXL backend is not supported on arm64 for TensorRT-LLM.
        # See here: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docker/common/install_nixl.sh
        make -C docker wheel_build BUILD_WHEEL_OPTS="--job_count ${JOBS}"
    fi

    # Copy the wheel to the host
    docker create --name trtllm_wheel_container docker.io/tensorrt_llm/wheel:latest
    docker cp trtllm_wheel_container:/src/tensorrt_llm/build "$OUTPUT_DIR/"
    cp "$OUTPUT_DIR/build"/*.whl "$OUTPUT_DIR/"
    docker rm trtllm_wheel_container || true
elif [ "${BUILD_DRIVER}" = "buildx" ]; then
    # Add a tiny scratch-based export stage that contains only the wheel files
    # so `docker buildx build --output type=local,dest=...` can extract them
    # without shipping the entire devel image back to the runner. We modify
    # the fork's docker/Dockerfile.multi in-place inside the throwaway /tmp
    # clone (not the source repo).
    DOCKERFILE="docker/Dockerfile.multi"
    if ! grep -q '^FROM scratch AS wheel_export$' "${DOCKERFILE}"; then
        cat >> "${DOCKERFILE}" <<'EOF'

# Appended by dynamo container/build_trtllm_wheel.sh to make the built wheel
# extractable via `docker buildx build --target wheel_export --output type=local,dest=<dir>`.
FROM scratch AS wheel_export
COPY --from=wheel /src/tensorrt_llm/build/ /
EOF
    fi

    # Assemble BUILD_WHEEL_ARGS the way docker/Makefile's %_build rule does,
    # but restrict the CUDA architecture set by default. The fork's
    # cutlass_kernels instantiations compile a separate .cu.o per (sm, tile)
    # combination; the stock build target "all" (SM70/75/80/86/89/90/100/120)
    # expands to ~thousands of translation units and dominates wheel build
    # wall-clock. The Kimi K2.5 + Eagle3 shadow-failover proof that this
    # fork wheel exists for currently targets nscale B200 (SM100), so
    # compiling only SM100 kernels cuts the CUDA compile count to the
    # subset we actually ship without changing which runtime symbols end
    # up in libtensorrt_llm.so (the VMM rollback fix in virtualMemory.cpp
    # is independent of CUDA arch).
    #
    # Callers that want broader arch coverage can override TRTLLM_CUDA_ARCHS
    # in the environment, e.g.
    #   TRTLLM_CUDA_ARCHS='90-real;100-real'   (H200 + B200)
    #   TRTLLM_CUDA_ARCHS='native'             (use detected host arch)
    TRTLLM_CUDA_ARCHS="${TRTLLM_CUDA_ARCHS:-100-real}"

    WHEEL_ARGS="--clean --benchmarks"
    WHEEL_ARGS+=" --cuda_architectures '${TRTLLM_CUDA_ARCHS}'"
    if [ "$ARCH" = "amd64" ]; then
        # Triton-devel-based build picks up NIXL at /opt/nvidia/nvda_nixl.
        WHEEL_ARGS+=" --extra-cmake-vars NIXL_ROOT=/opt/nvidia/nvda_nixl"
        DEVEL_STAGE="tritondevel"
        TARGETPLATFORM="linux/amd64"
    else
        DEVEL_STAGE="devel"
        TARGETPLATFORM="linux/arm64"
    fi
    WHEEL_ARGS+=" --job_count ${JOBS}"

    # Mirror Makefile defaults for BASE_IMAGE/BASE_TAG etc. by parsing the
    # `ARG <NAME>=<default>` lines from Dockerfile.multi. Falls back to the
    # ARG defaults already baked into the Dockerfile if anything is missing.
    arg_default() { grep -m1 "^ARG $1=" "${DOCKERFILE}" | sed -E 's/^ARG [^=]+=//; s/^"(.*)"$/\1/'; }
    BASE_IMAGE_VAL="$(arg_default BASE_IMAGE)"
    BASE_TAG_VAL="$(arg_default BASE_TAG)"
    TRITON_IMAGE_VAL="$(arg_default TRITON_IMAGE)"
    TRITON_BASE_TAG_VAL="$(arg_default TRITON_BASE_TAG)"

    # TRT_LLM_VER is derived from tensorrt_llm/version.py::__version__ just
    # like docker/Makefile does (TRT_LLM_VERSION). The devel stage's
    # generate_container_oss_attribution.sh treats it as a required TAG arg.
    TRT_LLM_VER_VAL="$(grep '^__version__' tensorrt_llm/version.py | grep -o '=.*' | tr -d '= "')"
    if [ -z "${TRT_LLM_VER_VAL}" ]; then
        echo "Error: could not parse __version__ from tensorrt_llm/version.py" >&2
        exit 1
    fi

    echo "buildx wheel build: arch=${ARCH} devel_stage=${DEVEL_STAGE} platform=${TARGETPLATFORM}"
    echo "  BASE_IMAGE=${BASE_IMAGE_VAL}:${BASE_TAG_VAL}"
    echo "  TRITON_IMAGE=${TRITON_IMAGE_VAL}:${TRITON_BASE_TAG_VAL}"
    echo "  TRT_LLM_VER=${TRT_LLM_VER_VAL}"
    echo "  BUILD_WHEEL_ARGS=${WHEEL_ARGS}"

    WHEEL_STAGE_DIR="${OUTPUT_DIR}/_wheel_stage"
    rm -rf "${WHEEL_STAGE_DIR}"
    mkdir -p "${WHEEL_STAGE_DIR}"

    # Optional BuildKit registry cache so subsequent runs skip the CUDA
    # compile entirely. Caller sets TRTLLM_WHEEL_CACHE_REF=<registry>/<repo>
    # and we derive a per-(arch,commit,cuda_archs,jobs) tag. mode=max stores
    # every intermediate layer (including the ccache / pip caches baked into
    # the build), which is what makes the second run cheap.
    CACHE_ARGS=""
    if [ -n "${TRTLLM_WHEEL_CACHE_REF:-}" ]; then
        CACHE_KEY="$(echo "${ARCH}-${TRTLLM_COMMIT}-${TRTLLM_CUDA_ARCHS}-j${JOBS}" | tr -c 'A-Za-z0-9_.-' '_')"
        echo "Using BuildKit registry cache: ${TRTLLM_WHEEL_CACHE_REF}:${CACHE_KEY}"
        CACHE_ARGS="--cache-from type=registry,ref=${TRTLLM_WHEEL_CACHE_REF}:${CACHE_KEY} "
        CACHE_ARGS+="--cache-to type=registry,ref=${TRTLLM_WHEEL_CACHE_REF}:${CACHE_KEY},mode=max"
    fi

    # shellcheck disable=SC2086
    docker buildx build \
        --progress=plain \
        --platform "${TARGETPLATFORM}" \
        --file "${DOCKERFILE}" \
        --target wheel_export \
        --build-arg "BASE_IMAGE=${BASE_IMAGE_VAL}" \
        --build-arg "BASE_TAG=${BASE_TAG_VAL}" \
        --build-arg "TRITON_IMAGE=${TRITON_IMAGE_VAL}" \
        --build-arg "TRITON_BASE_TAG=${TRITON_BASE_TAG_VAL}" \
        --build-arg "DEVEL_IMAGE=${DEVEL_STAGE}" \
        --build-arg "TRT_LLM_VER=${TRT_LLM_VER_VAL}" \
        --build-arg "BUILD_WHEEL_ARGS=${WHEEL_ARGS}" \
        ${CACHE_ARGS} \
        --output "type=local,dest=${WHEEL_STAGE_DIR}" \
        .

    # Flatten extracted files into OUTPUT_DIR/ so consumers can glob for
    # *.whl at the top level just like the legacy make flow.
    mkdir -p "${OUTPUT_DIR}/build"
    cp -a "${WHEEL_STAGE_DIR}"/. "${OUTPUT_DIR}/build/"
    find "${OUTPUT_DIR}/build" -maxdepth 3 -name '*.whl' -exec cp -v {} "${OUTPUT_DIR}/" \;
    rm -rf "${WHEEL_STAGE_DIR}"
else
    echo "Error: unknown BUILD_DRIVER='${BUILD_DRIVER}' (expected auto, buildx, or make)" >&2
    exit 2
fi

# Restore the original version file
mv ${VERSION_FILE}.bak $VERSION_FILE

)

# Store the commit hash in the output directory to ensure the wheel is built from the correct commit.
rm -rf $OUTPUT_DIR/commit.txt
echo ${ARCH}_${TRTLLM_COMMIT} > $OUTPUT_DIR/commit.txt

echo "TRT-LLM wheel built successfully."
ls -al $OUTPUT_DIR
