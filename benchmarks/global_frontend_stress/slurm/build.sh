#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds one binary on a compute node, in a target directory of its own.
# usage: build.sh frontend NAME SRC_DIR   -> $CAMPAIGN/builds/NAME/global-frontend
#        build.sh mock                    -> $CAMPAIGN/builds/mock/mock-pool
set -euo pipefail
source "$(dirname "$0")/env.sh"

kind=$1
case $kind in
    frontend)
        name=$2 src=$3
        out=$CAMPAIGN/builds/$name
        mkdir -p "$out"
        export CARGO_TARGET_DIR=$out/target CARGO_PROFILE_RELEASE_DEBUG=2 TMPDIR=$out/tmp
        # bindgen's clang needs gcc's headers (stdbool.h).
        export BINDGEN_EXTRA_CLANG_ARGS="-isystem $(gcc -print-file-name=include)"
        # Pingora's zlib-ng build needs cmake; nodes without one can use a PyPI wheel's here.
        if [[ -x $CAMPAIGN/env/cmake/bin/cmake ]]; then export PATH=$CAMPAIGN/env/cmake/bin:$PATH; fi
        # Nodes without internet access build Swagger UI from a downloaded zip.
        if [[ -n ${SWAGGER_UI_ZIP:-} ]]; then export SWAGGER_UI_DOWNLOAD_URL=file:$SWAGGER_UI_ZIP; fi
        mkdir -p "$TMPDIR"
        (cd "$src" && cargo build --release --locked -p dynamo-global-frontend --bin global-frontend) \
            > "$out/cargo.log" 2>&1 || { tail -n 40 "$out/cargo.log" >&2; exit 1; }
        cp "$CARGO_TARGET_DIR/release/global-frontend" "$out/global-frontend"
        git -C "$src" rev-parse HEAD > "$out/source-sha"
        sha256sum "$out/global-frontend" | tee "$out/sha256"
        ;;
    mock)
        out=$CAMPAIGN/builds/mock
        mkdir -p "$out"
        export CARGO_TARGET_DIR=$out/target
        (cd "$BENCH/mock_pool" && cargo build --release --locked --offline) \
            > "$out/cargo.log" 2>&1 || { tail -n 40 "$out/cargo.log" >&2; exit 1; }
        cp "$CARGO_TARGET_DIR/release/mock-pool" "$out/mock-pool"
        sha256sum "$out/mock-pool" | tee "$out/sha256"
        ;;
    *) echo "unknown build $kind" >&2; exit 2 ;;
esac
