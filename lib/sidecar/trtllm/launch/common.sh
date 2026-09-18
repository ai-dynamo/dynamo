#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared by agg.sh and disagg.sh. Both need the same OpenEngine bindings and the
# same context-length derivation; keeping one copy is what stops the two
# topologies drifting onto different contract revisions.
#
# Source it, then call:
#   trtllm_ensure_openengine_bindings "$TRTLLM_PYTHON"
#   trtllm_resolve_context_length "${EXTRA_ARGS[@]}"
#
# `trtllm_resolve_context_length` sets TRTLLM_MAX_SEQ_LEN_ARGS,
# TRTLLM_CONTEXT_LENGTH_ARGS, and TRTLLM_CONTEXT_LENGTH.

# `--grpc-protocol openengine` needs the OpenEngine bindings, which resolve only
# from a custom index. Both packages are pinned to BSR module commit
# 768a93c7b44e, the same revision the vendored protos in `proto/` were generated
# from (see `proto/README.md`), so the engine and the sidecar speak the same
# contract revision.
#
# The protobuf package is pinned by *gencode* version as well: buf publishes one
# build per protoc release, and a gencode newer than the runtime in the
# TensorRT-LLM image fails at import with "Detected incompatible Protobuf
# Gencode/Runtime versions". 33.5 matches the protobuf 6.33.x runtime those
# images ship. Raise it only together with the image's protobuf.
TRTLLM_OPENENGINE_PROTOBUF_VERSION="33.5.0.1.20260730172104+768a93c7b44e"
TRTLLM_OPENENGINE_GRPC_VERSION="1.78.1.1.20260730172104+768a93c7b44e"

# Installs the bindings unless they are already importable. Set
# TRTLLM_SKIP_BINDINGS_INSTALL=1 on an air-gapped host or one whose
# site-packages is read-only, and install them yourself.
trtllm_ensure_openengine_bindings() {
    local python="$1"
    if "$python" -c "import openengine.v1.openengine_pb2" >/dev/null 2>&1; then
        return 0
    fi
    if [[ -n "${TRTLLM_SKIP_BINDINGS_INSTALL:-}" ]]; then
        echo "OpenEngine bindings are missing and TRTLLM_SKIP_BINDINGS_INSTALL is set." >&2
        echo "Install them into $python before starting the engine." >&2
        return 1
    fi
    echo "Installing the pinned OpenEngine bindings into $python..."
    "$python" -m pip install --no-cache-dir \
        --extra-index-url https://buf.build/gen/python \
        "openengine-openengine-grpc-python==${TRTLLM_OPENENGINE_GRPC_VERSION}" \
        "openengine-openengine-protocolbuffers-python==${TRTLLM_OPENENGINE_PROTOBUF_VERSION}"
}

# Keeps the engine and the sidecar on one number. Started without
# `--max_seq_len`, TensorRT-LLM leaves `max_context_length` unset and the sidecar
# has no window to register, so pass the same value to both. When the caller
# supplies `--max_seq_len`, theirs wins and the sidecar adopts the engine's
# `Control.GetModelInfo` report rather than overriding it with a default it was
# never told about.
trtllm_resolve_context_length() {
    TRTLLM_MAX_SEQ_LEN_ARGS=()
    TRTLLM_CONTEXT_LENGTH_ARGS=()
    local arg supplied=0
    for arg in "$@"; do
        case "$arg" in
            --max_seq_len|--max_seq_len=*) supplied=1 ;;
        esac
    done
    if [[ "$supplied" -eq 0 ]]; then
        TRTLLM_CONTEXT_LENGTH="${TRTLLM_CONTEXT_LENGTH:-4096}"
        TRTLLM_MAX_SEQ_LEN_ARGS=(--max_seq_len "$TRTLLM_CONTEXT_LENGTH")
    fi
    if [[ -n "${TRTLLM_CONTEXT_LENGTH:-}" ]]; then
        TRTLLM_CONTEXT_LENGTH_ARGS=(--context-length "$TRTLLM_CONTEXT_LENGTH")
    fi
}
