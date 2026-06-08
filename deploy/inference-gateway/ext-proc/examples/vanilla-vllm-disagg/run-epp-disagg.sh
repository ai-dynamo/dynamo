#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch the native EPP in external mode with role-based prefill/decode
# partitioning, so it routes to a decode worker and emits x-prefiller-host-port
# for the decode-side P/D routing sidecar to run vLLM's prefill->decode handshake.
# Arg 1 = tokenizer-sidecar port (default 8790).
cd /work
export DYN_DISCOVERY_BACKEND=mem POD_NAMESPACE="${POD_NAMESPACE:-default}" DYN_EPP_EXTERNAL=true
export DYN_MODEL_NAME="${DYN_MODEL_NAME:-Qwen/Qwen3-0.6B}" DYN_KV_CACHE_BLOCK_SIZE=16
export DYN_EPP_INFERENCE_POOL="${DYN_EPP_INFERENCE_POOL:-epp-vanilla-pool}"
export DYN_USE_KV_EVENTS=false DYN_SECURE_SERVING=true
export DYN_EPP_PREFIX_MODE=precise DYN_EPP_TOKENIZE_URL="http://127.0.0.1:${1:-8790}/tokenize"
export DYN_OVERLAP_SCORE_WEIGHT=1.0 DYN_EPP_KV_EVENTS=true DYN_EPP_KV_EVENT_PORT=5557
export RUST_LOG="${RUST_LOG:-info}"
# --- disagg role partitioning ---
export DYN_EPP_ROLE_LABEL="nvidia.com/dynamo-component-type"
export DYN_EPP_PREFILL_ROLE_VALUES=prefill
export DYN_EPP_DECODE_ROLE_VALUES=decode
export DYN_ENFORCE_DISAGG="${DYN_ENFORCE_DISAGG:-true}"
exec -a dyn-epp-main /work/dynamo/target/release/dynamo-ext-proc
