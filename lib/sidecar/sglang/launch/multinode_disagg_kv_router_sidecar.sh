#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One node of a multinode prefill or decode group with NIXL transfer and KV routing.
# Defaults: two nodes per role, one GPU per node, Qwen/Qwen3-32B.

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo "Usage: NODE_RANK=<rank> DIST_INIT_ADDR=<role-leader:port> bash $0 <prefill|decode> [SGLang options...]"
    echo "Run once per node, with separate rendezvous addresses for prefill and decode."
    echo "All nodes and the frontend must share Dynamo namespace, discovery and event services."
    echo
    echo "Example: run each command on its respective node (four nodes total):"
    echo "  NODE_RANK=0 DIST_INIT_ADDR=prefill-leader:29500 SGLANG_BOOTSTRAP_HOST=prefill-leader bash $0 prefill"
    echo "  NODE_RANK=1 DIST_INIT_ADDR=prefill-leader:29500 bash $0 prefill"
    echo "  NODE_RANK=0 DIST_INIT_ADDR=decode-leader:29500 bash $0 decode"
    echo "  NODE_RANK=1 DIST_INIT_ADDR=decode-leader:29500 bash $0 decode"
    echo
    echo "Start the frontend separately: python3 -m dynamo.frontend --router-mode kv"
    echo "Use reachable leader addresses; NIXL requires connectivity between the two groups."
    echo "For model, topology and port overrides: bash $SCRIPT_DIR/multinode_kv_router_sidecar.sh --help"
    exit 0
fi

case "${1:-}" in
    prefill|decode) export ROLE="$1"; shift ;;
    *) echo "First argument must be prefill or decode; use --help for examples" >&2; exit 1 ;;
esac

# Reuse per-node engine/sidecar startup, validation, GPU budgeting and cleanup.
# ROLE enables NIXL; node 0 serves requests and followers relay local KV events.
exec bash "$SCRIPT_DIR/multinode_kv_router_sidecar.sh" "$@"
