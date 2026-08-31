#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared environment for the Dynamo frontend benchmark harness.
# Source from the other scripts: `source "$(dirname "$0")/env.sh"`
#
# REQUIRED: set DYN_REPO to your Dynamo checkout (or worktree) that contains the
# built `.venv` (uv venv + `maturin develop --uv --release`). Everything else
# derives from it and is overridable from the calling shell.

# --- Repo / venv -----------------------------------------------------------
export DYN_REPO="${DYN_REPO:?set DYN_REPO to your dynamo checkout, e.g. export DYN_REPO=/path/to/dynamo}"
export WORKTREE="$DYN_REPO"                       # alias used by some scripts
export VENV="${VENV:-$DYN_REPO/.venv}"
export DYN_PY="${DYN_PY:-$VENV/bin/python}"       # runs dynamo.frontend / dynamo.mocker
# aiperf client: a venv with aiperf installed (can be the same venv or another).
export AIPERF="${AIPERF:-$VENV/bin/aiperf}"
# FlameGraph scripts (stackcollapse-perf.pl, flamegraph.pl). Clone from
# https://github.com/brendangregg/FlameGraph if absent.
export FLAMEGRAPH_DIR="${FLAMEGRAPH_DIR:-$DYN_REPO/FlameGraph}"

# --- Model / topology ------------------------------------------------------
export MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
export ENDPOINT="${ENDPOINT:-dyn://dynamo.backend.generate}"   # mocker default
export HTTP_PORT="${HTTP_PORT:-8000}"
# Block size: frontend (--kv-cache-block-size) and mocker (--block-size) MUST
# agree. 64 is realistic; larger reduces the MOCK's per-request KV-block
# bookkeeping (see SKILL.md "Block size"). NOTE: very large sizes (>=2048) break
# the current mocker (requests hang, never emit a token).
export BLOCK_SIZE="${BLOCK_SIZE:-64}"
export NUM_WORKERS="${NUM_WORKERS:-4}"

# --- Discovery (etcd + nats) ----------------------------------------------
export ETCD_ENDPOINT="${ETCD_ENDPOINT:-localhost:2379}"
# Worker instances register one lease-backed key per worker here; dead workers
# self-expire. Used to verify exactly NUM_WORKERS are live and no stale ones linger.
export INSTANCE_PREFIX="${INSTANCE_PREFIX:-v1/instances/dynamo/backend/generate/}"
# count live registered workers (grep -c exits 1 on zero, which would trip set -e)
count_workers() { ETCDCTL_API=3 etcdctl --endpoints="$ETCD_ENDPOINT" get --prefix "$INSTANCE_PREFIX" --keys-only 2>/dev/null | grep -c 'generate/' || true; }

# --- CPU isolation ---------------------------------------------------------
# Frontend (HTTP + tokenizer + KV router) pinned to FRONTEND_CORES.
# Everything else (mock workers + aiperf client) on OTHER_CORES.
# Keep FRONTEND_CORES small (e.g. 4) to make frontend CPU effects observable;
# give OTHER_CORES enough cores that the client does not starve the mockers.
export FRONTEND_CORES="${FRONTEND_CORES:-0-3}"
export OTHER_CORES="${OTHER_CORES:-4-23}"

# --- Tokenizer flags (read by the frontend's model-card loader) ------------
export DYN_TOKENIZER="${DYN_TOKENIZER:-fastokens}"          # or "default" (HF)
export DYN_TOKENIZER_CACHE="${DYN_TOKENIZER_CACHE:-1}"
export DYN_TOKENIZER_CACHE_BYTES="${DYN_TOKENIZER_CACHE_BYTES:-$((1024 * 1024 * 1024))}"  # 1 GiB

# --- Output dirs -----------------------------------------------------------
export LOG_DIR="${LOG_DIR:-$DYN_REPO/bench/logs}"
export RESULTS_DIR="${RESULTS_DIR:-$DYN_REPO/bench/results}"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
