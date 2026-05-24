#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Shared defaults for the qwen3-tp2-experiments bundle. Sourced (not exec'd)
# from every orchestrator + launch script. Variables are set with the
# `: "${X:=default}"` idiom so an explicit env override always wins.
#
# Scope:
#   Qwen3-32B, block_size=64, max_model_len=131072 (128k), G1 prefix caching OFF.
#   Two TP=2 instances NUMA-aligned to a 4xGB200 box (or any 2x2 split).

# Repo root — the bundle lives at $REPO/.claude/skills/qwen3-tp2-experiments/.
SCRIPT_DIR_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${KVBM_REPO:=$(cd "$SCRIPT_DIR_ENV/../../.." && pwd)}"

# Venv: the container's /opt/dynamo/venv has both vllm 0.20.1 (via system
# site-packages) AND kvbm (editable, pointing at this repo).
: "${KVBM_VENV:=/opt/dynamo/venv}"

# Qwen3-32B downloaded earlier to /scratch/ryan/Qwen3-32B.
: "${KVBM_MODEL:=/scratch/ryan/Qwen3-32B}"

# Block + sequence sizing for ALL three scenarios.
: "${KVBM_HUB_BLOCK_SIZE:=64}"
: "${KVBM_HUB_MAX_SEQ_LEN:=131072}"        # 128 * 1024; 131072/64 = 2048 (clean)
: "${KVBM_MAX_MODEL_LEN:=$KVBM_HUB_MAX_SEQ_LEN}"
: "${KVBM_BLOCK_LAYOUT:=operational}"
: "${KVBM_HUB_LAYOUT:=$KVBM_BLOCK_LAYOUT}"

# G2 sizing: 128k contexts want headroom; 16 GiB per-instance G2 by default.
: "${KVBM_HUB_G2_MEMORY_GIB:=16}"
: "${KVBM_CPU_CACHE_GB:=16}"

# vLLM memory budget per instance. GB200 has 192GB; 32B bf16 weights take ~64GB
# total → 32GB per GPU at TP=2 → plenty of room. 0.85 leaves headroom for the
# 128k KV cache + activations.
: "${KVBM_TP2_GPU_MEMUTIL:=0.85}"
: "${KVBM_TP1_GPU_MEMUTIL:=0.85}"
: "${KVBM_MAX_NUM_SEQS:=4}"

# Built-from-this-repo binaries.
: "${KVBM_KVBMCTL_BIN:=$KVBM_REPO/target/debug/kvbmctl}"
: "${KVBM_HUB_BIN:=$KVBM_REPO/target/debug/kvbm_hub}"
: "${KVBM_CONNECTOR_MODULE_PATH:=kvbm.v2.vllm.connector}"

# Hub endpoints (kvbm-hub-bringup defaults).
: "${KVBM_HUB_DISCOVERY_PORT:=1337}"
: "${KVBM_HUB_CONTROL_PORT:=8337}"
: "${KVBM_HUB_VELO_PORT:=1338}"

# Readiness timeouts. TP=2 NCCL init + Qwen3-32B weight load is slow.
: "${KVBM_HUB_READY_TIMEOUT:=300}"
: "${KVBM_VLLM_READY_TIMEOUT:=900}"

# Logs land under /tmp/kvbm-experiments/<ts>-<label>/.
: "${KVBM_EXPERIMENTS_DIR:=/tmp/kvbm-experiments}"

export KVBM_REPO KVBM_VENV KVBM_MODEL \
       KVBM_HUB_BLOCK_SIZE KVBM_HUB_MAX_SEQ_LEN KVBM_MAX_MODEL_LEN \
       KVBM_BLOCK_LAYOUT KVBM_HUB_LAYOUT \
       KVBM_HUB_G2_MEMORY_GIB KVBM_CPU_CACHE_GB \
       KVBM_TP2_GPU_MEMUTIL KVBM_TP1_GPU_MEMUTIL KVBM_MAX_NUM_SEQS \
       KVBM_KVBMCTL_BIN KVBM_HUB_BIN KVBM_CONNECTOR_MODULE_PATH \
       KVBM_HUB_DISCOVERY_PORT KVBM_HUB_CONTROL_PORT KVBM_HUB_VELO_PORT \
       KVBM_HUB_READY_TIMEOUT KVBM_VLLM_READY_TIMEOUT \
       KVBM_EXPERIMENTS_DIR
