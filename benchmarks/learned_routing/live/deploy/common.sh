# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared settings for the live GPU deployment (sourced, not executed). Site-specific values (the
# cluster's ssh alias, Slurm account and partition, the shared-filesystem roots, the staged image,
# model and toolchain) have no defaults: set them in site.env next to this file (copy
# site.env.example), in the file named by LR_SITE_ENV, or in the environment, which wins. The entry
# points check what they need (lr_need), and stage_source.sh writes the node-side site settings next
# to the staged deploy scripts so jobs read the same values. Durable state lives on the cluster's
# shared filesystem; per-job hot state lives on the allocated node's local disk. See README.md.

LR_DEPLOY_DIR="${LR_DEPLOY_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Read KEY=VALUE lines (quotes and $VAR references allowed, # comments ignored) and export every key
# that is not already set.
lr_load_site_env() {
  local file="$1" line key value
  [[ -f "$file" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    key="${BASH_REMATCH[2]}"
    [[ -z "${!key+x}" ]] || continue
    eval "value=${BASH_REMATCH[3]}"
    export "$key=$value"
  done < "$file"
}
lr_load_site_env "${LR_SITE_ENV:-$LR_DEPLOY_DIR/site.env}"

# Durable roots on the cluster's shared filesystem (shared by login and compute nodes).
LR_SHARED_ROOT="${LR_SHARED_ROOT:-}"
LR_LIVE_ROOT="${LR_LIVE_ROOT:-}"
# A directory that already holds the vLLM 0.24.0 image and the pinned Qwen3-32B checkpoint, both
# read-only inputs here. LR_IMAGE_SQSH and LR_MODEL_MASTER can also name them directly.
LR_PRIOR_RUNTIME="${LR_PRIOR_RUNTIME:-}"
LR_IMAGE_SQSH="${LR_IMAGE_SQSH:-${LR_PRIOR_RUNTIME:+$LR_PRIOR_RUNTIME/images/vllm-v0.24.0.sqsh}}"
LR_IMAGE_REF="${LR_IMAGE_REF:-vllm/vllm-openai@sha256:f9de5cd9fa907fbf6dbba691eb7db095d48ad58ea283e3eba7142f9a91e186e8}"
LR_IMAGE_SQSH_BYTES="${LR_IMAGE_SQSH_BYTES:-18085605376}"
LR_MODEL_ID="${LR_MODEL_ID:-Qwen/Qwen3-32B}"
LR_MODEL_REVISION="${LR_MODEL_REVISION:-9216db5781bf21249d130ec9da846c4624c16137}"
LR_MODEL_MASTER="${LR_MODEL_MASTER:-${LR_PRIOR_RUNTIME:+$LR_PRIOR_RUNTIME/models/Qwen3-32B}}"
LR_MODEL_MANIFEST="${LR_MODEL_MANIFEST:-$LR_DEPLOY_DIR/qwen3-32b-9216db57.manifest.json}"
LR_TOOLCHAIN_ENV="${LR_TOOLCHAIN_ENV:-${LR_SHARED_ROOT:+$LR_SHARED_ROOT/toolchains/env.sh}}"
LR_VLLM_VERSION="${LR_VLLM_VERSION:-0.24.0}"
# Site settings the allocated nodes need; stage_source.sh ships their resolved values (lr_node_site_env).
LR_NODE_SITE_KEYS=(LR_SHARED_ROOT LR_LIVE_ROOT LR_PRIOR_RUNTIME LR_IMAGE_SQSH LR_MODEL_MASTER LR_TOOLCHAIN_ENV
  LR_NODE_BASE LR_NODE_TAG)

# Source identity: the campaign commit to build. stage_source.sh prints it.
LR_COMMIT="${LR_COMMIT:-}"
LR_TAG="${LR_COMMIT:0:12}"
LR_SRC="${LR_SRC:-$LR_LIVE_ROOT/src/$LR_TAG}"
LR_WHEEL_DIR="${LR_WHEEL_DIR:-$LR_LIVE_ROOT/wheels/$LR_TAG}"
LR_VENV="${LR_VENV:-$LR_LIVE_ROOT/venvs/$LR_TAG-vllm$LR_VLLM_VERSION}"
LR_CACHE_ROOT="${LR_CACHE_ROOT:-$LR_LIVE_ROOT/cache}"
LR_RUNS_ROOT="${LR_RUNS_ROOT:-$LR_LIVE_ROOT/runs}"

# Per-job node-local state. /raid/scratch is world-writable local NVMe on DGX H100 nodes; /tmp is
# tmpfs (RAM) and is only the fallback. LR_NODE_TAG (optional) prefixes the per-job directory name.
LR_JOB="${SLURM_JOB_ID:-nojob}"
LR_NODE_BASE="${LR_NODE_BASE:-/raid/scratch}"
LR_NODE_TAG="${LR_NODE_TAG:-}"
LR_NODE_ROOT="${LR_NODE_ROOT:-$LR_NODE_BASE/${LR_NODE_TAG:+$LR_NODE_TAG-}lr-$LR_JOB}"
LR_MODEL_DIR="${LR_MODEL_DIR:-$LR_NODE_ROOT/models/Qwen3-32B}"
LR_NODE_SQSH="${LR_NODE_SQSH:-$LR_NODE_ROOT/vllm-v$LR_VLLM_VERSION.sqsh}"
LR_CONTAINER="${LR_CONTAINER:-lr-vllm-$LR_JOB}"

# Serving topology. Workers are TP2 pairs of GPUs; masks follow the DGX H100 NUMA layout
# (GPUs 0-3 and CPUs 0-55,112-167 on NUMA 0; GPUs 4-7 and CPUs 56-111,168-223 on NUMA 1).
# health_check.py records nvidia-smi topo so a different layout is visible.
LR_WORKERS_PER_NODE="${LR_WORKERS_PER_NODE:-4}"
LR_HTTP_PORT="${LR_HTTP_PORT:-18000}"
LR_NAMESPACE="${LR_NAMESPACE:-lr-live-$LR_JOB}"
LR_FRONTEND_CPUS="${LR_FRONTEND_CPUS:-28-41,140-153}"
LR_CLIENT_CPUS="${LR_CLIENT_CPUS:-42-55,154-167}"
LR_AUX_CPUS="${LR_AUX_CPUS:-84-111,196-223}"
lr_worker_gpus() { local i=$1; echo "$((2 * i)),$((2 * i + 1))"; }
lr_worker_cpus() {
  case "$1" in
    0) echo "0-13,112-125" ;;
    1) echo "14-27,126-139" ;;
    2) echo "56-69,168-181" ;;
    3) echo "70-83,182-195" ;;
    *) echo "error: no CPU mask for local worker $1" >&2; return 1 ;;
  esac
}
lr_worker_system_port() { echo $((19100 + $1)); }
lr_worker_kv_port() { echo $((15600 + 10 * $1)); }

# Container mounts: the shared-filesystem root (durable inputs and outputs) plus the job's
# node-local root.
LR_MOUNTS="${LR_MOUNTS:-${LR_SHARED_ROOT:+$LR_SHARED_ROOT:$LR_SHARED_ROOT,}$LR_NODE_ROOT:$LR_NODE_ROOT}"

export PYTHONDONTWRITEBYTECODE=1

die() {
  echo "error: $*" >&2
  exit 1
}

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"
}

# lr_need VAR... : fail unless every named site setting is set.
lr_need() {
  local var
  for var in "$@"; do
    [[ -n "${!var:-}" ]] || die "set $var in site.env (see site.env.example) or the environment"
  done
}

# The node-side site settings as site.env lines (stage_source.sh writes them next to the staged
# deploy scripts, where common.sh reads them on the allocated nodes).
lr_node_site_env() {
  local key
  for key in "${LR_NODE_SITE_KEYS[@]}"; do
    [[ -z "${!key:-}" ]] || printf '%s=%q\n' "$key" "${!key}"
  done
}

lr_require_commit() {
  [[ "$LR_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "set LR_COMMIT to the full 40-hex campaign commit"
}

# Environment for every process inside the vLLM container: CUDA 13 forward compatibility on a
# 535-series driver, offline Hugging Face, shared compile caches,
# and no inherited router or worker knobs (the policy plan is the only source of router config).
lr_container_env() {
  export VLLM_ENABLE_CUDA_COMPATIBILITY=1
  export VLLM_CUDA_COMPATIBILITY_PATH=/usr/local/cuda-13.0/compat
  export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}"
  export HOME="$LR_NODE_ROOT/home"
  export XDG_CACHE_HOME="$LR_NODE_ROOT/cache"
  export HF_HOME="$LR_NODE_ROOT/hf-home"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export TORCHINDUCTOR_CACHE_DIR="$LR_CACHE_ROOT/vllm-$LR_VLLM_VERSION/torchinductor"
  export VLLM_CACHE_ROOT="$LR_CACHE_ROOT/vllm-$LR_VLLM_VERSION/vllm"
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export PYTHONHASHSEED=0
  mkdir -p "$HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"
  local name
  for name in $(compgen -e); do
    case "$name" in
      DYN_ROUTER_* | DYN_ACTIVE_* | DYN_FPM_* | DYN_FORWARDPASS_* | DYN_FIDELITY_* \
        | DYN_SESSION_* | DYN_MIGRATION_* | DYN_KV_* | DYN_SHARED_CACHE_* | DYN_USE_REMOTE_INDEXER \
        | DYN_SERVE_INDEXER | DYN_ENABLE_SESSION_PREFIX_INDEX | DYN_LOAD_AWARE | HF_TOKEN \
        | HUGGING_FACE_HUB_TOKEN | PYTHONPATH | VLLM_USE_* | VLLM_ATTENTION_BACKEND)
        unset "$name"
        ;;
    esac
  done
}

# Runtime planes. File discovery needs every process on one node (inotify); the two-node variant
# switches to etcd (job_node.sh sets LR_DISCOVERY=etcd and ETCD_ENDPOINTS).
lr_runtime_env() {
  export DYN_NAMESPACE="$LR_NAMESPACE"
  export DYN_DISCOVERY_BACKEND="${LR_DISCOVERY:-file}"
  export DYN_REQUEST_PLANE=tcp DYN_RESPONSE_PLANE=tcp DYN_EVENT_PLANE=zmq
  export DYN_FILE_KV="$LR_NODE_ROOT/discovery"
  mkdir -p "$DYN_FILE_KV"
  local ip
  ip="$(getent ahostsv4 "$(hostname)" | awk 'NR == 1 {print $1}')"
  if [[ -n "$ip" ]]; then
    export DYN_TCP_RPC_HOST="$ip" DYN_TCP_RESPONSE_STREAM_HOST="$ip" DYN_EVENT_PLANE_HOST="$ip"
  fi
}

lr_node_ip() {
  getent ahostsv4 "$1" | awk 'NR == 1 {print $1}'
}
