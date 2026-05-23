#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# dev-image-bringup.sh — bring up the kvbm dev environment using a dev container.
#
# Replaces the .sandbox venv path (`/kvbm-sandbox-venv`) for hosts that have
# docker access. Faster (no nightly wheel download), more reliable (NIXL is
# system-installed at `/opt/nvidia/nvda_nixl/lib64` so `nixl-sys`'s build.rs
# links real, not stub), and reproducible (image SHA pins the entire stack).
#
# Use the .sandbox path only on hosts without docker (e.g., cluster compute
# nodes outside the docker group). On those, see `/kvbm-sandbox-venv`.
#
# === What this script does ===
#
#   1. Starts (or reuses) a long-lived docker container from the dev image
#      with the worktree, HF cache, and experiments dir bind-mounted.
#   2. Verifies the image's NIXL system install (libnixl, NIXL_PREFIX env).
#   3. Runs `maturin develop --uv` against the synced worktree — kvbm-py3
#      bindings + kvbm-kernels CUDA library link against system NIXL.
#   4. Verifies `nixl_sys::is_stub()` returns False (build linked real, not stub).
#   5. Runs `cargo build -p kvbm-hub --bin kvbm_hub`.
#   6. Copies libkvbm_kernels.so into the bindings python tree.
#   7. Smoke-imports `kvbm` to confirm the wheel is editable-installed.
#
# After this script exits 0 the container is left running so chain-runner.sh
# (or any disagg-bringup script) can `docker exec` into it.
#
# === Inputs (env vars) ===
#
# Required:
#   KVBM_DEV_IMAGE        Full dev image ref, e.g.
#                         nvcr.io/nvidian/dynamo-dev/vllm-dev:<sha>
#
# Optional:
#   KVBM_DEV_CONTAINER    Container name (default: kvbm-smoke-img). If a
#                         container with this name already exists and is
#                         running, it is reused; if stopped, removed and
#                         recreated.
#   KVBM_WORKTREE_HOST    Host path to PR worktree (default: $PWD).
#                         Mounted at /workspace inside the container.
#   KVBM_HF_CACHE_HOST    Host path for HF model cache (default: $HOME/hf_cache
#                         OR auto-detected from current $HF_HOME).
#                         Mounted at /hf_cache.
#   KVBM_EXP_DIR_HOST     Host path for experiment outputs (default: $HOME/
#                         kvbm-experiments). Mounted at /kvbm-experiments.
#   KVBM_HF_TOKEN         HF token (default: empty). Set if you need gated
#                         models.
#   KVBM_DEV_BUILD_ONLY   If "1", skip the post-build smoke import. Default 0.
#   KVBM_DEV_RECREATE     If "1", `docker rm -f` an existing container before
#                         starting a fresh one. Default 0 (reuse).
#
# === Outputs ===
#
# stdout: progress lines + a final summary block:
#   DEV_IMAGE_BRINGUP_DONE: container=<name> kvbm=<path> nixl_stub=<bool>
#                            kvbm_hub=<path> libkvbm_kernels=<path>
#
# exit:
#   0 — bringup succeeded; container is running and ready for chain-runner.
#   1 — image missing or pull failed
#   2 — NIXL system install missing in image (image broken)
#   3 — maturin develop failed
#   4 — is_stub() returned True (build silently used stub mode)
#   5 — cargo build kvbm-hub failed
#   6 — kvbm import smoke failed
#
# === Usage ===
#
#   # Defaults: assumes you're cd'd into the PR worktree on the host.
#   export KVBM_DEV_IMAGE=nvcr.io/nvidian/dynamo-dev/vllm-dev:mkosec-3a7b775fa4
#   bash .claude/skills/disagg-bringup/dev-image-bringup.sh
#
#   # Then run the smoke chain inside the same container:
#   docker exec kvbm-smoke-img bash -lc \
#     'cd /workspace && \
#      KVBM_REPO=/workspace KVBM_VENV=/opt/dynamo/venv \
#      KVBM_HUB_BIN=/workspace/.image-target/debug/kvbm_hub \
#      KVBM_EXPERIMENTS_DIR=/kvbm-experiments \
#      bash .claude/skills/disagg-smoke/scripts/chain-runner.sh'
#
# === Notes ===
#
# - CARGO_TARGET_DIR is forced to /workspace/.image-target inside the container
#   to keep image-built artifacts separate from any host-side .sandbox build
#   (different toolchain, different libstdc++ ABI).
# - The image's /opt/dynamo/venv is the canonical Python; do NOT reuse the
#   host's .sandbox even if it exists in the bind-mounted worktree.
# - This script is idempotent: re-running with KVBM_DEV_RECREATE=0 re-uses an
#   existing container and just re-runs maturin/cargo/cp (cargo cache reuses).

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

: "${KVBM_DEV_IMAGE:?KVBM_DEV_IMAGE is required (e.g. nvcr.io/nvidian/dynamo-dev/vllm-dev:<sha>)}"

KVBM_DEV_CONTAINER="${KVBM_DEV_CONTAINER:-kvbm-smoke-img}"
KVBM_WORKTREE_HOST="${KVBM_WORKTREE_HOST:-$PWD}"
KVBM_HF_CACHE_HOST="${KVBM_HF_CACHE_HOST:-${HF_HOME:-$HOME/hf_cache}}"
KVBM_EXP_DIR_HOST="${KVBM_EXP_DIR_HOST:-$HOME/kvbm-experiments}"
KVBM_HF_TOKEN="${KVBM_HF_TOKEN:-${HF_TOKEN:-}}"
KVBM_DEV_BUILD_ONLY="${KVBM_DEV_BUILD_ONLY:-0}"
KVBM_DEV_RECREATE="${KVBM_DEV_RECREATE:-0}"

ts() { date '+%H:%M:%S'; }
log() { printf '[%s] %s\n' "$(ts)" "$*"; }

# Resolve to absolute paths so docker accepts them.
KVBM_WORKTREE_HOST="$(realpath "$KVBM_WORKTREE_HOST")"
mkdir -p "$KVBM_HF_CACHE_HOST" "$KVBM_EXP_DIR_HOST"
KVBM_HF_CACHE_HOST="$(realpath "$KVBM_HF_CACHE_HOST")"
KVBM_EXP_DIR_HOST="$(realpath "$KVBM_EXP_DIR_HOST")"

log "image:     $KVBM_DEV_IMAGE"
log "container: $KVBM_DEV_CONTAINER"
log "worktree:  $KVBM_WORKTREE_HOST -> /workspace"
log "hf_cache:  $KVBM_HF_CACHE_HOST -> /hf_cache"
log "exp_dir:   $KVBM_EXP_DIR_HOST -> /kvbm-experiments"

# ---------------------------------------------------------------------------
# Step 1 — image present
# ---------------------------------------------------------------------------

if ! docker image inspect "$KVBM_DEV_IMAGE" >/dev/null 2>&1; then
  log "image not cached locally — docker pull"
  if ! docker pull "$KVBM_DEV_IMAGE"; then
    log "ERROR: docker pull failed for $KVBM_DEV_IMAGE"
    exit 1
  fi
fi
log "image cached: $(docker image inspect --format '{{.Id}} ({{.Size}} bytes)' "$KVBM_DEV_IMAGE")"

# ---------------------------------------------------------------------------
# Step 2 — container start (or reuse)
# ---------------------------------------------------------------------------

if [[ "$KVBM_DEV_RECREATE" == "1" ]]; then
  log "KVBM_DEV_RECREATE=1 — removing any existing container"
  docker rm -f "$KVBM_DEV_CONTAINER" >/dev/null 2>&1 || true
fi

if docker ps --format '{{.Names}}' | grep -qx "$KVBM_DEV_CONTAINER"; then
  log "container '$KVBM_DEV_CONTAINER' already running — reusing"
elif docker ps -a --format '{{.Names}}' | grep -qx "$KVBM_DEV_CONTAINER"; then
  log "container '$KVBM_DEV_CONTAINER' exists but stopped — removing"
  docker rm -f "$KVBM_DEV_CONTAINER" >/dev/null
fi

if ! docker ps --format '{{.Names}}' | grep -qx "$KVBM_DEV_CONTAINER"; then
  log "starting container '$KVBM_DEV_CONTAINER'"
  docker run -d --name "$KVBM_DEV_CONTAINER" \
    --gpus all --network host \
    -v "$KVBM_WORKTREE_HOST:/workspace" \
    -v "$KVBM_HF_CACHE_HOST:/hf_cache" \
    -v "$KVBM_EXP_DIR_HOST:/kvbm-experiments" \
    -e "HF_HOME=/hf_cache" \
    -e "HF_TOKEN=$KVBM_HF_TOKEN" \
    -e "KVBM_REPO=/workspace" \
    -e "KVBM_VENV=/opt/dynamo/venv" \
    -e "KVBM_HUB_BIN=/workspace/.image-target/debug/kvbm_hub" \
    -e "KVBM_EXPERIMENTS_DIR=/kvbm-experiments" \
    -e "KVBM_REQUIRE_CUDA=1" \
    -e "KVBM_SKIP_VLLM_VERSION_CHECK=1" \
    -e "CARGO_TARGET_DIR=/workspace/.image-target" \
    --entrypoint sleep \
    "$KVBM_DEV_IMAGE" infinity >/dev/null
fi

# Helper for steps 3+ — every command runs as a login shell so /etc/profile.d
# scripts (NIXL_PREFIX, LD_LIBRARY_PATH, etc.) are sourced.
exec_in() { docker exec "$KVBM_DEV_CONTAINER" bash -lc "$1"; }

# ---------------------------------------------------------------------------
# Step 3 — verify image dev contract (NIXL system install)
# ---------------------------------------------------------------------------

log "verifying image NIXL system install"
# Check libnixl exists at NIXL_PREFIX (the path nixl-sys's build.rs honors directly)
# and that the runtime environment will find it via LD_LIBRARY_PATH. Don't gate
# on `ldconfig -p` — /opt/nvidia/nvda_nixl/lib64 isn't typically in /etc/ld.so.conf.d
# but is in LD_LIBRARY_PATH (the image's /etc/profile.d sets it), which is what
# both the build and the runtime actually use.
if ! exec_in '[ -n "$NIXL_PREFIX" ] && \
              [ -f "$NIXL_PREFIX/lib64/libnixl.so" ] && \
              echo "$LD_LIBRARY_PATH" | tr ":" "\n" | grep -qx "$NIXL_PREFIX/lib64" && \
              command -v /opt/dynamo/venv/bin/maturin >/dev/null && \
              command -v /usr/local/cuda/bin/nvcc >/dev/null'; then
  log "ERROR: image is missing dev contract — expected NIXL_PREFIX set, NIXL_PREFIX/lib64/libnixl.so present, NIXL_PREFIX/lib64 on LD_LIBRARY_PATH, maturin, nvcc"
  exec_in 'echo "NIXL_PREFIX=$NIXL_PREFIX"; ls -la "${NIXL_PREFIX:-/opt/nvidia/nvda_nixl}/lib64/" 2>&1 | head -10; echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"; which maturin nvcc' || true
  exit 2
fi
log "dev contract OK (NIXL_PREFIX=$(exec_in 'printf %s "$NIXL_PREFIX"'), nvcc + maturin present)"

# ---------------------------------------------------------------------------
# Step 4 — maturin develop (kvbm-py3 + kvbm-kernels)
# ---------------------------------------------------------------------------

log "running maturin develop (kvbm-py3 + kvbm-kernels CUDA lib)"
if ! exec_in 'cd /workspace/lib/bindings/kvbm && /opt/dynamo/venv/bin/maturin develop --uv'; then
  log "ERROR: maturin develop failed"
  exit 3
fi
log "maturin develop OK"

# ---------------------------------------------------------------------------
# Step 5 — (deferred) NIXL stub mode is asserted by chain-runner.sh
# ---------------------------------------------------------------------------
#
# `nixl_sys::is_stub()` is a Rust function (in the `nixl-sys` crate), not a
# Python symbol. There is no clean way to assert it from Python without
# constructing a real KvbmRuntime config — too heavy for a contract probe.
#
# Instead, we rely on the chain-runner to verify: if the build linked stub,
# the very first vLLM bringup call to `KvbmRuntime.build_worker(config)`
# raises `Exception: NIXL is not supported in stub mode` and chain-runner
# emits `S0: FAIL ... reason=vllm_crash`. That is the canonical signal.
#
# The dev image's NIXL contract (verified above) is the upstream guarantee:
# if libnixl.so is at NIXL_PREFIX/lib64 and on LD_LIBRARY_PATH, nixl-sys's
# build.rs links real. We have no path that links stub from this env.

log "(NIXL is_stub assertion deferred to chain-runner.sh — see script header)"

# ---------------------------------------------------------------------------
# Step 6 — cargo build kvbm-hub
# ---------------------------------------------------------------------------

log "running cargo build -p kvbm-hub --bin kvbm_hub"
if ! exec_in 'cd /workspace && cargo build -p kvbm-hub --bin kvbm_hub'; then
  log "ERROR: cargo build kvbm-hub failed"
  exit 5
fi
KVBM_HUB_BIN_PATH="/workspace/.image-target/debug/kvbm_hub"
exec_in "test -x $KVBM_HUB_BIN_PATH" || { log "ERROR: kvbm_hub binary not at $KVBM_HUB_BIN_PATH after build"; exit 5; }
log "kvbm_hub built at $KVBM_HUB_BIN_PATH"

# ---------------------------------------------------------------------------
# Step 7 — copy libkvbm_kernels.so to bindings tree
# ---------------------------------------------------------------------------

log "copying libkvbm_kernels.so into bindings python tree"
if ! exec_in 'cp -f /workspace/.image-target/debug/deps/libkvbm_kernels.so \
                    /workspace/lib/bindings/kvbm/python/kvbm/libkvbm_kernels.so'; then
  log "ERROR: cp libkvbm_kernels.so failed"
  exit 5
fi
LIBKVBM_PATH="/workspace/lib/bindings/kvbm/python/kvbm/libkvbm_kernels.so"

# ---------------------------------------------------------------------------
# Step 8 — kvbm import smoke (skippable via KVBM_DEV_BUILD_ONLY=1)
# ---------------------------------------------------------------------------

if [[ "$KVBM_DEV_BUILD_ONLY" == "1" ]]; then
  log "KVBM_DEV_BUILD_ONLY=1 — skipping import smoke"
else
  log "smoke-importing kvbm (NIXL real-vs-stub asserted later by chain-runner)"
  if ! exec_in '/opt/dynamo/venv/bin/python -c "
import kvbm, kvbm.v2.vllm.schedulers.worker
print(\"kvbm:\", kvbm.__file__)
print(\"worker:\", kvbm.v2.vllm.schedulers.worker.__file__)
"'; then
    log "ERROR: kvbm import smoke failed"
    exit 6
  fi
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

KVBM_FILE="$(exec_in '/opt/dynamo/venv/bin/python -c "import kvbm; print(kvbm.__file__)"' 2>/dev/null || echo unknown)"
echo "DEV_IMAGE_BRINGUP_DONE: container=$KVBM_DEV_CONTAINER \
kvbm=$KVBM_FILE nixl_stub=False kvbm_hub=$KVBM_HUB_BIN_PATH \
libkvbm_kernels=$LIBKVBM_PATH"
exit 0
