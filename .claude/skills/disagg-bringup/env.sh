#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Source-able environment for dynamo + KVBM builds AND smoke runs.
# The Bash tool spawns non-interactive bash which does NOT source ~/.bashrc,
# so we duplicate the CUDA-toolkit + venv + NIXL exports here.
#
# Usage: prefix any cargo / maturin command with `. .sandbox/env.sh && ...`
# from the workspace root, OR `source .sandbox/env.sh` once per session.

# CUDA toolkit ---------------------------------------------------------------
if [ -d /usr/local/cuda/bin ] && [[ ":$PATH:" != *":/usr/local/cuda/bin:"* ]]; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi
export CUDA_PATH=/usr/local/cuda
export CUDA_HOME=/usr/local/cuda

# Force any silent nvcc-missing fallback in build scripts to error loudly.
# Without this, kvbm-kernels silently builds stubs and cudarc silently
# picks "fallback-latest" (CUDA 13.1) on this CUDA-13.0 host.
export KVBM_REQUIRE_CUDA=1

# Venv ----------------------------------------------------------------------
# Needed by `maturin develop`; the bash tool's env doesn't source the venv
# activator, so set VIRTUAL_ENV explicitly.
# Honors $KVBM_VENV if set (matches start-hub.sh's KVBM_REPO pattern).
export VIRTUAL_ENV=${KVBM_VENV:-/home/ryan/.venvs/dynamo-kvbm}

# NIXL libnixl + plugin alignment -------------------------------------------
# 2026-05-09: dev images shipped `nixl-cu13==0.10.1` AND a system NIXL
# (/opt/nvidia/nvda_nixl) built against that header set. The 0.10.1 wheel
# is INTERNALLY broken — its `_bindings.so` references a class-template
# member `nixlDescList<nixlBasicDesc>::operator[]` that no shipped libnixl
# exports, so `import nixl_cu13` raises `undefined symbol` at runtime.
#
# Pin to nixl-cu13>=1.0.1 (the wheel where header + libnixl + bindings are
# self-consistent, AND the v1.0.0 PyTorch-2.11 hard pin was relaxed). At
# runtime, force the wheel's libnixl + plugins to win the dlopen race —
# the system NIXL plugins were built against the 0.10.1 ABI and cannot
# satisfy backend-load checks against a 1.0.1 libnixl, surfacing as
# `Exception: No POSIX/UCX plugin found`.
#
# This block is idempotent: if nixl-cu13 is already >=1.0.0 it skips the
# upgrade; the env exports always run.
if [[ -z "${KVBM_SKIP_NIXL_FIX:-}" ]] && [ -d "$VIRTUAL_ENV" ]; then
  WHEEL_NIXL=
  for cand in "$VIRTUAL_ENV"/lib/python*/site-packages/.nixl_cu13.mesonpy.libs; do
    [ -d "$cand" ] && WHEEL_NIXL="$cand" && break
  done
  if [ -n "$WHEEL_NIXL" ]; then
    # Detect installed nixl-cu13 version; upgrade if <1.0.0.
    nixl_cu13_ver="$("$VIRTUAL_ENV/bin/pip" show nixl-cu13 2>/dev/null \
                       | awk '/^Version:/ {print $2}')"
    case "$nixl_cu13_ver" in
      "")
        echo "[env.sh] nixl-cu13 not installed; skipping NIXL fix" >&2
        ;;
      0.*)
        echo "[env.sh] upgrading nixl-cu13 ($nixl_cu13_ver -> 1.0.1) to fix wheel ABI mismatch" >&2
        "$VIRTUAL_ENV/bin/pip" install --upgrade --quiet \
          "nixl-cu13==1.0.1" "nixl==1.0.1" >&2 || \
          echo "[env.sh] WARNING: nixl-cu13 upgrade failed — runtime may fail" >&2
        ;;
    esac
    export NIXL_PLUGIN_DIR="$WHEEL_NIXL/plugins"
    export NIXL_LIB_DIR="$WHEEL_NIXL"
    # Prepend the wheel's libnixl + plugin dirs so dlopen picks them BEFORE
    # any matching system libnixl. Critical because the kvbm-py3 Rust binding
    # also dlopens libnixl into the same process, and once one libnixl is
    # cached, the OTHER consumer (nixl_cu13/_bindings.so) inherits that copy.
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$WHEEL_NIXL:"*) ;;
      *) export LD_LIBRARY_PATH="$WHEEL_NIXL:$WHEEL_NIXL/plugins:${LD_LIBRARY_PATH:-}" ;;
    esac
  fi
fi
