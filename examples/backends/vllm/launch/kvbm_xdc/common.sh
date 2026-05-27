# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Source-only helpers shared by the KVBM cross-datacenter launch scripts.

kvbm_xdc_prepend_path() {
  local var_name=$1
  local entry=$2
  [ -n "$entry" ] || return 0
  [ -d "$entry" ] || return 0

  local current=${!var_name-}
  case ":$current:" in
    *":$entry:"*) ;;
    *) export "$var_name"="$entry${current:+:$current}" ;;
  esac
}

kvbm_xdc_resolve_python() {
  if [ -n "${PYTHON_BIN:-}" ]; then
    if [ ! -x "$PYTHON_BIN" ]; then
      echo "PYTHON_BIN is set but not executable: $PYTHON_BIN" >&2
      return 2
    fi
    export PYTHON_BIN
    return 0
  fi

  if [ -n "${VENV:-}" ] && [ -x "$VENV/bin/python" ]; then
    PYTHON_BIN="$VENV/bin/python"
  elif [ -x /opt/dynamo/venv/bin/python ]; then
    VENV=/opt/dynamo/venv
    PYTHON_BIN="$VENV/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python3)
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python)
  else
    echo "No usable Python found; set PYTHON_BIN or VENV" >&2
    return 2
  fi

  export VENV PYTHON_BIN
}

kvbm_xdc_configure_base_paths() {
  if [ -n "${VENV:-}" ] && [ -d "$VENV/bin" ]; then
    kvbm_xdc_prepend_path PATH "$VENV/bin"
  fi
  kvbm_xdc_prepend_path PATH /usr/local/cargo/bin
  kvbm_xdc_prepend_path PATH /usr/local/cuda/bin
  export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
}

kvbm_xdc_configure_python_paths() {
  local worktree=${WORKTREE:-/workspace}
  KVBM_XDC_USE_WORKTREE_PYTHON=${KVBM_XDC_USE_WORKTREE_PYTHON:-0}

  case "$KVBM_XDC_USE_WORKTREE_PYTHON" in
    0|false|off)
      export KVBM_XDC_USE_WORKTREE_PYTHON
      ;;
    1|true|on)
      local kvbm_core
      kvbm_core=$(find "$worktree/lib/bindings/kvbm/python/kvbm" -maxdepth 1 -name '_core*.so' -print -quit 2>/dev/null || true)
      if [ -z "$kvbm_core" ]; then
        echo "KVBM_XDC_USE_WORKTREE_PYTHON=1 but $worktree/lib/bindings/kvbm/python/kvbm has no built _core*.so" >&2
        return 2
      fi

      local path
      for path in \
        "$worktree/lib/bindings/python/src" \
        "$worktree/lib/bindings/kvbm/python" \
        "$worktree/components/src"; do
        kvbm_xdc_prepend_path PYTHONPATH "$path"
      done
      export KVBM_XDC_USE_WORKTREE_PYTHON
      ;;
    *)
      echo "KVBM_XDC_USE_WORKTREE_PYTHON must be 0 or 1, got $KVBM_XDC_USE_WORKTREE_PYTHON" >&2
      return 2
      ;;
  esac

  local path
  if [ -n "${NIXL_PREFIX:-}" ]; then
    for path in \
      "$NIXL_PREFIX/lib64" \
      "$NIXL_PREFIX/lib" \
      "$NIXL_PREFIX/lib/x86_64-linux-gnu" \
      "$NIXL_PREFIX/plugins"; do
      kvbm_xdc_prepend_path LD_LIBRARY_PATH "$path"
    done
    if [ -z "${KVBM_XDC_PRESERVE_NIXL_ENV:-}" ] \
      && [ -d "$NIXL_PREFIX/plugins" ]; then
      export NIXL_PLUGIN_DIR="$NIXL_PREFIX/plugins"
    fi
    if [ -z "${KVBM_XDC_PRESERVE_NIXL_ENV:-}" ]; then
      if [ -d "$NIXL_PREFIX/lib64" ]; then
        export NIXL_LIB_DIR="$NIXL_PREFIX/lib64"
      elif [ -d "$NIXL_PREFIX/lib" ]; then
        export NIXL_LIB_DIR="$NIXL_PREFIX/lib"
      fi
    fi
  fi

  for path in \
    "$worktree/.image-target/debug/deps" \
    "$worktree/.image-target-kvbm/debug/deps" \
    /usr/local/lib/python*/site-packages/.nixl_cu*.mesonpy.libs \
    /usr/local/lib/python*/site-packages/nixl_cu*.libs; do
    kvbm_xdc_prepend_path LD_LIBRARY_PATH "$path"
    kvbm_xdc_prepend_path LD_LIBRARY_PATH "$path/plugins"
    if [ -z "${NIXL_PREFIX:-}" ] \
      && [ -z "${KVBM_XDC_PRESERVE_NIXL_ENV:-}" ] \
      && [ -d "$path/plugins" ]; then
      export NIXL_PLUGIN_DIR="$path/plugins"
    fi
    if [ -z "${NIXL_PREFIX:-}" ] \
      && [ -z "${KVBM_XDC_PRESERVE_NIXL_ENV:-}" ]; then
      case "$path" in
        */.nixl_cu*.mesonpy.libs) export NIXL_LIB_DIR="$path" ;;
      esac
    fi
  done
}

kvbm_xdc_configure_ucx_fallback() {
  KVBM_FORCE_UCX_LOCAL_FALLBACK=${KVBM_FORCE_UCX_LOCAL_FALLBACK:-auto}

  if [ "$KVBM_FORCE_UCX_LOCAL_FALLBACK" = "auto" ]; then
    if [ "${NODE_ROLE:-}" = "all" ] \
      && [ "${KVBM_TRANSFER_TOPOLOGY:-}" = "kvbm-hub" ] \
      && [ ! -d /sys/module/nvidia_peermem ]; then
      KVBM_FORCE_UCX_LOCAL_FALLBACK=1
    else
      KVBM_FORCE_UCX_LOCAL_FALLBACK=0
    fi
  fi
  export KVBM_FORCE_UCX_LOCAL_FALLBACK

  export UCX_RCACHE_MAX_UNRELEASED=${UCX_RCACHE_MAX_UNRELEASED:-1024}
  if [ "$KVBM_FORCE_UCX_LOCAL_FALLBACK" = "1" ]; then
    export UCX_TLS=${UCX_TLS:-tcp,sm,self,cuda_copy,cuda_ipc}
    export UCX_NET_DEVICES=${UCX_NET_DEVICES:-^all}
    export UCX_LOG_LEVEL=${UCX_LOG_LEVEL:-error}
  fi
}

kvbm_xdc_source_optional_runtime_env() {
  SOURCE_KVBM_RUNTIME_ENV=${SOURCE_KVBM_RUNTIME_ENV:-0}
  [ "$SOURCE_KVBM_RUNTIME_ENV" = "1" ] || return 0

  local worktree=${WORKTREE:-/workspace}
  local env_script=${KVBM_RUNTIME_ENV_SCRIPT:-}
  if [ -z "$env_script" ]; then
    echo "SOURCE_KVBM_RUNTIME_ENV=1 requires KVBM_RUNTIME_ENV_SCRIPT" >&2
    return 2
  fi
  if [ ! -f "$env_script" ]; then
    echo "SOURCE_KVBM_RUNTIME_ENV=1 but runtime env script was not found: $env_script" >&2
    return 2
  fi

  export KVBM_REPO=${KVBM_REPO:-$worktree}
  if [ -n "${VENV:-}" ]; then
    export KVBM_VENV=${KVBM_VENV:-$VENV}
  fi

  set +u
  # shellcheck source=/dev/null
  . "$env_script"
  set -u
}

kvbm_xdc_prepare_runtime() {
  kvbm_xdc_resolve_python
  kvbm_xdc_configure_base_paths
  kvbm_xdc_configure_ucx_fallback
  kvbm_xdc_source_optional_runtime_env
  kvbm_xdc_configure_python_paths
}
