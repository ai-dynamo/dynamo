# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import ctypes.util
import importlib.resources as ir
import os
from pathlib import Path

import torch


def is_cuda_13() -> bool:
    """
    Returns True if either:
      - the installed PyTorch was built against CUDA 13.x, or
      - the CUDA runtime on this machine reports version 13.x.

    Works even when no GPU is visible; only the runtime library needs to be present
    for the fallback check.
    """
    # 1) PyTorch build CUDA (e.g., "13.0", "12.1"); None if CPU build
    try:
        built = getattr(torch.version, "cuda", None)
        if built:
            major = int(str(built).split(".")[0])
            if major == 13:
                return True
    except Exception:
        pass

    # 2) Actual CUDA runtime (libcudart) on the system
    try:
        path = ctypes.util.find_library("cudart") or "libcudart.so"
        libcudart = ctypes.CDLL(path)
        ver = ctypes.c_int()
        rc = libcudart.cudaRuntimeGetVersion(ctypes.byref(ver))
        if rc == 0 and ver.value:
            # e.g., 13000 -> 13.0
            major = ver.value // 1000
            return major == 13
    except Exception:
        pass

    return False


def set_cu13_nixl_plugin_path() -> str:
    """
    If NIXL_PLUGIN_DIR is unset/empty, set it to:
        <kvbm package dir>/nixl-cu13/plugins
    Returns the resolved path that will be used.
    """
    cur = os.environ.get("NIXL_PLUGIN_DIR")
    if cur:
        return cur

    # Prefer importlib.resources for installed wheels
    try:
        # 'kvbm' is your top-level package (adjust if different)
        base = ir.files("kvbm") / "nixl-cu13" / "plugins"
        path = Path(base)
    except Exception:
        # Fallback to this file's location
        path = Path(__file__).resolve().parent / "nixl-cu13" / "plugins"

    os.environ["NIXL_PLUGIN_DIR"] = str(path)
    return str(path)


def is_dyn_runtime_enabled() -> bool:
    """
    Return True if DYN_RUNTIME_ENABLED_KVBM is set to '1' or 'true' (case-insensitive).
    DYN_RUNTIME_ENABLED_KVBM indicates if KVBM should use the existing DistributedRuntime
    in the current environment.
    """
    val = os.environ.get("DYN_RUNTIME_ENABLED_KVBM", "").strip().lower()
    return val in {"1", "true"}
