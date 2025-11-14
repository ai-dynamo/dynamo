# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import importlib
import os
from typing import Callable, Optional


def is_dyn_runtime_enabled() -> bool:
    """
    Return True if DYN_RUNTIME_ENABLED_KVBM is set to '1' or 'true' (case-insensitive).
    DYN_RUNTIME_ENABLED_KVBM indicates if KVBM should use the existing DistributedRuntime
    in the current environment.

    WRN: Calling DistributedRuntime.detached() can crash the entire process if
    dependencies are not satisfied, and it cannot be caught with try/except in Python.
    TODO: Make DistributedRuntime.detached() raise a catchable Python exception and
    avoid crashing the process.
    """
    val = os.environ.get("DYN_RUNTIME_ENABLED_KVBM", "").strip().lower()
    return val in {"1", "true"}


def maybe_import_offload_filter() -> Optional[Callable[[int], bool]]:
    have_module_env = "DYN_KVBM_CONNECTOR_OFFLOAD_FILTER_MODULE" in os.environ
    have_class_env = "DYN_KVBM_CONNECTOR_OFFLOAD_FILTER_CLASS" in os.environ
    if not have_module_env and not have_class_env:
        return None
    elif have_module_env != have_class_env:
        raise ValueError("Either both or neither of DYN_KVBM_CONNECTOR_OFFLOAD_FILTER_MODULE and DYN_KVBM_CONNECTOR_OFFLOAD_FILTER_CLASS must be set")

    module_str = os.environ["DYN_KVBM_CONNECTOR_OFFLOAD_FILTER_MODULE"]
    cls_str = os.environ["DYN_KVBM_CONNECTOR_OFFLOAD_FILTER_CLASS"]
    try:
        filter_module = importlib.import_module(module_str)
    except ImportError as e:
        raise ImportError(f"Failed to import offload filter module {module_str}") from e

    try:
        filter_class = getattr(filter_module, cls_str)
    except AttributeError as e:
        raise AttributeError(f"Failed to get offload filter class {cls_str} from module {module_str}") from e

    try:
        filter_instance = filter_class()
    except Exception as e:
        raise ValueError(f"Failed to instantiate offload filter class {cls_str} from module {module_str}") from e

    return filter_instance.should_offload
