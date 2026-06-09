# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo common utility submodules."""

from importlib import import_module
from types import ModuleType

# Keep this package lazy: restore-placeholder entrypoints import
# dynamo.common.utils.snapshot.restore_context before loading heavy backend/runtime
# modules, and eager utility imports can pull those dependencies in too early.

__all__ = [
    "endpoint_types",
    "engine_response",
    "namespace",
    "nvtx_utils",
    "time_section",
    "paths",
    "prometheus",
    "runtime",
]


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
