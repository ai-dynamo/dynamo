# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conftest for dynamo.common memory unit tests.

Handles conditional test collection to prevent import errors when torch is
not installed in the current environment.
"""

from __future__ import annotations

import importlib

# Cached result of probing torch.
# `None` = not attempted, `True` = importable, `False` = raised.
_torch_importable: bool | None = None


def _can_import_torch() -> bool:
    """Import torch once and cache the result."""
    global _torch_importable
    if _torch_importable is None:
        try:
            torch = importlib.import_module("torch")
            _torch_importable = (
                torch.__spec__ is not None and torch.__spec__.loader is not None
            )
        except ImportError:
            _torch_importable = False
    return _torch_importable


def pytest_ignore_collect(collection_path, config):
    """Skip collecting memory test files if torch isn't installed."""
    filename = collection_path.name
    if filename.startswith("test_") and not _can_import_torch():
        return True
    return None
