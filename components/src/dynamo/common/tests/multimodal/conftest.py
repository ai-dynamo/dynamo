# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for multimodal tests.

Autouse fixture ensures the shared HTTP client singleton is closed after
each test so ``DYN_HTTP_BACKEND`` changes take effect between runs and
no "Unclosed client session" warning bleeds across tests.

Handles conditional test collection to prevent import errors when optional
deps (torch / Pillow) are not installed in the current environment.
"""

from __future__ import annotations

import importlib

import pytest_asyncio

from dynamo.common.http import close_http_client

# Cached results of probing optional deps used by multimodal unit tests.
# `None` = not attempted, `True` = importable, `False` = raised.
_torch_importable: bool | None = None
_pil_importable: bool | None = None


def _can_import_torch() -> bool:
    """Import torch once and cache the result (multimodal eagerly imports it)."""
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


def _can_import_pil() -> bool:
    """Import Pillow once and cache the result (multimodal tests import it)."""
    global _pil_importable
    if _pil_importable is None:
        try:
            pil = importlib.import_module("PIL")
            _pil_importable = (
                pil.__spec__ is not None and pil.__spec__.loader is not None
            )
        except ImportError:
            _pil_importable = False
    return _pil_importable


def pytest_ignore_collect(collection_path, config):
    """Skip collecting multimodal test files when optional deps are missing."""
    filename = collection_path.name
    if filename.startswith("test_") and (
        not _can_import_torch() or not _can_import_pil()
    ):
        return True
    return None


@pytest_asyncio.fixture(autouse=True)
async def _close_shared_http_client():
    yield
    await close_http_client()
