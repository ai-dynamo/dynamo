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

import pytest
import pytest_asyncio

from dynamo.common.http import close_http_client

# The decode carriers ``codec_errors`` probes. Only these names are answered by
# the ``carrier_imports`` fixture; every other import runs for real.
_MEDIA_CARRIERS = ("cv2", "av", "decord")

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


def can_import_deps() -> bool:
    return _can_import_torch() and _can_import_pil()


def pytest_ignore_collect(collection_path, config) -> bool | None:
    """Skip collecting multimodal test files when optional deps are missing.

    ``pytest_ignore_collect`` is a firstresult hook: returning ``False`` vetoes
    all other implementations, including pytest's own ``--ignore-glob`` and
    ``norecursedirs`` handling. Return ``None`` when this hook has no opinion.
    """
    filename = collection_path.name
    if filename.startswith("test_") and not can_import_deps():
        return True
    return None


@pytest.fixture
def carrier_imports(monkeypatch):
    """Mock media-carrier imports while leaving unrelated imports intact."""
    # Import after collection's optional-dependency guard: the multimodal
    # package imports torch, which CPU-only runtime images may not ship.
    from dynamo.common.multimodal import codec_errors

    def _set(
        present: tuple[str, ...] = (), error: str | BaseException | None = None
    ) -> None:
        real = importlib.import_module

        def _fake(name: str, *args, **kwargs):
            if name in _MEDIA_CARRIERS:
                if name in present:
                    return object()
                if isinstance(error, BaseException):
                    raise error
                raise ImportError(error or f"No module named '{name}'")
            return real(name, *args, **kwargs)

        monkeypatch.setattr(codec_errors.importlib, "import_module", _fake)

    return _set


@pytest_asyncio.fixture(autouse=True)
async def _close_shared_http_client():
    yield
    await close_http_client()
