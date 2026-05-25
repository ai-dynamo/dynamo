# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify dynamo.nixl_connect imports on hosts without NIXL bindings.

The NIXL pip wheel ships CUDA only (`nixl-cu12`). On platforms without a
NIXL wheel (e.g. AMD ROCm hosts) the module must still import so that
transitive importers — router, planner, frontend, AMD aggregated /
Mooncake-based disaggregated paths — load. The deferred ImportError
should be raised only when an operation that actually needs NIXL is
attempted (e.g. constructing a `Connector` -> `Connection`).
"""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge]


def _reimport_without_nixl():
    """Re-import dynamo.nixl_connect with nixl.* masked out of sys.modules."""
    for mod in list(sys.modules):
        if mod == "dynamo.nixl_connect" or mod.startswith("nixl"):
            del sys.modules[mod]

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "nixl" or name.startswith("nixl."):
            raise ImportError(f"No module named '{name}' (simulated)")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        return importlib.import_module("dynamo.nixl_connect")


def test_module_imports_without_nixl():
    """`import dynamo.nixl_connect` must succeed when nixl is unavailable."""
    mod = _reimport_without_nixl()
    assert mod.nixl_api is None
    assert mod.nixl_bindings is None
    assert mod._NIXL_IMPORT_ERROR is not None
    assert isinstance(mod._NIXL_IMPORT_ERROR, ImportError)


def test_require_nixl_raises_when_missing():
    """`_require_nixl()` re-raises the deferred ImportError with original cause."""
    mod = _reimport_without_nixl()
    with pytest.raises(ImportError) as excinfo:
        mod._require_nixl()
    assert "NIXL Python bindings must be installed" in str(excinfo.value)
    assert excinfo.value.__cause__ is mod._NIXL_IMPORT_ERROR


def test_connection_construction_raises_without_nixl():
    """Constructing a Connection without NIXL must raise the deferred error."""
    mod = _reimport_without_nixl()
    fake_connector = MagicMock(spec=mod.Connector)
    fake_connector.name = "test"
    with pytest.raises(ImportError, match="NIXL Python bindings must be installed"):
        mod.Connection(fake_connector, 1)


def test_module_imports_with_nixl_present():
    """Sanity check: when nixl is available, the module exposes the real bindings.

    Skipped automatically when the real nixl wheel isn't installed.
    """
    try:
        import nixl._api  # noqa: F401
    except ImportError:
        pytest.skip("real nixl wheel not installed; covered on CUDA CI")

    for cached in list(sys.modules):
        if cached == "dynamo.nixl_connect":
            del sys.modules[cached]
    mod = importlib.import_module("dynamo.nixl_connect")
    assert mod.nixl_api is not None
    assert mod.nixl_bindings is not None
    assert mod._NIXL_IMPORT_ERROR is None
