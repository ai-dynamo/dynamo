# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify dynamo.nixl_connect imports on hosts without NIXL bindings.

The NIXL pip wheel ships CUDA only (`nixl-cu12`). On platforms without a
NIXL wheel (e.g. AMD ROCm hosts) the module must still import so that
transitive importers — router, planner, frontend, AMD aggregated /
Mooncake-based disaggregated paths — load. The deferred ImportError
should be raised only when a code path that actually touches the
bindings is exercised (today: constructing a `Connector` → `Connection`).
"""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

# Pure Python import-error logic; no GPU needed. The `gpu_0` marker is
# required by .ai/pytest-guidelines.md so CI matrices that filter by
# GPU class (`-m gpu_0`) pick it up.
pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.fixture(scope="module")
def nixl_connect_no_nixl():
    """Re-import dynamo.nixl_connect exactly once with nixl masked.

    Uses `patch.dict(sys.modules, ...)` (auto-restoring) to mask nixl —
    same pattern as `test_nixl_connect_unit.py`. Setting
    `sys.modules[name] = None` causes `import name` to raise
    ModuleNotFoundError, matching the real AMD-host behavior.

    Scope is `module` (not `function`) on purpose: re-executing the
    `dynamo.nixl_connect` module body twice in a single session triggers
    a `_has_torch_function already has a docstring` RuntimeError from
    `torch.overrides`. One re-import per file is safe; per-test re-import
    is not. Per-test state is handled via `monkeypatch.setattr` below.
    """
    saved = sys.modules.get("dynamo.nixl_connect")
    with patch.dict(
        sys.modules,
        {"nixl": None, "nixl._api": None, "nixl._bindings": None},
    ):
        sys.modules.pop("dynamo.nixl_connect", None)
        yield importlib.import_module("dynamo.nixl_connect")
    if saved is None:
        sys.modules.pop("dynamo.nixl_connect", None)
    else:
        sys.modules["dynamo.nixl_connect"] = saved


def test_module_imports_without_nixl(nixl_connect_no_nixl):
    """`import dynamo.nixl_connect` must succeed when nixl is unavailable."""
    mod = nixl_connect_no_nixl
    # `nixl_api` / `nixl_bindings` are replaced with proxies, not None,
    # so any attribute access or call surfaces the deferred ImportError
    # with a clear message rather than `AttributeError: 'NoneType'...`.
    assert isinstance(mod.nixl_api, mod._NixlImportProxy)
    assert isinstance(mod.nixl_bindings, mod._NixlImportProxy)


def test_proxy_raises_clear_error_on_attribute_access(nixl_connect_no_nixl):
    """Any attribute access on the proxy raises the deferred ImportError."""
    mod = nixl_connect_no_nixl
    # Covers `nixl_api.nixl_agent`, `nixl_bindings.nixlXferDList`, and
    # any future use site.
    with pytest.raises(ImportError, match="NIXL Python bindings must be installed"):
        _ = mod.nixl_api.nixl_agent
    with pytest.raises(ImportError, match="NIXL Python bindings must be installed"):
        _ = mod.nixl_bindings.nixlXferDList


def test_proxy_raises_clear_error_on_call(nixl_connect_no_nixl):
    """Calling the proxy itself (e.g. `nixl_api(...)`) raises the deferred error."""
    mod = nixl_connect_no_nixl
    with pytest.raises(ImportError, match="NIXL Python bindings must be installed"):
        mod.nixl_api()


def test_proxy_chains_original_import_error(nixl_connect_no_nixl):
    """The re-raised ImportError preserves the original cause via __cause__."""
    mod = nixl_connect_no_nixl
    with pytest.raises(ImportError) as excinfo:
        _ = mod.nixl_api.anything
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_connection_construction_raises_without_nixl(nixl_connect_no_nixl):
    """Constructing a Connection without NIXL must raise the deferred error.

    The runtime use site is `nixl_api.nixl_agent(self._name)` in
    `Connection.__init__`. With the proxy in place no explicit guard is
    needed — the call itself raises.
    """
    mod = nixl_connect_no_nixl
    fake_connector = MagicMock(spec=mod.Connector)
    fake_connector.name = "test"
    with pytest.raises(ImportError, match="NIXL Python bindings must be installed"):
        mod.Connection(fake_connector, 1)


def test_real_nixl_path_unchanged(nixl_connect_no_nixl, monkeypatch):
    """When the bindings are present, attribute access goes to the real module.

    Simulates the CUDA path by monkey-patching the imported module's
    `nixl_api` to a Mock and verifying ordinary attribute access works.
    """
    mod = nixl_connect_no_nixl
    real_like = MagicMock()
    real_like.nixl_agent.return_value = MagicMock(name="agent")
    monkeypatch.setattr(mod, "nixl_api", real_like)
    # Attribute access + call now work normally.
    agent = mod.nixl_api.nixl_agent("name")
    assert agent is real_like.nixl_agent.return_value
