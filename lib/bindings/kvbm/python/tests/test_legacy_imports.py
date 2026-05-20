# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy import-path back-compat regression tests."""

import importlib
import importlib.util
import subprocess
import sys

import pytest

VLLM = importlib.util.find_spec("vllm") is not None
NIXL = importlib.util.find_spec("nixl") is not None
requires_vllm = pytest.mark.skipif(not VLLM, reason="requires vllm")
requires_kvbm_runtime = pytest.mark.skipif(
    not NIXL, reason="requires kvbm runtime dependencies"
)


@requires_kvbm_runtime
def test_legacy_connector_importable_without_forcing_vllm():
    """The legacy connector module stays lazy until vLLM resolves a class."""
    script = (
        "import sys, importlib;"
        "importlib.import_module('kvbm.vllm_integration.connector');"
        "leaked = sorted(m for m in sys.modules if m == 'vllm' or m.startswith('vllm.'));"
        "assert not leaked, f'vllm leaked into sys.modules: {leaked}';"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "ok"


@requires_vllm
@requires_kvbm_runtime
def test_legacy_connector_resolves_to_v2():
    mod = importlib.import_module("kvbm.vllm_integration.connector")
    cls = mod.DynamoConnector
    from kvbm.v2.vllm.schedulers.connector import DynamoConnector as Canonical

    assert cls is Canonical
    assert cls.__module__ == "kvbm.v2.vllm.schedulers.connector"


@requires_vllm
@requires_kvbm_runtime
def test_legacy_connector_secondary_exports():
    mod = importlib.import_module("kvbm.vllm_integration.connector")
    assert mod.DynamoConnectorMetadata.__module__ == (
        "kvbm.v2.vllm.schedulers.connector"
    )
    assert mod.PdConnector.__module__ == (
        "kvbm.vllm_integration.connector.pd_connector"
    )
    assert mod.PdConnectorMetadata.__module__ == (
        "kvbm.vllm_integration.connector.pd_connector"
    )


@requires_vllm
@requires_kvbm_runtime
def test_legacy_consolidator_config_shim():
    """The legacy shim at ``kvbm.vllm_integration.consolidator_config`` resolves to
    ``kvbm.v2.vllm_integration.consolidator_config``. v1 is dead.

    The signature, return shape (Optional[Tuple[str, str, str]]), env-var
    contract (DYN_KVBM_KV_EVENTS_ENABLE_CONSOLIDATOR opt-out, default-on),
    and port-derivation (DYN_KVBM_LEADER_ZMQ_PUB_PORT + 1000) are
    unchanged — only the implementation moved.
    """
    from kvbm.v2.vllm_integration.consolidator_config import (
        get_consolidator_endpoints as canonical,
    )
    from kvbm.vllm_integration.consolidator_config import get_consolidator_endpoints

    assert get_consolidator_endpoints is canonical
    assert get_consolidator_endpoints.__module__ == (
        "kvbm.v2.vllm_integration.consolidator_config"
    )


@requires_kvbm_runtime
def test_top_level_kvbm_has_v2():
    import kvbm

    assert kvbm._V2_AVAILABLE


@requires_kvbm_runtime
def test_v2_canonical_vllm_connector_lazy_no_vllm():
    """Importing kvbm.v2.vllm.connector must not transitively import vllm.

    The lazy __getattr__ shim defers the schedulers chain (which imports
    vllm.distributed.kv_transfer.*) until vllm itself asks for
    DynamoConnector.

    Note: kvbm.v2.vllm.__init__ does call version_check() which touches
    vllm.version. That submodule is allowed; the assertion below tolerates
    it but blocks anything heavier.
    """
    script = (
        "import sys, importlib;"
        "importlib.import_module('kvbm.v2.vllm.connector');"
        "leaked = sorted("
        "m for m in sys.modules "
        "if (m == 'vllm' or m.startswith('vllm.')) and m != 'vllm.version'"
        ");"
        "assert not leaked, f'vllm leaked into sys.modules: {leaked}';"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "ok"


@requires_vllm
@requires_kvbm_runtime
def test_v2_canonical_vllm_connector_resolves_to_schedulers_impl():
    mod = importlib.import_module("kvbm.v2.vllm.connector")
    cls = mod.DynamoConnector
    from kvbm.v2.vllm.schedulers.connector import DynamoConnector as Canonical

    assert cls is Canonical
    assert cls.__module__ == "kvbm.v2.vllm.schedulers.connector"


@requires_vllm
@requires_kvbm_runtime
def test_v2_canonical_vllm_connector_exports_pdconnector():
    mod = importlib.import_module("kvbm.v2.vllm.connector")

    assert mod.PdConnector.__module__ == (
        "kvbm.vllm_integration.connector.pd_connector"
    )
    assert mod.PdConnectorMetadata.__module__ == (
        "kvbm.vllm_integration.connector.pd_connector"
    )
