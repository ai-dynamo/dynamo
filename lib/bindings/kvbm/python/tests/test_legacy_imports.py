# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy import-path back-compat regression tests (Phase 1)."""

import importlib
import importlib.util
import subprocess
import sys

import pytest

VLLM = importlib.util.find_spec("vllm") is not None
TRTLLM = importlib.util.find_spec("tensorrt_llm") is not None
requires_vllm = pytest.mark.skipif(not VLLM, reason="requires vllm")
requires_trtllm = pytest.mark.skipif(not TRTLLM, reason="requires tensorrt_llm")


def test_legacy_connector_importable_without_forcing_vllm():
    """Importing kvbm.vllm_integration.connector must not transitively import vllm.

    Run in a fresh subprocess so this is robust to other tests in the session
    that may have already imported vllm.
    """
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
def test_legacy_connector_resolves_to_v1():
    mod = importlib.import_module("kvbm.vllm_integration.connector")
    cls = mod.DynamoConnector
    from kvbm.v1.vllm_integration.connector.dynamo_connector import (
        DynamoConnector as Canonical,
    )

    assert cls is Canonical
    assert cls.__module__ == "kvbm.v1.vllm_integration.connector.dynamo_connector"


@requires_vllm
def test_legacy_connector_secondary_exports():
    mod = importlib.import_module("kvbm.vllm_integration.connector")
    for name in ("DynamoConnectorMetadata", "PdConnector", "PdConnectorMetadata"):
        obj = getattr(mod, name)
        assert obj.__module__.startswith("kvbm.v1.vllm_integration.connector.")


@requires_vllm
def test_legacy_consolidator_config_shim():
    from kvbm.v1.vllm_integration.consolidator_config import (
        get_consolidator_endpoints as canonical,
    )
    from kvbm.vllm_integration.consolidator_config import get_consolidator_endpoints

    assert get_consolidator_endpoints is canonical


@requires_trtllm
def test_legacy_trtllm_connector_resolves_to_v1():
    from kvbm.trtllm_integration.connector import (
        DynamoKVBMConnectorLeader,
        DynamoKVBMConnectorWorker,
    )

    assert DynamoKVBMConnectorLeader.__module__.startswith("kvbm.v1.trtllm_integration")
    assert DynamoKVBMConnectorWorker.__module__.startswith("kvbm.v1.trtllm_integration")


def test_legacy_utils_nvtx_annotate_resolves_to_v1():
    from kvbm.utils import nvtx_annotate
    from kvbm.v1.utils import nvtx_annotate as canonical

    assert nvtx_annotate is canonical


def test_top_level_kvbm_has_v1_and_v2():
    import kvbm

    assert kvbm._V1_AVAILABLE and kvbm._V2_AVAILABLE


# ---------------------------------------------------------------------------
# Phase 4: canonical kvbm.v{1,2}.vllm.connector façades (1↔2 char mirror).
# ---------------------------------------------------------------------------


def test_v1_canonical_vllm_connector_lazy_no_vllm():
    """Importing kvbm.v1.vllm.connector must not transitively import vllm.

    Same subprocess pattern as test_legacy_connector_importable_without_forcing_vllm.
    """
    script = (
        "import sys, importlib;"
        "importlib.import_module('kvbm.v1.vllm.connector');"
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
def test_v1_canonical_vllm_connector_resolves_to_v1_impl():
    mod = importlib.import_module("kvbm.v1.vllm.connector")
    cls = mod.DynamoConnector
    from kvbm.v1.vllm_integration.connector.dynamo_connector import (
        DynamoConnector as Canonical,
    )

    assert cls is Canonical
    assert cls.__module__ == "kvbm.v1.vllm_integration.connector.dynamo_connector"


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
def test_v2_canonical_vllm_connector_resolves_to_schedulers_impl():
    mod = importlib.import_module("kvbm.v2.vllm.connector")
    cls = mod.DynamoConnector
    from kvbm.v2.vllm.schedulers.connector import DynamoConnector as Canonical

    assert cls is Canonical
    assert cls.__module__ == "kvbm.v2.vllm.schedulers.connector"
