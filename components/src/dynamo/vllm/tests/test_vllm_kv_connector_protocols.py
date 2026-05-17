# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`dynamo.vllm.kv_connector_protocols`.

Pins the wire shape of ``kv_transfer_params`` on both sides of the
prefill/decode boundary for each supported KV connector, plus the
factory's dispatch and fallback behavior. No real vllm engine is
required — vllm_config is a SimpleNamespace and Mooncake's bootstrap
helper is monkey-patched.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Optional

import pytest

from dynamo.vllm.kv_connector_protocols import (
    KV_CONNECTOR_PROTOCOLS,
    KvConnectorProtocol,
    MooncakeConnectorProtocol,
    NixlConnectorProtocol,
    make_kv_connector_protocol,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

_MOONCAKE_MOD = (
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector"
)


def _config(connector: Optional[str], **kv_extra) -> SimpleNamespace:
    if connector is None:
        return SimpleNamespace(kv_transfer_config=None)
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector=connector, **kv_extra)
    )


def _install_fake_mooncake(monkeypatch, host: str, port: int) -> None:
    """Install a fake leaf module exposing ``get_mooncake_bootstrap_addr``.

    Mooncake's real module pulls in heavy native deps; the protocol only
    cares about the helper's return shape, so we stub the leaf and
    materialize empty parent packages so the dotted import resolves.
    ``monkeypatch.setitem`` auto-restores ``sys.modules`` at test
    teardown — no cross-test leakage.
    """
    stub = ModuleType(_MOONCAKE_MOD)
    stub.get_mooncake_bootstrap_addr = lambda _cfg: (host, port)  # type: ignore[attr-defined]
    parts = _MOONCAKE_MOD.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        monkeypatch.setitem(
            sys.modules, parent, sys.modules.get(parent, ModuleType(parent))
        )
    monkeypatch.setitem(sys.modules, _MOONCAKE_MOD, stub)


@pytest.fixture
def fake_mooncake(monkeypatch):
    """Default fake bootstrap helper returning a fixed (host, port)."""
    _install_fake_mooncake(monkeypatch, "10.0.0.5", 8998)


# ---------------------------------------------------------------------------
# NixlConnectorProtocol
# ---------------------------------------------------------------------------


def test_nixl_prefill_request_shape():
    proto = NixlConnectorProtocol(_config("NixlConnector"))
    assert proto.prefill_request_kv_transfer_params() == {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }


def test_nixl_decode_passes_through_engine_response():
    """Decode params come straight off the engine response — NIXL is pull-based."""
    proto = NixlConnectorProtocol(_config("NixlConnector"))
    engine_payload = {"remote_engine_id": "eng-1", "remote_block_ids": [3, 4]}
    response = SimpleNamespace(kv_transfer_params=engine_payload)
    assert proto.decode_request_kv_transfer_params(response) is engine_payload


def test_nixl_decode_returns_none_when_engine_omits():
    """vLLM emits ``None`` on non-PD final responses; pass it through unchanged."""
    proto = NixlConnectorProtocol(_config("NixlConnector"))
    response = SimpleNamespace(kv_transfer_params=None)
    assert proto.decode_request_kv_transfer_params(response) is None


# ---------------------------------------------------------------------------
# MooncakeConnectorProtocol
# ---------------------------------------------------------------------------


def test_mooncake_prefill_request_shape(fake_mooncake):
    proto = MooncakeConnectorProtocol(_config("MooncakeConnector"))
    params = proto.prefill_request_kv_transfer_params()
    assert params["do_remote_decode"] is True
    assert params["do_remote_prefill"] is False
    assert isinstance(params["transfer_id"], str) and params["transfer_id"]
    # Mooncake side never asks vLLM to populate remote_* — those are
    # synthesized on the decode side from this prefill's bootstrap addr.
    assert set(params) == {"do_remote_decode", "do_remote_prefill", "transfer_id"}


def test_mooncake_transfer_id_persists_across_prefill_and_decode(fake_mooncake):
    cfg = _config("MooncakeConnector", engine_id="eng-xyz")
    proto = MooncakeConnectorProtocol(cfg)
    prefill_tid = proto.prefill_request_kv_transfer_params()["transfer_id"]
    decode_tid = proto.decode_request_kv_transfer_params(
        SimpleNamespace(kv_transfer_params=None)
    )["transfer_id"]
    assert prefill_tid == decode_tid


def test_mooncake_distinct_instances_get_distinct_transfer_ids(fake_mooncake):
    """Each request gets its own UUID — collisions would cross-wire transfers."""
    cfg = _config("MooncakeConnector")
    a = MooncakeConnectorProtocol(cfg).prefill_request_kv_transfer_params()[
        "transfer_id"
    ]
    b = MooncakeConnectorProtocol(cfg).prefill_request_kv_transfer_params()[
        "transfer_id"
    ]
    assert a != b


def test_mooncake_decode_uses_vllm_bootstrap_helper_and_prefixes_http(monkeypatch):
    """Decode bootstrap addr comes from vLLM's helper; ``http://`` is
    non-optional because the decode side does ``addr + "/query"``."""
    _install_fake_mooncake(monkeypatch, "192.168.0.110", 8998)
    cfg = _config("MooncakeConnector", engine_id="eng-prefill-0")
    proto = MooncakeConnectorProtocol(cfg)
    params = proto.decode_request_kv_transfer_params(
        SimpleNamespace(kv_transfer_params=None)
    )
    assert params == {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "transfer_id": params["transfer_id"],
        "remote_bootstrap_addr": "http://192.168.0.110:8998",
        "remote_engine_id": "eng-prefill-0",
    }


def test_mooncake_decode_ignores_engine_kv_transfer_params(fake_mooncake):
    """Mooncake is push-based; whatever vLLM returns on
    ``res.kv_transfer_params`` must not leak into the decode-side payload."""
    cfg = _config("MooncakeConnector", engine_id="eng-1")
    proto = MooncakeConnectorProtocol(cfg)
    response = SimpleNamespace(
        kv_transfer_params={"remote_engine_id": "WRONG", "remote_host": "decoy"}
    )
    params = proto.decode_request_kv_transfer_params(response)
    assert params["remote_engine_id"] == "eng-1"
    assert "remote_host" not in params


def test_mooncake_init_raises_clear_error_when_vllm_mooncake_unavailable(monkeypatch):
    """Missing vLLM Mooncake support must surface at protocol construction
    (request setup), not after the prefill has been submitted to vLLM."""
    parts = _MOONCAKE_MOD.split(".")
    # Evict the leaf and every ancestor so the import unambiguously fails.
    for i in range(len(parts), 0, -1):
        monkeypatch.delitem(sys.modules, ".".join(parts[:i]), raising=False)

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == _MOONCAKE_MOD or name.startswith(_MOONCAKE_MOD + "."):
            raise ImportError(f"No module named {_MOONCAKE_MOD!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(RuntimeError, match="MooncakeConnector PD requires vLLM"):
        MooncakeConnectorProtocol(_config("MooncakeConnector"))


def test_mooncake_decode_does_not_reimport_per_call(fake_mooncake):
    """Helper is resolved once in __init__ and cached on the instance, so
    repeated decode calls don't re-enter the import machinery."""
    cfg = _config("MooncakeConnector", engine_id="eng-1")
    proto = MooncakeConnectorProtocol(cfg)
    helper = proto._get_bootstrap_addr
    # Calling decode multiple times must not rebind the helper.
    for _ in range(3):
        proto.decode_request_kv_transfer_params(
            SimpleNamespace(kv_transfer_params=None)
        )
    assert proto._get_bootstrap_addr is helper


# ---------------------------------------------------------------------------
# Factory + registry
# ---------------------------------------------------------------------------


def test_make_kv_connector_protocol_dispatches_nixl():
    proto = make_kv_connector_protocol(_config("NixlConnector"))
    assert isinstance(proto, NixlConnectorProtocol)


def test_make_kv_connector_protocol_dispatches_mooncake(fake_mooncake):
    proto = make_kv_connector_protocol(_config("MooncakeConnector"))
    assert isinstance(proto, MooncakeConnectorProtocol)


def test_make_kv_connector_protocol_falls_back_to_nixl_for_missing_config():
    """No KVTransferConfig at all — preserve pre-existing (NIXL) behavior."""
    proto = make_kv_connector_protocol(SimpleNamespace())
    assert isinstance(proto, NixlConnectorProtocol)


def test_make_kv_connector_protocol_falls_back_to_nixl_when_config_is_none():
    proto = make_kv_connector_protocol(_config(None))
    assert isinstance(proto, NixlConnectorProtocol)


def test_make_kv_connector_protocol_raises_on_unknown_connector():
    """An unknown connector name must raise — a mismatch between the dynamo
    PD protocol and the vLLM engine's connector is a misconfiguration, not a
    benign default. Silently falling back to NIXL would emit the wrong wire
    shape and surface as opaque decode failures."""
    with pytest.raises(ValueError, match="HypotheticalFutureConnector") as exc:
        make_kv_connector_protocol(_config("HypotheticalFutureConnector"))
    msg = str(exc.value)
    # The error should help an operator self-diagnose.
    assert "MooncakeConnector" in msg and "NixlConnector" in msg


def test_registry_keys_match_vllm_connector_names():
    """Wire-format guard: KV_CONNECTOR_PROTOCOLS keys must match the strings
    vLLM uses in ``KVTransferConfig.kv_connector``."""
    assert set(KV_CONNECTOR_PROTOCOLS) == {"NixlConnector", "MooncakeConnector"}
    for cls in KV_CONNECTOR_PROTOCOLS.values():
        assert issubclass(cls, KvConnectorProtocol)


# ---------------------------------------------------------------------------
# Real-import contract test
#
# When the real vLLM Mooncake module is importable, verify the symbol the
# protocol actually depends on still exists with the expected signature.
# This catches upstream path / signature drift that the sys.modules-stubbed
# tests above would silently pass through.
# ---------------------------------------------------------------------------


_real_mooncake_spec = None
try:  # pragma: no cover - environment-dependent
    _real_mooncake_spec = importlib.util.find_spec(_MOONCAKE_MOD)
except (ImportError, ValueError):
    _real_mooncake_spec = None


@pytest.mark.skipif(
    _real_mooncake_spec is None,
    reason="vLLM Mooncake connector not installed in this environment",
)
def test_real_vllm_mooncake_helper_signature_is_compatible():
    """Contract test: when vLLM Mooncake is present, the helper the protocol
    imports must exist and be callable with a single vllm_config argument."""
    import inspect

    real = importlib.import_module(_MOONCAKE_MOD)
    fn = getattr(real, "get_mooncake_bootstrap_addr", None)
    assert callable(fn), (
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector."
        "get_mooncake_bootstrap_addr is missing — this PR will fail at runtime."
    )
    sig = inspect.signature(fn)
    # Must accept exactly one positional argument (the vllm_config).
    params = [p for p in sig.parameters.values() if p.kind != p.VAR_KEYWORD]
    assert len(params) >= 1, (
        f"get_mooncake_bootstrap_addr signature changed: expected at least 1 "
        f"positional arg, got {sig}"
    )
