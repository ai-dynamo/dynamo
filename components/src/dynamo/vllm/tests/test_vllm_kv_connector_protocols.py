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
    LMCacheMPConnectorProtocol,
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


def _bidir_config(enabled):
    """Build a NIXL config with an explicit bidirectional-transfer flag."""
    config = _config("NixlConnector")
    config.kv_transfer_config.get_from_extra_config = (
        lambda key, default: enabled if key == "bidirectional_kv_xfer" else default
    )
    return config


def test_nixl_prefill_preserves_native_decoder_coordinates_without_mutation():
    """Preserve opaque coordinates without aliasing nested caller data."""
    incoming = {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "remote_engine_id": "decoder-engine",
        "remote_request_id": "previous-turn",
        "remote_block_ids": [[0, 2], [4]],
        "remote_num_tokens": 128,
        "remote_blocks_expiry_time": 12345.25,
        "native_extension": {"future": [1]},
    }
    params = NixlConnectorProtocol(
        _bidir_config(True)
    ).prefill_request_kv_transfer_params(incoming)
    assert params["do_remote_decode"] is True
    assert params["do_remote_prefill"] is False
    for key in incoming.keys() - {"do_remote_decode", "do_remote_prefill"}:
        assert params[key] == incoming[key]
    params["remote_block_ids"][0].append(9)
    params["native_extension"]["future"].append(2)
    assert incoming["remote_block_ids"] == [[0, 2], [4]]
    assert incoming["native_extension"] == {"future": [1]}
    assert incoming["do_remote_decode"] is False
    assert incoming["do_remote_prefill"] is True


@pytest.mark.parametrize(
    "incoming",
    [
        None,
        {},
        {"remote_block_ids": None, "remote_num_tokens": 0},
        {"remote_block_ids": [[], []], "remote_num_tokens": 0},
    ],
)
def test_nixl_native_cold_request_does_not_require_bidirectional_config(incoming):
    """Allow cold requests without bidirectional transfer."""
    params = NixlConnectorProtocol(
        _bidir_config(False)
    ).prefill_request_kv_transfer_params(incoming)
    assert params["do_remote_decode"] is True
    assert params["do_remote_prefill"] is False
    assert params["remote_engine_id"] is None


@pytest.mark.parametrize("config", [_bidir_config(False), _config(None)])
def test_nixl_native_decoder_source_requires_bidirectional_config(config):
    """Require explicit bidirectional opt-in for retained decoder blocks."""
    with pytest.raises(ValueError, match="bidirectional_kv_xfer"):
        NixlConnectorProtocol(config).prefill_request_kv_transfer_params(
            {"remote_block_ids": [[0]], "remote_num_tokens": 64}
        )


@pytest.mark.parametrize("blocks", [[[0]], [[], []]])
@pytest.mark.parametrize("count", ["128", "", 1.5, -1, True, False, [], {}])
def test_nixl_rejects_invalid_native_token_counts(blocks, count):
    """Reject unsafe scheduler scalars even when cache groups are empty."""
    with pytest.raises(ValueError, match="remote_num_tokens"):
        NixlConnectorProtocol(_bidir_config(True)).prefill_request_kv_transfer_params(
            {"remote_block_ids": blocks, "remote_num_tokens": count}
        )


@pytest.mark.parametrize(
    "token_params",
    [
        {},
        {"remote_num_tokens": None},
        {"remote_num_tokens": 0},
        {"remote_num_tokens": 128},
    ],
)
def test_nixl_preserves_valid_native_token_counts(token_params):
    """Keep optional and non-negative integer counts unchanged."""
    params = NixlConnectorProtocol(
        _bidir_config(True)
    ).prefill_request_kv_transfer_params({"remote_block_ids": [[0]], **token_params})
    assert ("remote_num_tokens" in params) == ("remote_num_tokens" in token_params)
    assert params.get("remote_num_tokens") == token_params.get("remote_num_tokens")


@pytest.mark.parametrize(
    "blocks",
    [
        "blocks",
        "",
        b"blocks",
        {"group": [0]},
        {},
        1,
        1.5,
        True,
        False,
        [0],
        [None],
        ["bad"],
        [{}],
        [[-1]],
        [[True]],
        [[1.5]],
        [["0"]],
    ],
)
def test_nixl_rejects_malformed_native_block_ids(blocks):
    """Reject malformed groups and IDs with a field-specific error."""
    with pytest.raises((TypeError, ValueError), match="remote_block_ids"):
        NixlConnectorProtocol(_bidir_config(True)).prefill_request_kv_transfer_params(
            {"remote_block_ids": blocks, "remote_num_tokens": 128}
        )


@pytest.mark.parametrize(
    "blocks,has_blocks",
    [
        (None, False),
        ([], False),
        ((), False),
        ([[], []], False),
        (((), []), False),
        ([[0]], True),
        (([0], ()), True),
        (((0,), [1, 2]), True),
    ],
)
def test_nixl_preserves_supported_native_block_groups(blocks, has_blocks):
    """Preserve group forms and require opt-in only for actual blocks."""
    incoming = {"remote_block_ids": blocks}
    protocol = NixlConnectorProtocol(_bidir_config(has_blocks))
    params = protocol.prefill_request_kv_transfer_params(incoming)
    assert params["remote_block_ids"] == blocks
    if has_blocks:
        disabled = NixlConnectorProtocol(_bidir_config(False))
        with pytest.raises(ValueError, match="bidirectional_kv_xfer"):
            disabled.prefill_request_kv_transfer_params(incoming)


@pytest.mark.parametrize(
    "field,value",
    [
        ("remote_engine_id", []),
        ("remote_engine_id", {}),
        ("remote_engine_id", 1),
        ("remote_engine_id", True),
        ("remote_registration_generation", None),
        ("remote_registration_generation", "1"),
        ("remote_registration_generation", []),
        ("remote_registration_generation", -1),
        ("remote_registration_generation", True),
        ("remote_registration_generation", 1.5),
        ("remote_blocks_expiry_time", "123"),
        ("remote_blocks_expiry_time", {}),
        ("remote_blocks_expiry_time", True),
        ("remote_blocks_expiry_time", float("nan")),
        ("remote_blocks_expiry_time", float("inf")),
        ("remote_blocks_expiry_time", 10**400),
    ],
)
def test_nixl_rejects_invalid_native_receive_scalars(field, value):
    """Reject invalid worker metadata before it reaches receive hooks."""
    protocol = NixlConnectorProtocol(_bidir_config(True))
    with pytest.raises(ValueError, match=field):
        protocol.prefill_request_kv_transfer_params(
            {"remote_block_ids": [[0]], field: value}
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("remote_host", ""),
        ("remote_host", []),
        ("remote_host", {}),
        ("remote_host", 1),
        ("remote_host", True),
        ("remote_request_id", []),
        ("remote_request_id", {}),
        ("remote_request_id", 1),
        ("remote_request_id", True),
        ("remote_port", 0),
        ("remote_port", -1),
        ("remote_port", 65536),
        ("remote_port", "123"),
        ("remote_port", 1.5),
        ("remote_port", True),
        ("remote_port", False),
        ("remote_port", []),
        ("remote_port", {}),
    ]
    + [
        (field, value)
        for field in ("tp_size", "dcp_size", "pp_size")
        for value in (0, -1, "2", 1.5, True, None)
    ],
)
def test_nixl_rejects_invalid_native_endpoint_and_topology_scalars(field, value):
    """Reject malformed handshake and notification values at ingress."""
    with pytest.raises(ValueError, match=field):
        NixlConnectorProtocol(_bidir_config(True)).prefill_request_kv_transfer_params(
            {"remote_block_ids": [[0]], field: value}
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("remote_host", ""),
        ("remote_port", "123"),
        ("tp_size", None),
        ("_p_side_truncated", True),
    ],
)
def test_nixl_rejects_invalid_cold_native_handshake_scalars(field, value):
    """Validate supplied metadata even without retained decoder blocks."""
    with pytest.raises(ValueError, match=field):
        NixlConnectorProtocol(_bidir_config(False)).prefill_request_kv_transfer_params(
            {"remote_block_ids": [[], []], field: value}
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {
            "remote_host": None,
            "remote_port": None,
            "remote_request_id": None,
            "remote_block_size": None,
        },
        {"remote_host": "::1"},
        {"remote_host": "peer.example"},
        {"remote_port": 1},
        {"remote_port": 65535},
        {"remote_request_id": "previous-turn"},
        {"tp_size": 1, "dcp_size": 2, "pp_size": 8},
        {"tp_size": 3, "dcp_size": 5, "pp_size": 7, "remote_block_size": 17},
        {"remote_block_size": 1},
        {"remote_block_size": 128},
    ],
)
@pytest.mark.parametrize("blocks", [[[0]], [[], []]])
def test_nixl_preserves_partial_native_endpoint_and_topology_scalars(metadata, blocks):
    """Preserve partial coordinates without imposing topology or DNS policy."""
    incoming = {"remote_block_ids": blocks, **metadata}
    params = NixlConnectorProtocol(
        _bidir_config(True)
    ).prefill_request_kv_transfer_params(incoming)
    for field, value in incoming.items():
        assert params[field] == value
    for field in ("tp_size", "dcp_size", "pp_size", "remote_block_size"):
        assert (field in params) == (field in incoming)


@pytest.mark.parametrize("value", [True, False, None])
def test_nixl_rejects_caller_supplied_truncation_marker(value):
    """Keep scheduler-owned backoff state out of native coordinates."""
    with pytest.raises(ValueError, match="_p_side_truncated"):
        NixlConnectorProtocol(_bidir_config(True)).prefill_request_kv_transfer_params(
            {"remote_block_ids": [[0]], "_p_side_truncated": value}
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"remote_engine_id": None},
        {"remote_engine_id": "peer"},
        {"remote_registration_generation": 0},
        {"remote_registration_generation": 3},
        {"remote_blocks_expiry_time": None},
        {"remote_blocks_expiry_time": -1.0},
        {"remote_blocks_expiry_time": 123},
        {"remote_blocks_expiry_time": 123.5},
    ],
)
def test_nixl_preserves_valid_native_receive_scalars(metadata):
    """Preserve optional fields and leave expiry decisions to vLLM."""
    protocol = NixlConnectorProtocol(_bidir_config(True))
    params = protocol.prefill_request_kv_transfer_params(
        {"remote_block_ids": [[0]], **metadata}
    )
    for field in (
        "remote_registration_generation",
        "remote_blocks_expiry_time",
    ):
        assert (field in params) == (field in metadata)
    for field, value in metadata.items():
        assert params[field] == value


@pytest.mark.parametrize("value", [True, False, None])
def test_nixl_rejects_caller_supplied_receive_marker(value):
    """Prevent callers from suppressing vLLM's receive registration."""
    protocol = NixlConnectorProtocol(_bidir_config(True))
    with pytest.raises(ValueError, match="_remote_blocks_processed"):
        protocol.prefill_request_kv_transfer_params(
            {"remote_block_ids": [[0]], "_remote_blocks_processed": value}
        )


def test_native_decoder_source_rejected_for_cache_mediated_connector():
    """Reject decoder coordinates for cache-mediated coordination."""
    protocol = LMCacheMPConnectorProtocol(_config("LMCacheMPConnector"))
    with pytest.raises(ValueError, match="requires NixlConnector"):
        protocol.prefill_request_kv_transfer_params({"remote_block_ids": [[0]]})
    assert protocol.prefill_request_kv_transfer_params({"remote_block_ids": None}) == {}


def test_native_decoder_source_rejected_for_mooncake_connector(fake_mooncake):
    """Reject decoder coordinates for Mooncake push coordination."""
    protocol = MooncakeConnectorProtocol(_config("MooncakeConnector"))
    with pytest.raises(ValueError, match="requires NixlConnector"):
        protocol.prefill_request_kv_transfer_params({"remote_block_ids": [[0]]})
    assert (
        protocol.prefill_request_kv_transfer_params({"remote_block_ids": None})
        == protocol.prefill_request_kv_transfer_params()
    )


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
# LMCacheMPConnectorProtocol
# ---------------------------------------------------------------------------


def test_lmcache_mp_prefill_request_is_empty():
    proto = LMCacheMPConnectorProtocol(_config("LMCacheMPConnector"))
    assert proto.prefill_request_kv_transfer_params() == {}


def test_lmcache_mp_decode_returns_empty_and_ignores_engine_params():
    """Cache-mediated: nothing from the prefill response reaches decode,
    but the result is an empty dict (not None) so the prefill response
    keeps its disaggregated_params envelope for the prefill router."""
    proto = LMCacheMPConnectorProtocol(_config("LMCacheMPConnector"))
    response = SimpleNamespace(kv_transfer_params={"remote_engine_id": "decoy"})
    assert proto.decode_request_kv_transfer_params(response) == {}


# ---------------------------------------------------------------------------
# Factory + registry
# ---------------------------------------------------------------------------


def test_make_kv_connector_protocol_dispatches_lmcache_mp():
    proto = make_kv_connector_protocol(_config("LMCacheMPConnector"))
    assert isinstance(proto, LMCacheMPConnectorProtocol)


def test_multiconnector_prefers_transport_child_over_lmcache_mp():
    """A cache-mediated child must not make the wrapper ambiguous."""
    proto = make_kv_connector_protocol(
        _config(
            "MultiConnector",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "LMCacheMPConnector"},
                    {"kv_connector": "NixlConnector"},
                ]
            },
        )
    )
    assert isinstance(proto, NixlConnectorProtocol)


def test_multiconnector_dispatches_lone_lmcache_mp_child():
    proto = make_kv_connector_protocol(
        _config(
            "MultiConnector",
            kv_connector_extra_config={
                "connectors": [{"kv_connector": "LMCacheMPConnector"}]
            },
        )
    )
    assert isinstance(proto, LMCacheMPConnectorProtocol)


def test_make_kv_connector_protocol_dispatches_nixl():
    proto = make_kv_connector_protocol(_config("NixlConnector"))
    assert isinstance(proto, NixlConnectorProtocol)


def test_make_kv_connector_protocol_dispatches_mooncake(fake_mooncake):
    proto = make_kv_connector_protocol(_config("MooncakeConnector"))
    assert isinstance(proto, MooncakeConnectorProtocol)


def test_make_kv_connector_protocol_dispatches_multiconnector_to_nixl():
    """MultiConnector delegates PD coordination to its single PD-capable child."""
    proto = make_kv_connector_protocol(
        _config(
            "MultiConnector",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "NixlConnector"},
                    {
                        "kv_connector": "MooncakeStoreConnector",
                        "kv_connector_extra_config": {"load_async": True},
                    },
                ]
            },
        )
    )
    assert isinstance(proto, NixlConnectorProtocol)


def test_make_kv_connector_protocol_dispatches_multiconnector_to_mooncake(
    fake_mooncake,
):
    """Non-PD sub-connectors are ignored when selecting the PD protocol."""
    proto = make_kv_connector_protocol(
        _config(
            "MultiConnector",
            kv_connector_extra_config={
                "connectors": [
                    {
                        "kv_connector": "MooncakeStoreConnector",
                        "kv_connector_extra_config": {"load_async": True},
                    },
                    {"kv_connector": "MooncakeConnector"},
                ]
            },
        )
    )
    assert isinstance(proto, MooncakeConnectorProtocol)


def test_make_kv_connector_protocol_raises_when_multiconnector_has_no_pd_child():
    with pytest.raises(ValueError, match="no PD-capable sub-connector") as exc:
        make_kv_connector_protocol(
            _config(
                "MultiConnector",
                kv_connector_extra_config={
                    "connectors": [{"kv_connector": "MooncakeStoreConnector"}]
                },
            )
        )
    assert "MooncakeStoreConnector" in str(exc.value)


def test_make_kv_connector_protocol_raises_when_multiconnector_is_ambiguous():
    with pytest.raises(ValueError, match="multiple PD-capable sub-connectors") as exc:
        make_kv_connector_protocol(
            _config(
                "MultiConnector",
                kv_connector_extra_config={
                    "connectors": [
                        {"kv_connector": "NixlConnector"},
                        {"kv_connector": "MooncakeConnector"},
                    ]
                },
            )
        )
    assert "NixlConnector" in str(exc.value)
    assert "MooncakeConnector" in str(exc.value)


def test_make_kv_connector_protocol_dispatches_pd_connector_to_nixl():
    """Dynamo's PdConnector (dynamo.vllm.pd_connector) subclasses
    vLLM's MultiConnector with the same config shape — an offload child
    (e.g. DynamoConnector) plus NixlConnector for PD — so it must resolve
    through the same delegation instead of the unsupported-connector error."""
    proto = make_kv_connector_protocol(
        _config(
            "PdConnector",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "DynamoConnector"},
                    {"kv_connector": "NixlConnector"},
                ]
            },
        )
    )
    assert isinstance(proto, NixlConnectorProtocol)


def test_pd_connector_errors_name_the_wrapper():
    """Misconfiguration errors must name the connector the operator actually
    configured (PdConnector), not the MultiConnector base it routes through."""
    with pytest.raises(ValueError, match="PdConnector has no PD-capable"):
        make_kv_connector_protocol(
            _config(
                "PdConnector",
                kv_connector_extra_config={
                    "connectors": [{"kv_connector": "DynamoConnector"}]
                },
            )
        )


# ---------------------------------------------------------------------------
# MultiConnector: malformed-config validation
#
# Misconfiguration must surface as the intended ValueError, not an
# AttributeError from dereferencing an unexpected type.
# ---------------------------------------------------------------------------


def test_multiconnector_rejects_non_dict_extra_config():
    with pytest.raises(ValueError, match="kv_connector_extra_config to be a dict"):
        make_kv_connector_protocol(
            _config("MultiConnector", kv_connector_extra_config="bogus")
        )


def test_multiconnector_rejects_non_list_connectors():
    with pytest.raises(ValueError, match="be a list of connector configs"):
        make_kv_connector_protocol(
            _config(
                "MultiConnector",
                kv_connector_extra_config={"connectors": "NixlConnector"},
            )
        )


def test_multiconnector_rejects_non_dict_connector_entry():
    with pytest.raises(ValueError, match="must be a dict"):
        make_kv_connector_protocol(
            _config(
                "MultiConnector",
                kv_connector_extra_config={"connectors": ["NixlConnector"]},
            )
        )


# ---------------------------------------------------------------------------
# MultiConnector: child config binding
#
# vLLM instantiates each sub-connector with its own KVTransferConfig
# (child entry + engine_id fallback to the wrapper's). The delegated
# protocol must be bound the same way, or Mooncake's decode params would
# advertise the wrapper's engine_id as remote_engine_id and the decode
# side could never match the transfer.
# ---------------------------------------------------------------------------


def test_multiconnector_decode_uses_child_engine_id_not_wrapper(fake_mooncake):
    """Regression: child engine_id override must reach remote_engine_id."""
    proto = make_kv_connector_protocol(
        _config(
            "MultiConnector",
            engine_id="wrapper-eng",
            kv_connector_extra_config={
                "connectors": [
                    {
                        "kv_connector": "MooncakeStoreConnector",
                        "kv_connector_extra_config": {"load_async": True},
                    },
                    {"kv_connector": "MooncakeConnector", "engine_id": "mooncake-eng"},
                ]
            },
        )
    )
    params = proto.decode_request_kv_transfer_params(
        SimpleNamespace(kv_transfer_params=None)
    )
    assert params["remote_engine_id"] == "mooncake-eng"


def test_multiconnector_child_engine_id_falls_back_to_wrapper(fake_mooncake):
    """Children without their own engine_id inherit the wrapper's — the
    same fallback vLLM's MultiConnector applies when instantiating them."""
    proto = make_kv_connector_protocol(
        _config(
            "MultiConnector",
            engine_id="wrapper-eng",
            kv_connector_extra_config={
                "connectors": [{"kv_connector": "MooncakeConnector"}]
            },
        )
    )
    params = proto.decode_request_kv_transfer_params(
        SimpleNamespace(kv_transfer_params=None)
    )
    assert params["remote_engine_id"] == "wrapper-eng"


def test_multiconnector_child_config_does_not_leak_wrapper_internals():
    """The delegated protocol sees the child's config: the child's connector
    name, and the child's extra config (not the wrapper's, which holds the
    "connectors" list itself)."""
    proto = make_kv_connector_protocol(
        _config(
            "MultiConnector",
            engine_id="wrapper-eng",
            kv_connector_extra_config={
                "connectors": [{"kv_connector": "NixlConnector"}]
            },
        )
    )
    child_cfg = proto._vllm_config.kv_transfer_config
    assert child_cfg.kv_connector == "NixlConnector"
    assert child_cfg.kv_connector_extra_config == {}
    assert child_cfg.engine_id == "wrapper-eng"


def test_multiconnector_resolution_does_not_mutate_wrapper_config():
    """The child view is built on copies; the engine's own config object
    must keep its wrapper shape (other code paths still read it)."""
    cfg = _config(
        "MultiConnector",
        engine_id="wrapper-eng",
        kv_connector_extra_config={
            "connectors": [{"kv_connector": "NixlConnector", "engine_id": "nixl-eng"}]
        },
    )
    make_kv_connector_protocol(cfg)
    assert cfg.kv_transfer_config.kv_connector == "MultiConnector"
    assert cfg.kv_transfer_config.engine_id == "wrapper-eng"
    assert "connectors" in cfg.kv_transfer_config.kv_connector_extra_config


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
    assert set(KV_CONNECTOR_PROTOCOLS) == {
        "NixlConnector",
        "MooncakeConnector",
        "LMCacheMPConnector",
        "NeuronNixlConnector",
        "MooncakeConnector",
    }
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
