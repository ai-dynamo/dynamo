# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RPC composition and wire-error boundary tests."""

from gms_kv_ring.daemon.rpc_dispatch import HANDLERS, dispatch
from gms_kv_ring.daemon.rpc_types import required_digest, required_int, required_str


def test_handler_domains_compose_without_gaps_or_duplicates():
    assert len(HANDLERS) == 31
    assert "ping" in HANDLERS
    assert "attach_engine_pool" in HANDLERS
    assert "staging_scan" in HANDLERS
    assert "fetch_remote" in HANDLERS


def test_dispatch_handles_ping_and_unknown_operations():
    daemon = object()
    assert dispatch(daemon, {"op": "ping"}) == {"ok": True}
    assert dispatch(daemon, {"op": "missing"}) == {
        "ok": False,
        "error": "unknown op 'missing'",
    }


def test_dispatch_converts_handler_exceptions_to_wire_errors():
    response = dispatch(object(), {"op": "attach_engine_pool"})
    assert response == {"ok": False, "error": "KeyError: 'layers'"}


def test_shared_parsers_preserve_wire_coercions():
    msg = {"name": 7, "count": "11", "digest": "00ff"}
    assert required_str(msg, "name") == "7"
    assert required_int(msg, "count") == 11
    assert required_digest(msg, "digest") == b"\x00\xff"
