# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KVBM-specific conftest: reuse pre-existing NATS/etcd or spawn with dynamic ports.

Avoids conflicts with services already running on the host and removes the
requirement for nats-server/etcd binaries on PATH when services are available.
"""

import asyncio
import os
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
import requests

# Register the layered fixtures (deps / server / eval) for test discovery.
# See tests/kvbm_integration/fixtures/ and the README for the layered architecture.
from .fixtures import (  # noqa: F401
    kvbm_deps,
    kvbm_server,
    kvbm_server_spec,
    kvbm_tester,
)


def _parse_port(url: str, default: int) -> int:
    """Extract port from a URL like nats://localhost:4222 or http://localhost:2379."""
    parsed = urlparse(url)
    return parsed.port or default


def _nats_available(url: str) -> bool:
    """Probe NATS using async connect + JetStream check (same as NatsServer._nats_ready)."""
    import nats

    async def _check():
        try:
            nc = await nats.connect(url, connect_timeout=2)
            try:
                js = nc.jetstream()
                await js.account_info()
                return True
            finally:
                await nc.close()
        except Exception:
            return False

    try:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_check())
        finally:
            loop.close()
    except Exception:
        return False


def _etcd_available(endpoint: str) -> bool:
    """Probe etcd via its health endpoint."""
    try:
        resp = requests.get(f"{endpoint}/health", timeout=2)
        return resp.ok
    except Exception:
        return False


def _save_and_set_env(nats_url, etcd_url):
    """Save original env vars and set new ones. Returns originals for restore."""
    orig_nats = os.environ.get("NATS_SERVER")
    orig_etcd = os.environ.get("ETCD_ENDPOINTS")
    os.environ["NATS_SERVER"] = nats_url
    os.environ["ETCD_ENDPOINTS"] = etcd_url
    return orig_nats, orig_etcd


def _restore_env(orig_nats, orig_etcd):
    """Restore original env vars."""
    for key, orig in [("NATS_SERVER", orig_nats), ("ETCD_ENDPOINTS", orig_etcd)]:
        if orig is not None:
            os.environ[key] = orig
        else:
            os.environ.pop(key, None)


@pytest.fixture()
def runtime_services(request, discovery_backend, request_plane, durable_kv_events):
    """Use pre-existing NATS/etcd if reachable, otherwise spawn with dynamic ports."""
    nats_url = os.environ.get("NATS_SERVER", "nats://localhost:4222")
    etcd_url = os.environ.get("ETCD_ENDPOINTS", "http://localhost:2379")

    if _nats_available(nats_url) and _etcd_available(etcd_url):
        nats_port = _parse_port(nats_url, 4222)
        etcd_port = _parse_port(etcd_url, 2379)

        orig_nats, orig_etcd = _save_and_set_env(nats_url, etcd_url)
        yield SimpleNamespace(port=nats_port), SimpleNamespace(port=etcd_port)
        _restore_env(orig_nats, orig_etcd)
    else:
        # Fall back: spawn fresh instances with dynamic ports
        from tests.conftest import EtcdServer, NatsServer

        with NatsServer(
            request, port=0, disable_jetstream=not durable_kv_events
        ) as nats_proc:
            with EtcdServer(request, port=0) as etcd_proc:
                orig_nats, orig_etcd = _save_and_set_env(
                    f"nats://localhost:{nats_proc.port}",
                    f"http://localhost:{etcd_proc.port}",
                )
                yield nats_proc, etcd_proc
                _restore_env(orig_nats, orig_etcd)
