#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Layer A (v1) — provide NATS + etcd for layered local iteration.
#
# Mirrors tests/kvbm_integration/conftest.py:runtime_services: if reachable
# NATS+etcd are already running (defaults: nats://localhost:4222 +
# http://localhost:2379, overridable via NATS_SERVER / ETCD_ENDPOINTS), the
# script reuses them and exits 0 immediately after printing the exports.
# Otherwise it spawns fresh instances in the foreground (Ctrl-C to stop).
#
# v2 agg needs no deps (discovery defaults to None per kvbm-config/messenger.rs:43);
# there is no run_deps_v2.sh. v2 disagg will need a deps script — phase TBD.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

exec python - <<'PY'
import asyncio
import os
import sys
import time
from urllib.parse import urlparse


def _nats_available(url: str) -> bool:
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
    import requests

    try:
        return requests.get(f"{endpoint}/health", timeout=2).ok
    except Exception:
        return False


def _print_exports(nats_url: str, etcd_url: str, mode: str) -> None:
    print("", flush=True)
    print("=" * 64, flush=True)
    print(f"[deps] {mode}. Export these in shell 2 (run_server.sh):", flush=True)
    print(f"  export NATS_SERVER={nats_url}", flush=True)
    print(f"  export ETCD_ENDPOINTS={etcd_url}", flush=True)
    print("=" * 64, flush=True)


nats_url = os.environ.get("NATS_SERVER", "nats://localhost:4222")
etcd_url = os.environ.get("ETCD_ENDPOINTS", "http://localhost:2379")

print(f"[deps] probing existing services ({nats_url} / {etcd_url}) ...", flush=True)
if _nats_available(nats_url) and _etcd_available(etcd_url):
    _print_exports(nats_url, etcd_url, "REUSING existing services")
    print("[deps] reuse mode — nothing to keep alive, exiting 0.", flush=True)
    sys.exit(0)

print("[deps] not reachable; spawning fresh NATS+etcd ...", flush=True)
from tests.conftest import EtcdServer, NatsServer  # noqa: E402


class _Node:
    name = "run_deps_v1"


class _Req:
    node = _Node()

    def addfinalizer(self, _fn):
        return None


req = _Req()
with NatsServer(req, port=0, disable_jetstream=False) as nats:
    with EtcdServer(req, port=0) as etcd:
        spawned_nats = f"nats://localhost:{nats.port}"
        spawned_etcd = f"http://localhost:{etcd.port}"
        _print_exports(spawned_nats, spawned_etcd, "SPAWNED")
        print("[deps] Ctrl-C to stop.", flush=True)
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("[deps] stopping ...", flush=True)
PY
