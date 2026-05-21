# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-side RPC server living in the dynamo worker parent process.

The actual SourceG2DescriptorRegistry runs inside the engine subprocess
(spawned by TRT-LLM via OpenMPI). It exposes a ZMQ REP loop over a Unix
domain socket at ``/tmp/dynamo_remote_g2_ipc_<dynamo_pid>.sock``.

This module bridges that local socket to dynamo's network-discoverable
runtime endpoints. Two endpoints are registered:

* ``<namespace>.<component>.remote-g2-resolve`` — forwards resolve_and_lease
* ``<namespace>.<component>.remote-g2-release`` — forwards release_lease

Each endpoint handler is an async generator (dynamo's standard endpoint
shape). It uses ``loop.run_in_executor`` to call into the blocking ZMQ
REQ wrapper without blocking the event loop.

The ZMQ REQ wrapper holds a single ``zmq.REQ`` socket guarded by a
``threading.Lock``, since REQ/REP is strict lockstep (you cannot have
two in-flight requests on one socket). For POC throughput this is fine
— the registry's own critical section serializes calls anyway.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
import threading
import time
from typing import Any, AsyncIterator, Optional


class _ZmqReqWrapper:
    """Thread-safe ZMQ REQ client to the engine-subprocess REP service.

    REQ/REP requires strict send-then-recv lockstep on a single socket.
    The lock serializes concurrent callers; only one RPC is in flight
    at a time. That matches the registry's own ``self._lock``, so we're
    not introducing extra contention.
    """

    def __init__(self, socket_path: str, timeout_ms: int = 5000) -> None:
        import zmq

        self._socket_path = socket_path
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.RCVTIMEO = timeout_ms
        self._socket.SNDTIMEO = timeout_ms
        # `connect()` is lazy — actual connection happens on first send.
        self._socket.connect(f"ipc://{socket_path}")
        self._lock = threading.Lock()

    def request(self, method: str, payload: dict) -> dict:
        """Synchronously round-trip a single RPC. Returns the response dict.

        Always grabs the lock. Wire format mirrors what the REP loop in
        the engine subprocess expects: pickle-encoded
        ``{"method": ..., "payload": ...}`` request, pickle-encoded
        ``{"ok": bool, "result"|"error": ...}`` response.
        """
        with self._lock:
            self._socket.send(pickle.dumps({"method": method, "payload": payload}))
            raw = self._socket.recv()
        return pickle.loads(raw)


def _make_resolve_handler(req_wrapper: _ZmqReqWrapper):
    """Build the async-generator handler for the remote-g2-resolve endpoint."""

    async def handler(request: dict, context: Any = None) -> AsyncIterator[dict]:
        plan = (request or {}).get("plan")
        if plan is None:
            yield {"ok": False, "error": "missing 'plan' in request"}
            return
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None, req_wrapper.request, "resolve_and_lease", {"plan": plan}
            )
        except Exception as exc:
            logging.exception("remote_g2: resolve_and_lease RPC raised")
            yield {"ok": False, "error": repr(exc)}
            return
        yield response

    return handler


def _make_release_handler(req_wrapper: _ZmqReqWrapper):
    """Build the async-generator handler for the remote-g2-release endpoint."""

    async def handler(request: dict, context: Any = None) -> AsyncIterator[dict]:
        lease_id = (request or {}).get("lease_id")
        reason = (request or {}).get("reason", "ack")
        if lease_id is None:
            yield {"ok": False, "error": "missing 'lease_id' in request"}
            return
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None,
                req_wrapper.request,
                "release_lease",
                {"lease_id": lease_id, "reason": reason},
            )
        except Exception as exc:
            logging.exception("remote_g2: release_lease RPC raised")
            yield {"ok": False, "error": repr(exc)}
            return
        yield response

    return handler


def _make_metadata_handler(req_wrapper: _ZmqReqWrapper):
    """Build the async-generator handler for the remote-g2-metadata endpoint.

    Unary RPC. Returns the source NIXL agent's identity (remote_name,
    agent_desc bytes, connection_info, source_generation) so target
    workers can call load_remote_agent_by_connection before issuing
    READs.

    When the caller's request includes ``peer_name`` +
    ``peer_connection_info``, those are forwarded to the engine
    subprocess's REP loop, which calls
    ``source_agent.load_remote_agent_by_connection(peer_name, peer_conn)``
    before returning. This implements the bidirectional NIXL handshake.
    """

    async def handler(request: dict, context: Any = None) -> AsyncIterator[dict]:
        req = request or {}
        payload: dict = {}
        peer_name = req.get("peer_name")
        peer_conn = req.get("peer_connection_info")
        if peer_name:
            payload["peer_name"] = peer_name
        if peer_conn:
            payload["peer_connection_info"] = peer_conn
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None, req_wrapper.request, "get_metadata", payload
            )
        except Exception as exc:
            logging.exception("remote_g2: get_metadata RPC raised")
            yield {"ok": False, "error": repr(exc)}
            return
        yield response

    return handler


def _ipc_socket_path() -> str:
    """The Unix-domain-socket path the engine subprocess's REP service
    binds to. Keyed by the dynamo parent's PID so multiple workers on
    one host don't collide."""
    return f"/tmp/dynamo_remote_g2_ipc_{os.getpid()}.sock"


def _wait_for_socket(path: str, timeout_s: float = 30.0) -> bool:
    """Poll for the REP socket file to appear. The engine subprocess
    binds it during register_kv_caches, which happens late in engine
    init. By the time get_llm_engine returns, it should be up — but
    a short poll catches the case where this setup races ahead of the
    subprocess."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.25)
    return False


async def setup_source_rpc_endpoints(
    runtime: Any,
    namespace: str,
    component: str,
) -> Optional[_ZmqReqWrapper]:
    """Register the two source-side dynamo runtime endpoints and spawn
    their serving tasks. Returns the ``_ZmqReqWrapper`` so the caller
    can hold it alive (the socket closes when the wrapper is GCed).

    Returns ``None`` if the engine subprocess's ZMQ REP socket never
    appears (e.g. remote-G2 wasn't actually bootstrapped).
    """
    socket_path = _ipc_socket_path()
    if not _wait_for_socket(socket_path):
        logging.warning(
            "remote_g2: ZMQ REP socket %s never appeared; source RPC endpoints not registered",
            socket_path,
        )
        return None

    req_wrapper = _ZmqReqWrapper(socket_path)

    resolve_ep = runtime.endpoint(f"{namespace}.{component}.remote-g2-resolve")
    release_ep = runtime.endpoint(f"{namespace}.{component}.remote-g2-release")
    metadata_ep = runtime.endpoint(f"{namespace}.{component}.remote-g2-metadata")

    # serve_endpoint may return either a coroutine or a Future depending on
    # the dynamo runtime version. asyncio.ensure_future accepts both and
    # returns a Task/Future we can hold on to without blocking here.
    _resolve_task = asyncio.ensure_future(
        resolve_ep.serve_endpoint(_make_resolve_handler(req_wrapper))
    )
    _release_task = asyncio.ensure_future(
        release_ep.serve_endpoint(_make_release_handler(req_wrapper))
    )
    _metadata_task = asyncio.ensure_future(
        metadata_ep.serve_endpoint(_make_metadata_handler(req_wrapper))
    )

    logging.warning(
        "remote_g2: source RPC endpoints registered "
        "(resolve=%s.%s.remote-g2-resolve release=%s.%s.remote-g2-release "
        "metadata=%s.%s.remote-g2-metadata ipc=%s)",
        namespace,
        component,
        namespace,
        component,
        namespace,
        component,
        socket_path,
    )
    return req_wrapper
