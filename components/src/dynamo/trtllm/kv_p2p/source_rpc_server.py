# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-side RPC server living in the dynamo worker parent process.

The actual SourceG2DescriptorRegistry runs inside the engine subprocess
(spawned by TRT-LLM via OpenMPI). It exposes a ZMQ REP loop over a Unix
domain socket keyed by the parent dynamo PID and, for ADP, the DP/KV
rank context.

This module bridges that local socket to dynamo's network-discoverable
runtime endpoints. Three endpoints are registered:

* ``<namespace>.<component>.remote-g2-resolve`` — forwards resolve_and_lease
* ``<namespace>.<component>.remote-g2-release`` — forwards release_lease
* ``<namespace>.<component>.remote-g2-metadata`` — forwards get_metadata

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


class _SourceIpcRouter:
    """Route source RPCs to the engine subprocess for the plan's DP rank."""

    def __init__(
        self, wrappers: dict[int, _ZmqReqWrapper], default_dp_rank: int
    ) -> None:
        self._wrappers = wrappers
        self._default_dp_rank = default_dp_rank

    @property
    def dp_ranks(self) -> tuple[int, ...]:
        return tuple(sorted(self._wrappers))

    def _select_dp_rank(self, method: str, payload: dict) -> int:
        if method == "resolve_and_lease":
            plan = payload.get("plan")
            if isinstance(plan, dict):
                return int(plan.get("source_dp_rank", self._default_dp_rank))
        if method in {"release_lease", "get_metadata"}:
            return int(payload.get("source_dp_rank", self._default_dp_rank))
        return self._default_dp_rank

    def request(self, method: str, payload: dict) -> dict:
        dp_rank = self._select_dp_rank(method, payload)
        wrapper = self._wrappers.get(dp_rank)
        if wrapper is None:
            wrapper = self._wrappers[self._default_dp_rank]
            logging.warning(
                "remote_g2: no source IPC wrapper for dp_rank=%s; using dp_rank=%s",
                dp_rank,
                self._default_dp_rank,
            )
        return wrapper.request(method, payload)


def _make_resolve_handler(req_router: _SourceIpcRouter):
    """Build the async-generator handler for the remote-g2-resolve endpoint."""

    async def handler(request: dict, context: Any = None) -> AsyncIterator[dict]:
        plan = (request or {}).get("plan")
        if plan is None:
            yield {"ok": False, "error": "missing 'plan' in request"}
            return

        # Call the engine's ZMQ REP directly (blocking, ~5-10ms).
        # run_in_executor breaks the dynamo response stream with TP>1
        # because the await suspends the generator and the push_handler
        # invalidates the stream before the executor returns.
        try:
            response = req_router.request("resolve_and_lease", {"plan": plan})
        except Exception as exc:
            logging.exception("remote_g2: resolve_and_lease RPC raised")
            yield {"ok": False, "error": repr(exc)}
            return
        yield response

    return handler


def _make_release_handler(req_router: _SourceIpcRouter):
    """Build the async-generator handler for the remote-g2-release endpoint."""

    async def handler(request: dict, context: Any = None) -> AsyncIterator[dict]:
        lease_id = (request or {}).get("lease_id")
        reason = (request or {}).get("reason", "ack")
        source_dp_rank = (request or {}).get("source_dp_rank", 0)
        if lease_id is None:
            yield {"ok": False, "error": "missing 'lease_id' in request"}
            return
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None,
                req_router.request,
                "release_lease",
                {
                    "lease_id": lease_id,
                    "reason": reason,
                    "source_dp_rank": source_dp_rank,
                },
            )
        except Exception as exc:
            logging.exception("remote_g2: release_lease RPC raised")
            yield {"ok": False, "error": repr(exc)}
            return
        yield response

    return handler


def _make_metadata_handler(req_router: _SourceIpcRouter):
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
        try:
            payload: dict = {"source_dp_rank": int(req.get("source_dp_rank", 0))}
        except (TypeError, ValueError):
            yield {
                "ok": False,
                "error": f"invalid source_dp_rank: {req.get('source_dp_rank')!r}",
            }
            return

        peer_name = req.get("peer_name")
        peer_conn = req.get("peer_connection_info")
        if peer_name:
            payload["peer_name"] = peer_name
        if peer_conn:
            payload["peer_connection_info"] = peer_conn
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None, req_router.request, "get_metadata", payload
            )
        except Exception as exc:
            logging.exception("remote_g2: get_metadata RPC raised")
            yield {"ok": False, "error": repr(exc)}
            return
        yield response

    return handler


def _legacy_ipc_socket_path() -> str:
    """The Unix-domain-socket path the engine subprocess's REP service
    binds to. Keyed by the dynamo parent's PID so multiple workers on
    one host don't collide.

    With TP>1, rank 0's socket is at _tp0.sock. Try that first, fall
    back to the TP=1 path for backward compatibility.
    """
    pid = os.getpid()
    tp0_path = f"/tmp/dynamo_remote_g2_ipc_{pid}_tp0.sock"
    if os.path.exists(tp0_path):
        return tp0_path
    return f"/tmp/dynamo_remote_g2_ipc_{pid}.sock"


def _context_ipc_socket_path(dp_rank: int, kv_rank: int = 0) -> str:
    return f"/tmp/dynamo_remote_g2_ipc_{os.getpid()}_dp{dp_rank}_kv{kv_rank}.sock"


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


def _discover_ipc_router(timeout_s: float = 30.0) -> Optional[_SourceIpcRouter]:
    """Find source IPC sockets for either ADP DP ranks or legacy TP mode."""
    deadline = time.monotonic() + timeout_s
    wrappers: dict[int, _ZmqReqWrapper] = {}
    expected_dp_ranks = (0, 1)

    while time.monotonic() < deadline:
        for dp_rank in expected_dp_ranks:
            if dp_rank in wrappers:
                continue
            socket_path = _context_ipc_socket_path(dp_rank)
            if os.path.exists(socket_path):
                wrappers[dp_rank] = _ZmqReqWrapper(socket_path)
                logging.warning(
                    "remote_g2: discovered source IPC socket for dp_rank=%d at %s",
                    dp_rank,
                    socket_path,
                )

        if len(wrappers) == len(expected_dp_ranks):
            return _SourceIpcRouter(wrappers, default_dp_rank=0)

        if not wrappers:
            legacy_path = _legacy_ipc_socket_path()
            if os.path.exists(legacy_path):
                return _SourceIpcRouter(
                    {0: _ZmqReqWrapper(legacy_path)},
                    default_dp_rank=0,
                )

        time.sleep(0.25)

    if wrappers:
        logging.warning(
            "remote_g2: only discovered source IPC sockets for dp_ranks=%s",
            sorted(wrappers),
        )
        default_dp_rank = 0 if 0 in wrappers else min(wrappers)
        return _SourceIpcRouter(wrappers, default_dp_rank=default_dp_rank)
    return None


async def setup_source_rpc_endpoints(
    runtime: Any,
    namespace: str,
    component: str,
) -> Optional[_SourceIpcRouter]:
    """Register the two source-side dynamo runtime endpoints and spawn
    their serving tasks. Returns the ``_ZmqReqWrapper`` so the caller
    can hold it alive (the socket closes when the wrapper is GCed).

    Returns ``None`` if the engine subprocess's ZMQ REP socket never
    appears (e.g. remote-G2 wasn't actually bootstrapped).
    """
    req_router = _discover_ipc_router()
    if req_router is None:
        logging.warning(
            "remote_g2: no source ZMQ REP socket appeared; source RPC endpoints not registered",
        )
        return None

    resolve_ep = runtime.endpoint(f"{namespace}.{component}.remote-g2-resolve")
    release_ep = runtime.endpoint(f"{namespace}.{component}.remote-g2-release")
    metadata_ep = runtime.endpoint(f"{namespace}.{component}.remote-g2-metadata")

    # serve_endpoint may return either a coroutine or a Future depending on
    # the dynamo runtime version. asyncio.ensure_future accepts both and
    # returns a Task/Future we can hold on to without blocking here.
    _resolve_task = asyncio.ensure_future(
        resolve_ep.serve_endpoint(_make_resolve_handler(req_router))
    )
    _release_task = asyncio.ensure_future(
        release_ep.serve_endpoint(_make_release_handler(req_router))
    )
    _metadata_task = asyncio.ensure_future(
        metadata_ep.serve_endpoint(_make_metadata_handler(req_router))
    )

    # Direct ZMQ TCP REP for resolve — bypasses dynamo's client.direct()
    # response stream which fails with TP>1. The target connects directly
    # to this port via ZMQ REQ/REP (reliable, no response stream issues).
    # Wire format: same as the IPC socket (pickle request/response).
    import zmq as _zmq
    _direct_ctx = _zmq.Context.instance()
    _direct_rep = _direct_ctx.socket(_zmq.REP)
    _DIRECT_ZMQ_PORT = 18888
    _direct_rep.bind(f"tcp://0.0.0.0:{_DIRECT_ZMQ_PORT}")
    _direct_port = _DIRECT_ZMQ_PORT

    # Read our own worker_id so the "identify" method can report it.
    _my_worker_id = os.environ.get("DYNAMO_REMOTE_G2_WORKER_ID", "")

    def _direct_loop():
        while True:
            try:
                raw = _direct_rep.recv()
            except Exception:
                logging.exception("remote_g2: direct TCP REP recv failed")
                return
            try:
                req = pickle.loads(raw)
                method = req.get("method") if isinstance(req, dict) else None
                payload = (req.get("payload") or {}) if isinstance(req, dict) else {}
                if method == "identify":
                    # Lightweight probe: return our worker_id so the target
                    # can match IPs to worker identities.
                    response = {
                        "ok": True,
                        "worker_id": _my_worker_id,
                        "dp_ranks": req_router.dp_ranks,
                    }
                elif method == "resolve_and_lease":
                    response = req_router.request("resolve_and_lease", payload)
                elif method == "release_lease":
                    response = req_router.request("release_lease", payload)
                elif method == "get_metadata":
                    response = req_router.request("get_metadata", payload)
                else:
                    response = {"ok": False, "error": f"unknown method: {method!r}"}
            except Exception as exc:
                logging.exception("remote_g2: direct TCP REP handler raised")
                response = {"ok": False, "error": repr(exc)}
            try:
                _direct_rep.send(pickle.dumps(response))
            except Exception:
                logging.exception("remote_g2: direct TCP REP send failed")

    _direct_thread = threading.Thread(
        target=_direct_loop, name="remote_g2_direct_tcp", daemon=True
    )
    _direct_thread.start()

    # Write the direct TCP address + worker_id to a well-known file.
    import socket as _socket
    _my_ip = _socket.gethostbyname(_socket.gethostname())
    _direct_addr = f"{_my_ip}:{_direct_port}"
    _direct_addr_file = f"/tmp/dynamo_remote_g2_direct_{os.getpid()}.addr"
    with open(_direct_addr_file, "w") as f:
        f.write(_direct_addr)
    logging.warning(
        "remote_g2: direct TCP REP bound at %s worker_id=%s (file=%s)",
        _direct_addr, _my_worker_id, _direct_addr_file,
    )

    logging.warning(
        "remote_g2: source RPC endpoints registered "
        "(resolve=%s.%s.remote-g2-resolve release=%s.%s.remote-g2-release "
        "metadata=%s.%s.remote-g2-metadata dp_ranks=%s)",
        namespace,
        component,
        namespace,
        component,
        namespace,
        component,
        req_router.dp_ranks,
    )
    return req_router
