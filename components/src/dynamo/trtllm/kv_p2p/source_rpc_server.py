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
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional


REMOTE_G2_DIRECT_CONTROL_PORT_ENV = "DYN_REMOTE_G2_DIRECT_CONTROL_PORT"
REMOTE_G2_DIRECT_CONTROL_BIND_HOST_ENV = "DYN_REMOTE_G2_DIRECT_CONTROL_BIND_HOST"
REMOTE_G2_DIRECT_CONTROL_ADVERTISE_HOST_ENV = "DYN_REMOTE_G2_DIRECT_CONTROL_ADVERTISE_HOST"


class RemoteG2DirectControlServer:
    """Small source-side direct control endpoint for remote-G2.

    Stage 1 only exposes ``identify`` so targets can validate that a
    discovered TCP endpoint belongs to the intended source worker before
    any side-effecting resolve/lease call is moved off Dynamo
    ``client.direct``.
    """

    def __init__(
        self,
        *,
        worker_id: int | str,
        dp_rank: int = 0,
        tp_rank: int = 0,
        process_generation: str,
        source_generation: int | str,
        bind_host: str = "127.0.0.1",
        advertise_host: str = "127.0.0.1",
        bind_port: int | None = None,
        timeout_ms: int = 5000,
    ) -> None:
        self._identity = {
            "worker_id": str(worker_id),
            "dp_rank": int(dp_rank),
            "tp_rank": int(tp_rank),
            "process_generation": str(process_generation),
            "source_generation": str(source_generation),
            "protocol_version": 1,
        }
        self._bind_host = bind_host
        self._advertise_host = advertise_host
        self._bind_port = (
            bind_port if bind_port is not None else self._port_from_env()
        )
        self._timeout_ms = timeout_ms
        self._ctx = None
        self._socket = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._port: Optional[int] = None

    @staticmethod
    def _port_from_env() -> int | None:
        raw = os.environ.get(REMOTE_G2_DIRECT_CONTROL_PORT_ENV)
        if raw is None or raw == "" or raw == "0":
            return None
        try:
            port = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"{REMOTE_G2_DIRECT_CONTROL_PORT_ENV} must be an integer port"
            ) from exc
        if port <= 0 or port > 65535:
            raise ValueError(
                f"{REMOTE_G2_DIRECT_CONTROL_PORT_ENV} must be in [1, 65535]"
            )
        return port

    @property
    def identity(self) -> dict:
        return dict(self._identity)

    @property
    def address(self) -> str:
        if self._port is None:
            raise RuntimeError("remote-G2 direct control server is not started")
        return f"tcp://{self._advertise_host}:{self._port}"

    def start(self) -> "RemoteG2DirectControlServer":
        import zmq

        if self._thread is not None:
            raise RuntimeError("remote-G2 direct control server already started")

        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.REP)
        self._socket.RCVTIMEO = self._timeout_ms
        self._socket.SNDTIMEO = self._timeout_ms
        if self._bind_port is None:
            self._port = self._socket.bind_to_random_port(
                f"tcp://{self._bind_host}"
            )
        else:
            self._socket.bind(f"tcp://{self._bind_host}:{self._bind_port}")
            self._port = self._bind_port
        self._thread = threading.Thread(
            target=self._serve,
            name="remote_g2_direct_control",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _serve(self) -> None:
        import zmq

        if self._socket is None:
            return
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while not self._stop.is_set():
            try:
                events = dict(poller.poll(100))
            except Exception:
                if not self._stop.is_set():
                    logging.exception("remote_g2: direct control poll failed")
                return
            if self._socket not in events:
                continue
            try:
                request = pickle.loads(self._socket.recv())
                method = request.get("method") if isinstance(request, dict) else None
                if method == "identify":
                    response = {"ok": True, "result": dict(self._identity)}
                else:
                    response = {"ok": False, "error": f"unknown method: {method!r}"}
            except Exception as exc:
                logging.exception("remote_g2: direct control handler failed")
                response = {"ok": False, "error": repr(exc)}
            try:
                self._socket.send(pickle.dumps(response))
            except Exception:
                if not self._stop.is_set():
                    logging.exception("remote_g2: direct control send failed")
                return


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
        self._timeout_ms = timeout_ms
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.RCVTIMEO = timeout_ms
        self._socket.SNDTIMEO = timeout_ms
        # `connect()` is lazy — actual connection happens on first send.
        self._socket.connect(f"ipc://{socket_path}")
        self._lock = threading.Lock()

    def _reset_socket(self) -> None:
        """Close and recreate the REQ socket. Must be called under self._lock."""
        import zmq
        try:
            self._socket.close(linger=0)
        except Exception:
            pass
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.RCVTIMEO = self._timeout_ms
        self._socket.SNDTIMEO = self._timeout_ms
        self._socket.connect(f"ipc://{self._socket_path}")

    def request(self, method: str, payload: dict, _max_retries: int = 3) -> dict:
        """Synchronously round-trip a single RPC with retry on timeout.

        On socket error, resets the socket and retries up to _max_retries
        times with backoff. Prevents the EFSM deadlock where a single
        timeout permanently kills the REQ socket.
        """
        with self._lock:
            last_exc: Optional[Exception] = None
            for _attempt in range(_max_retries):
                try:
                    self._socket.send(pickle.dumps({"method": method, "payload": payload}))
                    raw = self._socket.recv()
                    return pickle.loads(raw)
                except Exception as exc:
                    last_exc = exc
                    logging.warning(
                        "remote_g2: ZMQ request failed (attempt %d/%d): %s — resetting socket",
                        _attempt + 1, _max_retries, exc,
                    )
                    self._reset_socket()
                    if _attempt < _max_retries - 1:
                        time.sleep(0.5 * (_attempt + 1))
            raise last_exc  # type: ignore[misc]


@dataclass
class RemoteG2SourceRpcHandle:
    req_wrapper: _ZmqReqWrapper
    resolve_task: asyncio.Future
    release_task: asyncio.Future
    metadata_task: asyncio.Future
    control_info_task: asyncio.Future
    direct_control: RemoteG2DirectControlServer

    @property
    def direct_address(self) -> str:
        return self.direct_control.address

    @property
    def direct_identity(self) -> dict:
        return self.direct_control.identity


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


def _make_control_info_handler(direct_control: RemoteG2DirectControlServer):
    """Build a non-side-effecting endpoint for direct-control discovery."""

    async def handler(request: dict, context: Any = None) -> AsyncIterator[dict]:
        del request, context
        yield {
            "ok": True,
            "result": {
                "address": direct_control.address,
                "identity": direct_control.identity,
            },
        }

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


def _direct_control_bind_host() -> str:
    return os.environ.get(REMOTE_G2_DIRECT_CONTROL_BIND_HOST_ENV, "0.0.0.0")


def _direct_control_advertise_host() -> str:
    configured = os.environ.get(REMOTE_G2_DIRECT_CONTROL_ADVERTISE_HOST_ENV)
    if configured:
        return configured
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        logging.exception("remote_g2: failed to infer direct control advertise host")
        return "127.0.0.1"


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
) -> Optional[RemoteG2SourceRpcHandle]:
    """Register source-side Dynamo runtime and direct-control endpoints.

    Returns a handle so the caller can keep sockets, tasks, and the
    direct-control thread alive for the worker lifetime.

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
    worker_id = os.environ.get("DYNAMO_REMOTE_G2_WORKER_ID", "")
    dp_rank = int(os.environ.get("DYNAMO_REMOTE_G2_DP_RANK", "0"))
    # Stage 1 uses process generation as the source generation until the
    # source memory-registration path exposes an independent generation.
    process_generation = uuid.uuid4().hex
    direct_control = RemoteG2DirectControlServer(
        worker_id=worker_id,
        dp_rank=dp_rank,
        tp_rank=0,
        process_generation=process_generation,
        source_generation=process_generation,
        bind_host=_direct_control_bind_host(),
        advertise_host=_direct_control_advertise_host(),
    ).start()

    resolve_ep = runtime.endpoint(f"{namespace}.{component}.remote-g2-resolve")
    release_ep = runtime.endpoint(f"{namespace}.{component}.remote-g2-release")
    metadata_ep = runtime.endpoint(f"{namespace}.{component}.remote-g2-metadata")
    control_info_ep = runtime.endpoint(
        f"{namespace}.{component}.remote-g2-control-info"
    )

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
    _control_info_task = asyncio.ensure_future(
        control_info_ep.serve_endpoint(_make_control_info_handler(direct_control))
    )

    logging.warning(
        "remote_g2: source RPC endpoints registered "
        "(resolve=%s.%s.remote-g2-resolve release=%s.%s.remote-g2-release "
        "metadata=%s.%s.remote-g2-metadata control_info=%s.%s.remote-g2-control-info "
        "ipc=%s direct=%s identity=%s)",
        namespace,
        component,
        namespace,
        component,
        namespace,
        component,
        namespace,
        component,
        socket_path,
        direct_control.address,
        direct_control.identity,
    )
    return RemoteG2SourceRpcHandle(
        req_wrapper=req_wrapper,
        resolve_task=_resolve_task,
        release_task=_release_task,
        metadata_task=_metadata_task,
        control_info_task=_control_info_task,
        direct_control=direct_control,
    )
