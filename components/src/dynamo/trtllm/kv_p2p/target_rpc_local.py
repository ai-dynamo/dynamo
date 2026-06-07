# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target-side local IPC bridge: parent ZMQ REP loop that translates
requests from the target engine subprocess into dynamo client.direct(...)
calls on the source worker.

This is the mirror of source_rpc_server.py:
* source side: parent REQ ──network──► source-engine-subprocess REP
* target side: target-engine-subprocess REQ ──ipc──► parent REP ──network──► source

The parent REP socket binds at
``/tmp/dynamo_remote_g2_target_<dynamo_pid>.sock``. The engine subprocess
walks its /proc parent chain to find the dynamo parent's PID and
connects to the same path. Wire format mirrors source side: pickle
``{"method": ..., "payload": ...}`` request, pickle
``{"ok": bool, "result": dict | None, "error": str | None}`` response.

Lease lifecycle: the connector's ``release_lease(lease_id, reason)``
signature does not carry source_worker_id, so this bridge maintains a
``lease_id → source_worker_id`` mapping populated from successful
resolve responses. Release calls look up the mapping; if the lease is
unknown (e.g. parent restarted between resolve and release) the call
is silently dropped — the source-side lease will time out via its TTL
mechanism.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
import threading
import time
from typing import Any, Optional

from .target_rpc_client import _TargetRpcClient


class _LeaseSourceMap:
    """Thread-safe ``lease_id → source_worker_id`` map.

    Populated on successful resolve, consulted on release. Bounded only
    by total active leases — entries should be removed by the connector
    via release_lease. POC-level: no eviction policy.
    """

    def __init__(self) -> None:
        self._map: dict[str, int] = {}
        self._lock = threading.Lock()

    def put(self, lease_id: str, source_worker_id: int) -> None:
        with self._lock:
            self._map[lease_id] = source_worker_id

    def pop(self, lease_id: str) -> Optional[int]:
        with self._lock:
            return self._map.pop(lease_id, None)


def _ipc_socket_path() -> str:
    return f"/tmp/dynamo_remote_g2_target_{os.getpid()}.sock"


_DIRECT_ZMQ_PORT = 18888  # Must match source_rpc_server.py
_source_ip_cache: dict[int, str] = {}


def _discover_source_ip(source_worker_id: int) -> Optional[str]:
    """Discover the source worker's pod IP via k8s DNS + ZMQ identify.

    All workers bind port 18888, so we can't just take the first
    reachable IP. Instead, send an ``identify`` request to each
    candidate and match the returned ``worker_id`` against the
    desired ``source_worker_id``.
    """
    if source_worker_id in _source_ip_cache:
        return _source_ip_cache[source_worker_id]

    import pickle
    import socket
    try:
        import zmq
    except ImportError:
        return None

    try:
        infos = socket.getaddrinfo(
            "kv-p2p-test-trtllmworker", None,
            socket.AF_INET, socket.SOCK_STREAM,
        )
    except Exception:
        return None

    seen_ips: set[str] = set()
    for info in infos:
        ip = info[4][0]
        if ip in seen_ips:
            continue
        seen_ips.add(ip)
        addr = f"tcp://{ip}:{_DIRECT_ZMQ_PORT}"
        ctx = zmq.Context.instance()
        req = ctx.socket(zmq.REQ)
        req.RCVTIMEO = 2000
        req.SNDTIMEO = 2000
        try:
            req.connect(addr)
            req.send(pickle.dumps({"method": "identify"}))
            raw = req.recv()
            resp = pickle.loads(raw)
            remote_wid = resp.get("worker_id", "")
            if str(remote_wid) == str(source_worker_id):
                _source_ip_cache[source_worker_id] = ip
                return ip
        except Exception:
            logging.debug(
                "remote_g2: identify probe to %s failed", addr,
            )
        finally:
            req.close()

    logging.warning(
        "remote_g2: _discover_source_ip found no match for "
        "worker_id=%s among %s",
        source_worker_id, seen_ips,
    )
    return None


def _zmq_direct_resolve(source_ip: str, payload: dict) -> Optional[dict]:
    """Direct ZMQ REQ/REP to the source worker's direct TCP port.
    Bypasses dynamo's client.direct() response stream entirely.
    """
    import pickle
    import zmq

    addr = f"tcp://{source_ip}:{_DIRECT_ZMQ_PORT}"
    ctx = zmq.Context.instance()
    req = ctx.socket(zmq.REQ)
    req.RCVTIMEO = 10000
    req.SNDTIMEO = 10000
    try:
        req.connect(addr)
        req.send(pickle.dumps({
            "method": "resolve_and_lease",
            "payload": payload,
        }))
        raw = req.recv()
        return pickle.loads(raw)
    except Exception:
        logging.exception("remote_g2: direct ZMQ resolve to %s failed", addr)
        return None
    finally:
        req.close()


def _dispatch_resolve(
    client: _TargetRpcClient,
    payload: dict,
    loop: asyncio.AbstractEventLoop,
    lease_map: _LeaseSourceMap,
    timeout_s: float,
) -> dict:
    """Resolve via direct ZMQ TCP, falling back to dynamo client.direct().

    Returns the standard ``{"ok": ..., "result": ..., "error": ...}`` shape.
    """
    plan = payload.get("plan")
    source_worker_id = payload.get("source_worker_id")
    if plan is None or source_worker_id is None:
        return {"ok": False, "error": "missing plan or source_worker_id"}

    try:
        source_worker_id = int(source_worker_id)
    except (TypeError, ValueError):
        return {"ok": False, "error": f"invalid source_worker_id: {source_worker_id!r}"}

    # Try direct ZMQ TCP first (bypasses dynamo response stream).
    source_ip = _discover_source_ip(source_worker_id)
    if source_ip:
        response = _zmq_direct_resolve(source_ip, {"plan": plan})
        if response is not None:
            # Success — process lease mapping and return.
            if isinstance(response, dict) and response.get("ok"):
                inner = response.get("result")
                if isinstance(inner, dict):
                    lease_id = inner.get("lease_id")
                    if lease_id:
                        lease_map.put(str(lease_id), source_worker_id)
            return response
        logging.warning(
            "remote_g2: direct ZMQ resolve failed, falling back to client.direct()"
        )

    # Fallback: dynamo client.direct() (works with TP=1).
    coro = client.resolve_and_lease(plan, source_worker_id)
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        response = future.result(timeout=timeout_s)
    except Exception as exc:
        logging.exception("remote_g2: target REP resolve dispatch raised")
        return {"ok": False, "error": repr(exc)}

    if response is None:
        return {"ok": False, "error": "transport_failure"}
    # The source endpoint already yields the wire-shape envelope
    # ``{"ok": True, "result": <flat_dict>}``. Pass it through verbatim —
    # the engine subprocess's _make_resolve_callable expects exactly that
    # outer shape and reads `.result` for the flat dict.
    if not isinstance(response, dict):
        return {"ok": False, "error": f"non_dict_response:{type(response).__name__}"}
    inner = response.get("result") if isinstance(response.get("result"), dict) else {}
    lease_id = inner.get("lease_id")
    if lease_id:
        lease_map.put(str(lease_id), source_worker_id)
    return response


def _dispatch_metadata(
    client: _TargetRpcClient,
    payload: dict,
    loop: asyncio.AbstractEventLoop,
    timeout_s: float,
) -> dict:
    """Forward a get_metadata call from the engine subprocess via the
    target parent's dynamo client to the source worker's
    remote-g2-metadata endpoint. Returns the standard
    ``{"ok": True, "result": <flat_dict>}`` envelope the engine
    subprocess's _TargetReqWrapper expects.

    Note: ``agent_desc`` is raw bytes; both pickle and dynamo NATS
    transports carry it transparently."""
    source_worker_id = payload.get("source_worker_id")
    if source_worker_id is None:
        return {"ok": False, "error": "missing source_worker_id"}
    try:
        source_worker_id = int(source_worker_id)
    except (TypeError, ValueError):
        return {"ok": False, "error": f"invalid source_worker_id: {source_worker_id!r}"}

    # Pass through peer_name + peer_connection_info to enable the
    # bidirectional NIXL handshake.
    peer_name = payload.get("peer_name", "")
    peer_conn = payload.get("peer_connection_info", "")
    coro = client.get_source_metadata(
        source_worker_id,
        peer_name=str(peer_name) if peer_name else "",
        peer_connection_info=str(peer_conn) if peer_conn else "",
    )
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        response = future.result(timeout=timeout_s)
    except Exception as exc:
        logging.exception("remote_g2: target REP metadata dispatch raised")
        return {"ok": False, "error": repr(exc)}

    if response is None:
        return {"ok": False, "error": "transport_failure"}
    return {"ok": True, "result": response}


def _dispatch_release(
    client: _TargetRpcClient,
    payload: dict,
    loop: asyncio.AbstractEventLoop,
    lease_map: _LeaseSourceMap,
    timeout_s: float,
) -> dict:
    lease_id = payload.get("lease_id")
    if lease_id is None:
        return {"ok": False, "error": "missing lease_id"}
    lease_id = str(lease_id)
    reason = str(payload.get("reason", "ack"))
    source = lease_map.pop(lease_id)
    if source is None:
        # Lease unknown — silently treat as ok. Source-side TTL will
        # garbage-collect orphaned leases.
        return {"ok": True, "result": False}

    coro = client.release_lease(lease_id, source, reason)
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        response = future.result(timeout=timeout_s)
    except Exception as exc:
        logging.exception("remote_g2: target REP release dispatch raised")
        return {"ok": False, "error": repr(exc)}

    if response is None:
        return {"ok": False, "error": "transport_failure"}
    completed = bool(response.get("result")) if isinstance(response, dict) else False
    return {"ok": True, "result": completed}


def bind_target_rpc_socket() -> Optional[Any]:
    """Phase 1: Bind the ZMQ REP socket early so the engine subprocess
    can find it during register_kv_caches. Returns the socket on success,
    None on failure. Call start_target_rpc_dispatch() later to start the
    dispatch loop.
    """
    import zmq

    socket_path = _ipc_socket_path()
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass

    ctx = zmq.Context.instance()
    try:
        rep = ctx.socket(zmq.REP)
        rep.bind(f"ipc://{socket_path}")
    except Exception:
        logging.exception(
            "remote_g2: target REP bind failed at %s", socket_path
        )
        return None

    logging.warning(
        "remote_g2: target REP bound at %s (phase 1 — dispatch loop pending)",
        socket_path,
    )
    return rep


def start_target_rpc_dispatch(
    rep: Any,
    runtime: Any,
    namespace: str,
    component: str,
    *,
    timeout_s: float = 30.0,
) -> dict:
    """Phase 2: Start the dispatch loop. Call AFTER the engine is fully
    initialized so the event loop and runtime are stable for client.direct().
    """
    from .target_rpc_client import build_target_rpc_client

    client = build_target_rpc_client(runtime, namespace, component)
    lease_map = _LeaseSourceMap()
    loop = asyncio.get_event_loop()

    def _loop_body() -> None:
        while True:
            try:
                raw = rep.recv()
            except Exception as e:
                logging.error(
                    "remote_g2: target REP recv failed; exiting loop: type=%s repr=%s",
                    type(e).__name__,
                    repr(e),
                )
                return
            try:
                req = pickle.loads(raw)
                method = req.get("method") if isinstance(req, dict) else None
                payload = (req.get("payload") or {}) if isinstance(req, dict) else {}
                if method == "resolve":
                    response = _dispatch_resolve(
                        client, payload, loop, lease_map, timeout_s
                    )
                elif method == "release":
                    response = _dispatch_release(
                        client, payload, loop, lease_map, timeout_s
                    )
                elif method == "metadata":
                    response = _dispatch_metadata(
                        client, payload, loop, timeout_s
                    )
                else:
                    response = {
                        "ok": False,
                        "error": f"unknown method: {method!r}",
                    }
            except Exception as exc:
                logging.exception("remote_g2: target REP dispatch raised")
                response = {"ok": False, "error": repr(exc)}
            try:
                payload_bytes = pickle.dumps(response)
            except Exception as e:
                logging.error(
                    "remote_g2: target REP pickle.dumps failed: type=%s repr=%s response_type=%s response_repr=%s",
                    type(e).__name__,
                    repr(e),
                    type(response).__name__,
                    repr(response)[:300],
                )
                # Send a synthetic error so the REQ side can unblock.
                payload_bytes = pickle.dumps(
                    {"ok": False, "error": f"pickle_failed:{type(e).__name__}"}
                )
            try:
                rep.send(payload_bytes)
            except Exception as e:
                logging.error(
                    "remote_g2: target REP send failed: type=%s repr=%s bytes_len=%d",
                    type(e).__name__,
                    repr(e),
                    len(payload_bytes),
                )

    thread = threading.Thread(
        target=_loop_body, name="remote_g2_target_rep", daemon=True
    )
    thread.start()

    logging.warning(
        "remote_g2: target dispatch started (namespace=%s component=%s)",
        namespace,
        component,
    )

    return {
        "socket_path": _ipc_socket_path(),
        "client": client,
        "lease_map": lease_map,
        "thread": thread,
        "rep": rep,
    }


def setup_target_rpc_local(
    runtime: Any,
    namespace: str,
    component: str,
    *,
    timeout_s: float = 30.0,
) -> Optional[dict]:
    """Bind the parent's ZMQ REP socket and start the dispatch loop.

    Backward-compatible wrapper that calls both phases. For TP>1,
    prefer calling bind_target_rpc_socket() early and
    start_target_rpc_dispatch() after the engine is up.
    """
    rep = bind_target_rpc_socket()
    if rep is None:
        return None
    return start_target_rpc_dispatch(
        rep, runtime, namespace, component, timeout_s=timeout_s
    )
