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


def _dispatch_resolve(
    client: _TargetRpcClient,
    payload: dict,
    loop: asyncio.AbstractEventLoop,
    lease_map: _LeaseSourceMap,
    timeout_s: float,
) -> dict:
    """Run the async resolve coroutine on the parent's loop and wait.

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
    # PROBE: dump the resolved descriptors so we can verify the source-side
    # registry actually returned matching blocks. Trim each descriptor for log readability.
    descs = inner.get("descriptors") or []
    pbs = inner.get("per_block_status") or []
    logging.warning(
        "PROBE rpc_chain resolve_response lease_id=%s num_tokens=%s reason=%s source_generation=%s "
        "descriptors_count=%d per_block_status_count=%d  descriptors_head=%s",
        lease_id,
        inner.get("num_tokens"),
        inner.get("reason"),
        inner.get("source_generation"),
        len(descs),
        len(pbs),
        [{k: d.get(k) for k in ("block_hash", "descriptor_generation",
                                "pool_id", "byte_offset", "byte_length")}
         for d in descs[:3]],
    )
    return response


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


def setup_target_rpc_local(
    runtime: Any,
    namespace: str,
    component: str,
    *,
    timeout_s: float = 30.0,
) -> Optional[dict]:
    """Bind the parent's ZMQ REP socket and start the dispatch loop.

    Returns a handle dict (kept alive by the caller) on success, or
    ``None`` on bind failure. The dispatch loop runs in a daemon thread;
    it uses ``asyncio.run_coroutine_threadsafe`` to dispatch into the
    parent's running event loop, where the dynamo client lives.
    """
    import zmq

    from .target_rpc_client import build_target_rpc_client

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
                logging.warning(
                    "PROBE rpc_chain target_rep_recv pid=%d method=%s "
                    "payload_keys=%s",
                    os.getpid(),
                    method,
                    list(payload.keys()) if isinstance(payload, dict) else None,
                )
                if method == "resolve":
                    response = _dispatch_resolve(
                        client, payload, loop, lease_map, timeout_s
                    )
                elif method == "release":
                    response = _dispatch_release(
                        client, payload, loop, lease_map, timeout_s
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
        "remote_g2: target REP bound at %s (namespace=%s component=%s)",
        socket_path,
        namespace,
        component,
    )

    return {
        "socket_path": socket_path,
        "client": client,
        "lease_map": lease_map,
        "thread": thread,
        "rep": rep,
    }
