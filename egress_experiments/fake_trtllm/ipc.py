# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The zmq path, modelled after ``tensorrt_llm/executor/ipc.py``'s
``FusedIpcQueue``.

Socket type
-----------
``proxy.py:_setup_queues`` picks the result socket by postproc config::

    socket_type=zmq.PULL if self.enable_postprocess_parallel else zmq.PAIR

The dynamo submission config runs ``num_postprocess_workers: 0``, so
``enable_postprocess_parallel`` is False and the real result lane is a
**zmq.PAIR** -- one sender (the executor worker process), one receiver (the
proxy dispatch thread). PULL only appears on the ``trtllm-serve`` side of the
diagram, where 4 PostprocWorker processes are the multiple senders. This module
therefore models PAIR, which is what the dynamo column actually runs.

Two backends, identical semantics:

* ``zmq`` when pyzmq is importable (the real transport),
* a stdlib ``socket.socketpair`` with length-prefixed pickle frames otherwise,
  so the simulation runs on a bare interpreter.

Both preserve the property the diagram turns on: a response crosses a process
boundary and is **deserialised on the proxy dispatch thread**, not on the event
loop, and one ``put`` of a list is ONE message however many responses it holds
(``FusedIpcQueue.put``: ``batch = obj if isinstance(obj, list) else [obj]``).
"""

from __future__ import annotations

import os
import pickle
import socket
import struct
import tempfile
import uuid
from typing import Any, List, Optional

_HDR = struct.Struct("!I")

# Set DYN_SIM_IPC=socket to force the stdlib backend even when pyzmq is present.
_FORCE = os.environ.get("DYN_SIM_IPC", "").strip().lower()

try:  # pragma: no cover - depends on the environment
    import zmq as _zmq
except ImportError:  # pragma: no cover
    _zmq = None

USING_ZMQ = _zmq is not None and _FORCE != "socket"


class _Endpoint:
    """Common surface: ``put`` a message, ``get`` a message, ``close``."""

    def put(self, obj: Any) -> None:
        raise NotImplementedError

    def get(self, timeout: Optional[float] = None) -> Optional[List[Any]]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _SocketEndpoint(_Endpoint):
    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._buf = bytearray()
        self._timeout: Optional[float] = None

    def put(self, obj: Any) -> None:
        # FusedIpcQueue.put: a list travels as ONE message.
        batch = obj if isinstance(obj, list) else [obj]
        payload = pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)
        self._sock.sendall(_HDR.pack(len(payload)) + payload)

    def _recv_exactly(self, n: int) -> Optional[bytes]:
        while len(self._buf) < n:
            try:
                chunk = self._sock.recv(65536)
            except socket.timeout:
                return None
            except OSError:
                return None
            if not chunk:
                return None
            self._buf.extend(chunk)
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def get(self, timeout: Optional[float] = None) -> Optional[List[Any]]:
        if timeout != self._timeout:
            self._sock.settimeout(timeout)
            self._timeout = timeout
        header = self._recv_exactly(_HDR.size)
        if header is None:
            return None
        (size,) = _HDR.unpack(header)
        # A partial frame must not be silently treated as a clean shutdown:
        # blocking again for the tail is correct, so drop the timeout here.
        if self._timeout is not None:
            self._sock.settimeout(None)
            self._timeout = None
        payload = self._recv_exactly(size)
        if payload is None:
            return None
        return pickle.loads(payload)

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


class _ZmqEndpoint(_Endpoint):
    def __init__(self, address: str, bind: bool):
        # A forked child must NOT reuse the inherited singleton context: zmq
        # contexts do not survive fork. The connecting side is always the
        # child, so give it a fresh one.
        self._ctx = _zmq.Context.instance() if bind else _zmq.Context()
        self._sock = self._ctx.socket(_zmq.PAIR)
        self._sock.setsockopt(_zmq.LINGER, 0)
        if bind:
            self._sock.bind(address)
        else:
            self._sock.connect(address)

    def put(self, obj: Any) -> None:
        batch = obj if isinstance(obj, list) else [obj]
        self._sock.send_pyobj(batch)

    def get(self, timeout: Optional[float] = None) -> Optional[List[Any]]:
        try:
            if timeout is not None and not self._sock.poll(int(timeout * 1000)):
                return None
            return self._sock.recv_pyobj()
        except _zmq.ZMQError:
            return None

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass


class Link:
    """A PAIR link between the proxy (parent) and the executor worker (child).

    ``parent`` is live immediately. The child calls :meth:`open_child` *after*
    fork -- required for zmq (contexts do not survive fork) and harmless for
    the socket backend.
    """

    def __init__(self, name: str):
        self.name = name
        if USING_ZMQ:
            path = os.path.join(
                tempfile.gettempdir(), f"dynsim-{name}-{uuid.uuid4().hex}.ipc"
            )
            self._address = f"ipc://{path}"
            self._path = path
            self.parent: _Endpoint = _ZmqEndpoint(self._address, bind=True)
            self._child_sock = None
        else:
            parent_sock, child_sock = socket.socketpair()
            self._address = None
            self._path = None
            self.parent = _SocketEndpoint(parent_sock)
            self._child_sock = child_sock

    def open_child(self) -> _Endpoint:
        """Called in the child process to obtain its end."""
        if USING_ZMQ:
            return _ZmqEndpoint(self._address, bind=False)
        assert self._child_sock is not None
        return _SocketEndpoint(self._child_sock)

    def close_child_in_parent(self) -> None:
        """Drop the parent's copy of the child fd so EOF propagates."""
        if self._child_sock is not None:
            try:
                self._child_sock.close()
            except OSError:
                pass
            self._child_sock = None

    def close(self) -> None:
        self.parent.close()
        self.close_child_in_parent()
        if self._path:
            try:
                os.unlink(self._path)
            except OSError:
                pass
