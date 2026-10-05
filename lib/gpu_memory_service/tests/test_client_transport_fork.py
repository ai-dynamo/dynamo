# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A GMS session belongs to the process that connected it.

Persistent claims live as long as the server-side socket. A forked child that
kept the parent's socket would extend the parent's claims past the parent's
exit, so the child must drop its inherited copy and connect its own session.
"""

import os
import socket
import tempfile

import pytest
from gpu_memory_service.client.rpc import _GMSRPCTransport
from gpu_memory_service.common.protocol.messages import (
    ReleasePersistentAllocationRequest,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_forked_child_drops_inherited_session_without_closing_parents():
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as root:
        path = os.path.join(root, "gms.sock")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(path)
        listener.listen(1)
        transport = _GMSRPCTransport(path)
        transport.connect()
        server_side, _ = listener.accept()

        pid = os.fork()
        if pid == 0:  # Child: report through the exit status only.
            code = 0
            try:
                if transport.is_connected:
                    code = 1
                    # Fail fast instead of blocking on the parent's session.
                    transport._socket.settimeout(2)
                try:
                    transport.request(
                        ReleasePersistentAllocationRequest("engine", "kv"), object
                    )
                    code = 2
                except RuntimeError as exc:
                    if "inherited through fork" not in str(exc):
                        code = 3
            finally:
                os._exit(code)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0

        # The child's exit closed only its copy: the session is still open.
        assert transport.is_connected
        server_side.setblocking(False)
        with pytest.raises(BlockingIOError):
            server_side.recv(1)

        transport.close()
        server_side.setblocking(True)
        assert server_side.recv(1) == b""  # EOF once the owner closes.
        server_side.close()
        listener.close()
