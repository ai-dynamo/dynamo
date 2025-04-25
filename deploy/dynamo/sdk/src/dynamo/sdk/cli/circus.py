#  SPDX-FileCopyrightText: Copyright (c) 2020 Atalaya Tech. Inc
#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES

# Once planner v1 goes live - this will be be full of more granular APIs

from __future__ import annotations

import contextlib
import os
import pathlib
import shlex
import sys
from dataclasses import dataclass
from typing import Any, Callable

import psutil
from circus.arbiter import Arbiter as _Arbiter
from circus.sockets import CircusSocket
from circus.watcher import Watcher

from .utils import ServiceProtocol


class Arbiter(_Arbiter):
    """Arbiter with cleanup support via exit_stack."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.exit_stack = contextlib.ExitStack()

    def start(self, cb: Callable[[Any], Any] | None = None) -> None:
        """Start arbiter and enter context."""
        self.exit_stack.__enter__()
        fut = super().start(cb)
        if exc := fut.exception():
            raise exc

    def stop(self) -> None:
        """Stop arbiter and cleanup resources."""
        self.exit_stack.__exit__(None, None, None)
        return super().stop()


@dataclass
class CircusRunner:
    """Simple server wrapper for arbiter lifecycle management."""

    arbiter: Arbiter

    def stop(self) -> None:
        self.arbiter.stop()

    @property
    def running(self) -> bool:
        return self.arbiter.running

    def __enter__(self) -> CircusRunner:
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()


MAX_AF_UNIX_PATH_LENGTH = 103


def create_circus_watcher(
    name: str,
    args: list[str],
    *,
    cmd: str = sys.executable,
    use_sockets: bool = True,
    **kwargs: Any,
) -> Watcher:
    return Watcher(
        name=name,
        cmd=shlex.quote(cmd) if psutil.POSIX else cmd,
        args=args,
        copy_env=True,
        stop_children=True,
        use_sockets=use_sockets,
        graceful_timeout=86400,
        respawn=False,  # TODO
        **kwargs,
    )


def create_arbiter(
    watchers: list[Watcher], *, threaded: bool = False, **kwargs: Any
) -> Arbiter:
    endpoint_port = int(os.environ.get("DYN_CIRCUS_ENDPOINT_PORT", "41234"))
    pubsub_port = int(os.environ.get("DYN_CIRCUS_PUBSUB_PORT", "52345"))

    return Arbiter(
        watchers,
        endpoint=f"tcp://127.0.0.1:{endpoint_port}",
        pubsub_endpoint=f"tcp://127.0.0.1:{pubsub_port}",
        check_delay=kwargs.pop("check_delay", 10),
        **kwargs,
    )


def path_to_uri(path: str) -> str:
    """
    Convert a path to a URI.

    Args:
        path: Path to convert to URI.

    Returns:
        URI string. (quoted, absolute)
    """
    return pathlib.PurePosixPath(path).as_uri()


def _get_server_socket(
    service: ServiceProtocol,
    uds_path: str,
) -> tuple[str, CircusSocket]:
    """Create a Unix Domain Socket for a service.

    Args:
        service: The service to create a socket for
        uds_path: Base directory for Unix Domain Sockets
        port_stack: Not used in POSIX implementation, kept for interface compatibility

    Returns:
        Tuple of (socket URI, CircusSocket object)

    Raises:
        AssertionError: If socket path exceeds maximum length
    """
    socket_path = os.path.join(uds_path, f"{id(service)}.sock")
    assert (
        len(socket_path) < MAX_AF_UNIX_PATH_LENGTH
    ), f"Socket path '{socket_path}' exceeds maximum length of {MAX_AF_UNIX_PATH_LENGTH}"

    return path_to_uri(socket_path), CircusSocket(name=service.name, path=socket_path)
