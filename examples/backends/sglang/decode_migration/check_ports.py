#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Fail fast when a local migration-test port is already occupied."""

from __future__ import annotations

import socket
import sys


def main() -> None:
    sockets: list[socket.socket] = []
    try:
        for value in sys.argv[1:]:
            port = int(value)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(("0.0.0.0", port))
            except OSError as exc:
                raise SystemExit(f"Port {port} is unavailable: {exc}") from exc
            sockets.append(sock)
    finally:
        for sock in sockets:
            sock.close()


if __name__ == "__main__":
    main()
