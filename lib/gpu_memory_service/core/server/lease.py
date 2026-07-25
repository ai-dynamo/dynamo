# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Socket lease liveness shared by GMS transport adapters."""

from __future__ import annotations

import select


def socket_is_alive(sock: object) -> bool:
    """Return whether a connected socket has not reached EOF or reset."""
    try:
        fd = sock.fileno()  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return False
    if fd < 0:
        return False

    flags = select.POLLERR | select.POLLHUP | select.POLLNVAL
    if hasattr(select, "POLLRDHUP"):
        flags |= select.POLLRDHUP
    poller = select.poll()
    poller.register(fd, flags)
    return not poller.poll(0)
