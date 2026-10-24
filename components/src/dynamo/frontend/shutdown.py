# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import signal
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


class ShutdownHandle(Protocol):
    def shutdown(self) -> None:
        ...


def install_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    runtime: ShutdownHandle,
    frontend_shutdown: Optional[ShutdownHandle],
) -> None:
    """Install staged HTTP shutdown with a force-stop on repeated signals."""
    shutdown_requested = False
    force_stop_requested = False

    def signal_handler() -> None:
        nonlocal force_stop_requested, shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            if frontend_shutdown is not None:
                frontend_shutdown.shutdown()
            else:
                runtime.shutdown()
            return

        if frontend_shutdown is not None and not force_stop_requested:
            force_stop_requested = True
            logger.warning("Repeated shutdown signal; forcing runtime shutdown")
            runtime.shutdown()
            return

        logger.info("Shutdown already requested; ignoring repeated signal")

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
