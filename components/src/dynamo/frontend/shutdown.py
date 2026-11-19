# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import signal
from typing import Protocol

logger = logging.getLogger(__name__)


class ShutdownRuntime(Protocol):
    def shutdown(self) -> None:
        ...

    def force_shutdown(self) -> None:
        ...


def install_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    runtime: ShutdownRuntime,
) -> None:
    """Install staged HTTP shutdown with a force-stop on repeated signals."""
    shutdown_requested = False
    force_stop_requested = False

    def signal_handler() -> None:
        nonlocal force_stop_requested, shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            runtime.shutdown()
            return

        if not force_stop_requested:
            force_stop_requested = True
            logger.warning("Repeated shutdown signal; forcing runtime shutdown")
            runtime.force_shutdown()
            return

        logger.info("Shutdown already requested; ignoring repeated signal")

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
