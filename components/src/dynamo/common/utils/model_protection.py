# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared protected-model materialization lifecycle."""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def materialize_protected_session(
    session: Any, shutdown_event: asyncio.Event
) -> None:
    """Materialize off-loop and join the writer before returning or raising."""
    cancellation = session.cancellation()
    materialize = asyncio.create_task(asyncio.to_thread(session.materialize))
    shutdown = asyncio.create_task(shutdown_event.wait())
    try:
        done, _ = await asyncio.wait(
            (materialize, shutdown), return_when=asyncio.FIRST_COMPLETED
        )
        if shutdown in done and not materialize.done():
            cancellation.cancel()
        await asyncio.shield(materialize)
    except BaseException:
        cancellation.cancel()
        while not materialize.done():
            try:
                await asyncio.shield(materialize)
            except asyncio.CancelledError:
                cancellation.cancel()
            except BaseException:
                break
        if materialize.done() and not materialize.cancelled():
            try:
                materialize.result()
            except BaseException:
                logger.debug(
                    "protected weight materialization failed while the caller "
                    "was already unwinding",
                    exc_info=True,
                )
        raise
    finally:
        shutdown.cancel()
        await asyncio.gather(shutdown, return_exceptions=True)
