# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transactional vLLM sleep/wake and admission-pause state."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VllmEnginePauseController:
    """Coordinate memory residency with scheduler admission.

    Native vLLM sleep/wake controls memory residency, while
    pause_generation/resume_generation controls request admission. Keeping
    those states separate lets wake retries avoid remapping memory after a
    successful native wake and keeps partially-awake engines unpublished.
    """

    def __init__(self, engine_client: Any):
        self._engine_client = engine_client
        self._memory_sleeping = False
        self._generation_paused = False
        self._wake_requires_reconciliation = False

    @property
    def is_paused(self) -> bool:
        return self._memory_sleeping

    @property
    def needs_resume_recovery(self) -> bool:
        return self._generation_paused

    async def pause(self, *args: object) -> bool:
        if self._memory_sleeping or self._generation_paused:
            return False

        level = args[0] if args else None
        await self._engine_client.pause_generation()
        self._generation_paused = True
        try:
            if level is None:
                await self._engine_client.sleep()
            else:
                await self._engine_client.sleep(level)
        except Exception:
            try:
                await self._engine_client.resume_generation()
                self._generation_paused = False
            except Exception:
                logger.exception(
                    "Failed to resume generation after native vLLM sleep failure"
                )
            raise
        self._memory_sleeping = True
        self._wake_requires_reconciliation = False
        return True

    async def resume(self, tags: list[str] | None = None) -> bool:
        if not self._memory_sleeping and not self._generation_paused:
            return False

        if self._memory_sleeping:
            if not self._wake_requires_reconciliation:
                # Mark the native operation ambiguous before awaiting it. If
                # wake_up or the following residency query loses its response,
                # a retry reconciles authoritative state before remapping.
                self._wake_requires_reconciliation = True
                if tags is None:
                    await self._engine_client.wake_up()
                else:
                    await self._engine_client.wake_up(tags)

            is_sleeping = getattr(self._engine_client, "is_sleeping", None)
            if not callable(is_sleeping):
                raise RuntimeError(
                    "engine does not expose authoritative memory-residency state"
                )
            self._memory_sleeping = bool(await is_sleeping())
            self._wake_requires_reconciliation = False
            if self._memory_sleeping:
                raise RuntimeError("engine is still sleeping after partial wake")

        if self._generation_paused:
            await self._engine_client.resume_generation()
            self._generation_paused = False
        return True

    def mark_resumed(self) -> None:
        self._memory_sleeping = False
        self._generation_paused = False
        self._wake_requires_reconciliation = False
