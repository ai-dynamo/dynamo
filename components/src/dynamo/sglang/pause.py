# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.constants import GPU_MEMORY_ALL_TYPES
from sglang.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)

logger = logging.getLogger(__name__)


class SGLangEnginePauseController:
    def __init__(self, engine: Any):
        self._engine = engine
        self._is_paused = False
        self._generation_paused = False
        # Tags released and not yet resumed. Empty means nothing is offloaded, which
        # is what ``is_paused`` reports on, so a partial resume stays "paused".
        self._released_tags: set[str] = set()

    @property
    def is_paused(self) -> bool:
        # Keep the worker paused while any memory region remains unmapped.
        return self._is_paused or bool(self._released_tags)

    @property
    def needs_resume_recovery(self) -> bool:
        return self._generation_paused

    async def pause(self, tags: list[str] | None = None) -> bool:
        if self._is_paused or self._generation_paused:
            return False

        await self._engine.tokenizer_manager.pause_generation(PauseGenerationReqInput())
        self._generation_paused = True
        try:
            await self._engine.tokenizer_manager.release_memory_occupation(
                ReleaseMemoryOccupationReqInput(tags=tags),
                None,
            )
        except Exception:
            try:
                await self._engine.tokenizer_manager.continue_generation(
                    ContinueGenerationReqInput()
                )
                self._generation_paused = False
            except Exception:
                logger.exception(
                    "failed to resume generation after memory release failed"
                )
            raise

        self._is_paused = True
        self._released_tags = set(tags) if tags else set(GPU_MEMORY_ALL_TYPES)
        return True

    async def resume(self, tags: list[str] | None = None) -> bool:
        if not self._is_paused and not self._generation_paused:
            return False

        if self._is_paused:
            await self._engine.tokenizer_manager.resume_memory_occupation(
                ResumeMemoryOccupationReqInput(tags=tags),
                None,
            )
            self._released_tags -= set(tags) if tags else set(GPU_MEMORY_ALL_TYPES)
            if self._released_tags:
                # Generation must stay paused until every released region is back.
                return True
            self._is_paused = False
        if self._generation_paused:
            await self._engine.tokenizer_manager.continue_generation(
                ContinueGenerationReqInput()
            )
            self._generation_paused = False
        return True

    def mark_resumed(self) -> None:
        if self._released_tags:
            return
        self._is_paused = False
        self._generation_paused = False
