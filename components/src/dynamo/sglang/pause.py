# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)

logger = logging.getLogger(__name__)


class SGLangEnginePauseController:
    def __init__(self, engine: Any, *, release_memory_on_quiesce: bool = True) -> None:
        self._engine = engine
        self._release_memory_on_quiesce = release_memory_on_quiesce
        self._is_paused = False
        self._generation_paused = False

    @property
    def is_paused(self) -> bool:
        return self._is_paused

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
        return True

    async def quiesce(self, tags: list[str] | None = None) -> bool:
        if self._release_memory_on_quiesce:
            return await self.pause(tags)
        if self._is_paused or self._generation_paused:
            return False

        # A sleeping standby is already initialized against the shared,
        # lease-protected GMS KV pool. Keep those VA mappings and CUDA graph
        # bindings warm, but stop scheduling before it waits for ownership.
        # It remains absent from discovery, so it cannot receive a request or
        # write KV until the failover lock has fenced the old owner.
        await self._engine.tokenizer_manager.pause_generation(PauseGenerationReqInput())
        self._generation_paused = True
        logger.info("[GMS failover] SGLang standby quiesced with KV mapped")
        return True

    async def resume(self, tags: list[str] | None = None) -> bool:
        if not self._is_paused and not self._generation_paused:
            return False

        if self._is_paused:
            await self._engine.tokenizer_manager.resume_memory_occupation(
                ResumeMemoryOccupationReqInput(tags=tags),
                None,
            )
        if self._generation_paused:
            await self._engine.tokenizer_manager.continue_generation(
                ContinueGenerationReqInput()
            )
            self._generation_paused = False
        return True

    def mark_resumed(self) -> None:
        self._is_paused = False
        self._generation_paused = False
