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


class SGLangEngineQuiesceController:
    def __init__(self, engine: Any):
        self._engine = engine
        self._is_quiesced = False
        self._generation_paused = False

    @property
    def is_quiesced(self) -> bool:
        return self._is_quiesced

    @property
    def needs_resume_recovery(self) -> bool:
        return self._generation_paused

    async def quiesce(self, tags: list[str] | None = None) -> bool:
        if self._is_quiesced or self._generation_paused:
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

        self._is_quiesced = True
        return True

    async def resume(self, tags: list[str] | None = None) -> bool:
        if not self._is_quiesced and not self._generation_paused:
            return False

        if self._is_quiesced:
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
        self._is_quiesced = False
        self._generation_paused = False
