# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any


class SGLangEngineQuiesceController:
    def __init__(self, engine: Any):
        self._engine = engine
        self._is_quiesced = False

    @property
    def is_quiesced(self) -> bool:
        return self._is_quiesced

    async def quiesce(self, tags: list[str] | None = None) -> bool:
        if self._is_quiesced:
            return False

        from sglang.srt.managers.io_struct import (
            PauseGenerationReqInput,
            ReleaseMemoryOccupationReqInput,
        )

        await self._engine.tokenizer_manager.pause_generation(PauseGenerationReqInput())
        await self._engine.tokenizer_manager.release_memory_occupation(
            ReleaseMemoryOccupationReqInput(tags=tags),
            None,
        )
        self._is_quiesced = True
        return True

    async def resume(self, tags: list[str] | None = None) -> bool:
        if not self._is_quiesced:
            return False

        from sglang.srt.managers.io_struct import (
            ContinueGenerationReqInput,
            ResumeMemoryOccupationReqInput,
        )

        await self._engine.tokenizer_manager.resume_memory_occupation(
            ResumeMemoryOccupationReqInput(tags=tags),
            None,
        )
        await self._engine.tokenizer_manager.continue_generation(
            ContinueGenerationReqInput()
        )
        return True

    def mark_resumed(self) -> None:
        self._is_quiesced = False
