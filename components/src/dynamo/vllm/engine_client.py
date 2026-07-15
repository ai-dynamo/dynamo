# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural engine-client types shared by legacy vLLM integrations."""

from __future__ import annotations

from typing import Any, Protocol

from vllm.config import VllmConfig


class EngineCoreClient(Protocol):
    async def call_utility_async(self, method: str, *args: Any) -> Any:
        ...


class VllmEngineClient(Protocol):
    vllm_config: VllmConfig
    model_config: Any
    engine_core: EngineCoreClient
    tokenizer: Any
    log_stats: bool

    async def check_health(self) -> None:
        ...

    async def do_log_stats(self) -> None:
        ...

    def shutdown(self, *args: Any, **kwargs: Any) -> None:
        ...
