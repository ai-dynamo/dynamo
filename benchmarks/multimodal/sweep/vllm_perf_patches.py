# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Monkey-patches for vLLM to add [PERF] timing annotations.

Import this module before vLLM starts serving to instrument:
- InputProcessor.process_inputs  -> [PERF] vllm_process_inputs
- OpenAIServingChat._preprocess_chat -> [PERF] hf_processor (vllm serve path)

For the dynamo path, handlers.py already has:
- generate_tokens -> [PERF] llm_forward
- _extract_multimodal_data -> [PERF] extract_mm_data
And the Rust preprocessor has:
- gather_mm_data -> [PERF] gather_mm_data

Usage:
    import benchmarks.multimodal.sweep.vllm_perf_patches  # noqa: F401
    # then start vllm serve normally
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any

logger = logging.getLogger("dynamo.perf_patches")


def _patch_input_processor() -> None:
    """Wrap InputProcessor.process_inputs to emit [PERF] timing."""
    try:
        from vllm.v1.engine.input_processor import InputProcessor
    except ImportError:
        logger.debug("vLLM InputProcessor not found; skipping patch")
        return

    original = InputProcessor.process_inputs

    @functools.wraps(original)
    def timed_process_inputs(
        self: Any, request_id: str, *args: Any, **kwargs: Any
    ) -> Any:
        t0 = time.perf_counter()
        result = original(self, request_id, *args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[PERF] vllm_process_inputs request_id={request_id} "
            f"time_ms={elapsed_ms:.2f}"
        )
        return result

    InputProcessor.process_inputs = timed_process_inputs  # type: ignore[method-assign]
    logger.info("[PERF] Patched InputProcessor.process_inputs")


def _patch_openai_preprocess_chat() -> None:
    """Wrap OpenAIServingChat._preprocess/_preprocess_chat to emit [PERF] hf_processor timing.

    This fires for the vllm-baseline path (vllm serve) where vLLM's built-in
    OpenAI serving layer handles chat preprocessing including the HF processor.
    """
    try:
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    except ImportError:
        try:
            from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
        except ImportError:
            logger.debug("OpenAIServingChat not found; skipping patch")
            return

    # vLLM 0.19.0 uses _preprocess; older versions use _preprocess_chat
    method_name = "_preprocess_chat"
    if not hasattr(OpenAIServingChat, method_name):
        method_name = "_preprocess"
    if not hasattr(OpenAIServingChat, method_name):
        logger.debug(
            "OpenAIServingChat has neither _preprocess_chat nor _preprocess; "
            "skipping patch"
        )
        return

    original = getattr(OpenAIServingChat, method_name)

    @functools.wraps(original)
    async def timed_preprocess(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = await original(self, *args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[PERF] hf_processor time_ms={elapsed_ms:.2f}")
        return result

    setattr(OpenAIServingChat, method_name, timed_preprocess)
    logger.info(f"[PERF] Patched OpenAIServingChat.{method_name}")


_installed = False


def install() -> None:
    """Apply all [PERF] monkey-patches. Idempotent."""
    global _installed
    if _installed:
        return
    _installed = True
    _patch_input_processor()
    _patch_openai_preprocess_chat()


# Apply patches on import
install()
