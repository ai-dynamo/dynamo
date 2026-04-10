# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Monkey-patches for vLLM v0.19.0 V1 engine to add [PERF] timing annotations.

Import this module before vLLM starts serving to instrument both the
APIServer process and EngineCore process (inherited via fork).

vLLM V1 architecture:
  APIServer process:
    1. render_chat_async -> render_messages + tokenize + _process_multimodal
       (_process_multimodal runs HF processor ~800ms for multimodal)
    2. InputProcessor.process_inputs -> validation + EngineCoreRequest creation
    3. Send EngineCoreRequest over ZMQ to EngineCore

  EngineCore process (forked, inherits patches):
    4. Scheduler batches requests
    5. GPUModelRunner.execute_model:
       a. _execute_mm_encoder  (vision encoder on GPU)
       b. _model_forward       (LLM forward on GPU)

Patched stages for vllm-serve TTFT breakdown:
  [PERF] openai_preprocess    — total API preprocessing (APIServer)
  [PERF] hf_processor         — HF multimodal processor (APIServer)
  [PERF] vllm_process_inputs  — InputProcessor.process_inputs (APIServer)
  [PERF] vision_encoder       — _execute_mm_encoder (EngineCore/Worker)
  [PERF] llm_forward          — _model_forward (EngineCore/Worker)

For the dynamo path, handlers.py already has its own instrumentation.

Usage:
    import benchmarks.multimodal.sweep.vllm_perf_patches  # noqa: F401
    # then start vllm serve normally
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any

logger = logging.getLogger("vllm")


def _perf_log(msg: str) -> None:
    """Log to vllm logger."""
    logger.info(msg)


# ---------------------------------------------------------------------------
# APIServer patches
# ---------------------------------------------------------------------------


def _patch_input_processor() -> None:
    """Wrap InputProcessor.process_inputs to emit [PERF] timing."""
    try:
        from vllm.v1.engine.input_processor import InputProcessor
    except ImportError:
        logger.debug("[PERF] vLLM InputProcessor not found; skipping patch")
        return

    original = InputProcessor.process_inputs

    @functools.wraps(original)
    def timed_process_inputs(
        self: Any, request_id: str, *args: Any, **kwargs: Any
    ) -> Any:
        t0 = time.perf_counter()
        result = original(self, request_id, *args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _perf_log(
            f"[PERF] vllm_process_inputs request_id={request_id} "
            f"time_ms={elapsed_ms:.2f}"
        )
        return result

    InputProcessor.process_inputs = timed_process_inputs  # type: ignore[method-assign]
    logger.info("[PERF] Patched InputProcessor.process_inputs")


def _patch_hf_processor() -> None:
    """Wrap BaseRenderer._process_multimodal to emit [PERF] hf_processor.

    This is the heavy call (~800ms) that runs the HF image processor,
    tokenizer adjustments, and placeholder insertion. It runs in the
    APIServer process inside a thread pool.
    """
    try:
        from vllm.renderers.base import BaseRenderer
    except ImportError:
        logger.debug("[PERF] BaseRenderer not found; skipping hf_processor patch")
        return

    if not hasattr(BaseRenderer, "_process_multimodal"):
        logger.debug("[PERF] BaseRenderer._process_multimodal not found; skipping")
        return

    original = BaseRenderer._process_multimodal

    @functools.wraps(original)
    def timed_process_multimodal(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = original(self, *args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _perf_log(f"[PERF] hf_processor time_ms={elapsed_ms:.2f}")
        return result

    BaseRenderer._process_multimodal = timed_process_multimodal  # type: ignore[method-assign]
    logger.info("[PERF] Patched BaseRenderer._process_multimodal -> hf_processor")


def _patch_openai_preprocess() -> None:
    """Wrap OpenAIServingRender.preprocess_chat to emit [PERF] openai_preprocess.

    In vLLM v0.19.0, the old _preprocess/_preprocess_chat methods no longer
    exist. The preprocessing path is:
      OpenAIServingChat.create_chat_completion
        -> render_chat_request
        -> OpenAIServingRender.render_chat
        -> preprocess_chat (template + tokenize + HF processor)

    We patch preprocess_chat to capture the full preprocessing time.
    """
    try:
        from vllm.entrypoints.serve.render.serving import OpenAIServingRender
    except ImportError:
        logger.debug("[PERF] OpenAIServingRender not found; skipping patch")
        return

    if not hasattr(OpenAIServingRender, "preprocess_chat"):
        logger.debug("[PERF] OpenAIServingRender.preprocess_chat not found; skipping")
        return

    original = OpenAIServingRender.preprocess_chat

    @functools.wraps(original)
    async def timed_preprocess_chat(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = await original(self, *args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _perf_log(f"[PERF] openai_preprocess time_ms={elapsed_ms:.2f}")
        return result

    OpenAIServingRender.preprocess_chat = timed_preprocess_chat  # type: ignore[method-assign]
    logger.info("[PERF] Patched OpenAIServingRender.preprocess_chat -> openai_preprocess")


# ---------------------------------------------------------------------------
# EngineCore / Worker patches (inherited via fork)
# ---------------------------------------------------------------------------


def _patch_vision_encoder() -> None:
    """Wrap GPUModelRunner._execute_mm_encoder to emit [PERF] vision_encoder.

    This runs the vision tower (e.g., SigLIP) on GPU in the EngineCore process.
    Uses torch.cuda.synchronize() for accurate GPU timing.
    """
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception:
        logger.debug("[PERF] GPUModelRunner not importable; skipping vision_encoder patch")
        return

    if not hasattr(GPUModelRunner, "_execute_mm_encoder"):
        logger.debug("[PERF] GPUModelRunner._execute_mm_encoder not found; skipping")
        return

    original = GPUModelRunner._execute_mm_encoder

    @functools.wraps(original)
    def timed_execute_mm_encoder(self: Any, scheduler_output: Any) -> Any:
        # Only time if there are scheduled encoder inputs
        if not scheduler_output.scheduled_encoder_inputs:
            return original(self, scheduler_output)

        import torch

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = original(self, scheduler_output)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _perf_log(f"[PERF] vision_encoder time_ms={elapsed_ms:.2f}")
        return result

    GPUModelRunner._execute_mm_encoder = timed_execute_mm_encoder  # type: ignore[method-assign]
    logger.info("[PERF] Patched GPUModelRunner._execute_mm_encoder -> vision_encoder")


def _patch_llm_forward() -> None:
    """Wrap GPUModelRunner._model_forward to emit [PERF] llm_forward.

    This runs the LLM (e.g., Qwen3 transformer) forward pass on GPU.
    Uses torch.cuda.synchronize() for accurate GPU timing.

    Note: fires on every decode step too. For TTFT analysis, filter by
    looking at the first occurrence per request or by high latency (prefill).
    """
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception:
        logger.debug("[PERF] GPUModelRunner not importable; skipping llm_forward patch")
        return

    if not hasattr(GPUModelRunner, "_model_forward"):
        logger.debug("[PERF] GPUModelRunner._model_forward not found; skipping")
        return

    original = GPUModelRunner._model_forward

    @functools.wraps(original)
    def timed_model_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        import torch

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = original(self, *args, **kwargs)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _perf_log(f"[PERF] llm_forward time_ms={elapsed_ms:.2f}")
        return result

    GPUModelRunner._model_forward = timed_model_forward  # type: ignore[method-assign]
    logger.info("[PERF] Patched GPUModelRunner._model_forward -> llm_forward")


# ---------------------------------------------------------------------------
# Install all patches
# ---------------------------------------------------------------------------

_installed = False


def install() -> None:
    """Apply all [PERF] monkey-patches. Idempotent."""
    global _installed
    if _installed:
        return
    _installed = True

    # APIServer patches
    _patch_input_processor()
    _patch_hf_processor()
    _patch_openai_preprocess()

    # EngineCore/Worker patches (will activate after fork)
    _patch_vision_encoder()
    _patch_llm_forward()


# Apply patches on import
install()
