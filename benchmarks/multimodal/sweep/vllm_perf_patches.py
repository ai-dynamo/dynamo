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

Coarse-grained labels:
  [PERF] openai_preprocess    — total API preprocessing (APIServer)
  [PERF] hf_processor         — HF multimodal processor (APIServer)
  [PERF] vllm_process_inputs  — InputProcessor.process_inputs (APIServer)
  [PERF] vision_encoder       — _execute_mm_encoder (EngineCore/Worker)
  [PERF] llm_forward          — _model_forward (EngineCore/Worker)

Deep sub-step labels:
  [PERF] base64_decode        — pybase64.b64decode per image (data: URLs only)
  [PERF] pil_open_convert     — Image.open + mode convert per image
  [PERF] image_fetch          — total per-image fetch (base64+PIL)
  [PERF] render_messages      — parse_chat_messages + apply_chat_template
  [PERF] tokenize             — tokenizer.encode on rendered prompt
  [PERF] hf_processor_call    — actual HF processor __call__
  [PERF] mm_processor_apply   — total mm_processor.apply (wraps, not replaces)

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

# Suppress warmup logs — set to True once first real request arrives
_warmup_done = False
_request_count = 0


def _perf_log(msg: str) -> None:
    """Log to vllm logger. Suppressed during warmup."""
    if not _warmup_done:
        return
    logger.info(msg)


def _mark_warmup_done() -> None:
    """Called after warmup requests to enable PERF logging."""
    global _warmup_done, _request_count
    _request_count += 1
    # Warmup count from sweep config is 3, so start logging after 3 requests
    if _request_count > 3 and not _warmup_done:
        _warmup_done = True
        logger.info("[PERF] Warmup complete, enabling PERF logging")


# ---------------------------------------------------------------------------
# APIServer patches — coarse-grained (existing)
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
    """Wrap BaseRenderer._process_multimodal to emit [PERF] hf_processor."""
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
    """Wrap OpenAIServingRender.preprocess_chat to emit [PERF] openai_preprocess."""
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
        _mark_warmup_done()
        t0 = time.perf_counter()
        result = await original(self, *args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _perf_log(f"[PERF] openai_preprocess time_ms={elapsed_ms:.2f}")
        return result

    OpenAIServingRender.preprocess_chat = timed_preprocess_chat  # type: ignore[method-assign]
    logger.info("[PERF] Patched OpenAIServingRender.preprocess_chat -> openai_preprocess")


# ---------------------------------------------------------------------------
# APIServer patches — deep sub-step instrumentation
# ---------------------------------------------------------------------------


def _patch_image_media_io() -> None:
    """Replace ImageMediaIO.load_base64 to time base64 decode and PIL open.

    Emits per-image: [PERF] base64_decode, [PERF] pil_open_convert,
    [PERF] image_fetch. Scoped to base64 data: URLs only.
    """
    try:
        from vllm.multimodal.media.image import ImageMediaIO
    except ImportError:
        logger.debug("[PERF] ImageMediaIO not found; skipping image IO patches")
        return

    import pybase64
    from io import BytesIO

    from PIL import Image

    from vllm.multimodal.media.base import MediaWithBytes

    original_load_base64 = ImageMediaIO.load_base64

    @functools.wraps(original_load_base64)
    def timed_load_base64(self: Any, media_type: str, data: str) -> Any:
        t0 = time.perf_counter()

        # Step 1: base64 decode
        t_b64 = time.perf_counter()
        raw_bytes = pybase64.b64decode(data, validate=True)
        b64_ms = (time.perf_counter() - t_b64) * 1000
        b64_size_mb = len(data) / (1024 * 1024)
        raw_size_mb = len(raw_bytes) / (1024 * 1024)
        _perf_log(
            f"[PERF] base64_decode time_ms={b64_ms:.2f} "
            f"b64_size_mb={b64_size_mb:.2f} raw_size_mb={raw_size_mb:.2f}"
        )

        # Step 2: PIL open
        t_pil = time.perf_counter()
        image = Image.open(BytesIO(raw_bytes))
        pil_open_ms = (time.perf_counter() - t_pil) * 1000

        # Step 3: mode convert (forces JPEG decompression)
        t_conv = time.perf_counter()
        converted = self._convert_image_mode(image)
        convert_ms = (time.perf_counter() - t_conv) * 1000
        pil_total_ms = pil_open_ms + convert_ms
        w, h = converted.size
        _perf_log(
            f"[PERF] pil_open_convert time_ms={pil_total_ms:.2f} "
            f"pil_open_ms={pil_open_ms:.2f} convert_ms={convert_ms:.2f} "
            f"width={w} height={h}"
        )

        total_ms = (time.perf_counter() - t0) * 1000
        _perf_log(f"[PERF] image_fetch time_ms={total_ms:.2f}")

        return MediaWithBytes(converted, raw_bytes)

    ImageMediaIO.load_base64 = timed_load_base64  # type: ignore[method-assign]
    logger.info("[PERF] Patched ImageMediaIO.load_base64 -> base64_decode + pil_open_convert")


def _patch_render_messages() -> None:
    """Replace HfRenderer.render_messages_async to time parse_chat and chat_template.

    Replicates the async version's logic exactly from hf.py:681-736.
    Does NOT call rebuild_mm_uuids_from_mm_data (that's sync-only).
    """
    try:
        from vllm.renderers.hf import HfRenderer
    except ImportError:
        logger.debug("[PERF] HfRenderer not found; skipping render_messages patch")
        return

    if not hasattr(HfRenderer, "render_messages_async"):
        logger.debug("[PERF] HfRenderer.render_messages_async not found; skipping")
        return

    original = HfRenderer.render_messages_async

    @functools.wraps(original)
    async def timed_render_messages_async(self: Any, messages: Any, params: Any) -> Any:
        from typing import cast

        from vllm.entrypoints.chat_utils import parse_chat_messages_async
        from vllm.renderers.hf import (
            parse_dec_only_prompt,
            replace_vision_chunk_video_placeholder,
            resolve_chat_template_content_format,
        )

        t0 = time.perf_counter()
        model_config = self.model_config
        tokenizer = self.get_tokenizer()

        # Step 1: parse_chat_messages_async — fetches images (base64+PIL)
        t_parse = time.perf_counter()
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            model_config,
            content_format=resolve_chat_template_content_format(
                chat_template=params.chat_template,
                tools=params.chat_template_kwargs.get("tools"),
                given_format=params.chat_template_content_format,
                tokenizer=tokenizer,
                model_config=model_config,
            ),
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        parse_ms = (time.perf_counter() - t_parse) * 1000

        # Step 2: apply_chat_template — Jinja2 rendering
        t_tmpl = time.perf_counter()
        prompt_raw = await self._apply_chat_template_async(
            model_config,
            tokenizer,
            conversation,
            **params.get_apply_chat_template_kwargs(),
        )
        tmpl_ms = (time.perf_counter() - t_tmpl) * 1000

        # Kimi unified_vision_chunk path (async version — no rebuild_mm_uuids)
        if (
            self.use_unified_vision_chunk
            and mm_uuids is not None
            and mm_data is not None
        ):
            video_placeholder = getattr(
                model_config.hf_config, "video_placeholder", None
            )
            prompt_raw = cast(
                list[int],
                replace_vision_chunk_video_placeholder(
                    prompt_raw, mm_data, video_placeholder
                ),
            )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        total_ms = (time.perf_counter() - t0) * 1000
        _perf_log(
            f"[PERF] render_messages time_ms={total_ms:.2f} "
            f"parse_chat_ms={parse_ms:.2f} chat_template_ms={tmpl_ms:.2f}"
        )

        return conversation, prompt

    HfRenderer.render_messages_async = timed_render_messages_async  # type: ignore[method-assign]
    logger.info("[PERF] Patched HfRenderer.render_messages_async -> render_messages")


def _patch_tokenize() -> None:
    """Wrap BaseRenderer._tokenize_prompt_async to emit [PERF] tokenize."""
    try:
        from vllm.renderers.base import BaseRenderer
    except ImportError:
        logger.debug("[PERF] BaseRenderer not found; skipping tokenize patch")
        return

    if not hasattr(BaseRenderer, "_tokenize_prompt_async"):
        logger.debug("[PERF] _tokenize_prompt_async not found; skipping")
        return

    original = BaseRenderer._tokenize_prompt_async

    @functools.wraps(original)
    async def timed_tokenize(self: Any, prompt: Any, params: Any) -> Any:
        t0 = time.perf_counter()
        result = await original(self, prompt, params)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        n_tokens = len(result.get("prompt_token_ids", []))
        _perf_log(f"[PERF] tokenize time_ms={elapsed_ms:.2f} n_tokens={n_tokens}")
        return result

    BaseRenderer._tokenize_prompt_async = timed_tokenize  # type: ignore[method-assign]
    logger.info("[PERF] Patched BaseRenderer._tokenize_prompt_async -> tokenize")


def _patch_hf_processor_call() -> None:
    """Wrap BaseMultiModalProcessor._call_hf_processor for the actual HF __call__."""
    try:
        from vllm.multimodal.processing.processor import BaseMultiModalProcessor
    except ImportError:
        logger.debug("[PERF] BaseMultiModalProcessor not found; skipping")
        return

    if not hasattr(BaseMultiModalProcessor, "_call_hf_processor"):
        logger.debug("[PERF] _call_hf_processor not found; skipping")
        return

    original = BaseMultiModalProcessor._call_hf_processor

    @functools.wraps(original)
    def timed_call_hf(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = original(self, *args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _perf_log(f"[PERF] hf_processor_call time_ms={elapsed_ms:.2f}")
        return result

    BaseMultiModalProcessor._call_hf_processor = timed_call_hf  # type: ignore[method-assign]
    logger.info("[PERF] Patched BaseMultiModalProcessor._call_hf_processor -> hf_processor_call")


def _patch_process_tokens_async() -> None:
    """Wrap BaseRenderer._process_tokens_async to time the full async dispatch.

    This captures executor queue delay + _process_multimodal body.
    Emits [PERF] hf_processor_wait = total time including queue.
    Compare with [PERF] hf_processor (body only) to see queue delay.
    """
    try:
        from vllm.renderers.base import BaseRenderer
    except ImportError:
        logger.debug("[PERF] BaseRenderer not found; skipping process_tokens_async patch")
        return

    if not hasattr(BaseRenderer, "_process_tokens_async"):
        logger.debug("[PERF] _process_tokens_async not found; skipping")
        return

    original = BaseRenderer._process_tokens_async

    @functools.wraps(original)
    async def timed_process_tokens_async(self: Any, prompt: Any) -> Any:
        t0 = time.perf_counter()
        result = await original(self, prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        # Only log for multimodal (when _process_multimodal_async was called)
        if prompt.get("multi_modal_data"):
            _perf_log(f"[PERF] hf_processor_wait time_ms={elapsed_ms:.2f}")
        return result

    BaseRenderer._process_tokens_async = timed_process_tokens_async  # type: ignore[method-assign]
    logger.info("[PERF] Patched BaseRenderer._process_tokens_async -> hf_processor_wait")


def _patch_mm_processor_apply() -> None:
    """Wrap (NOT replace) BaseMultiModalProcessor.apply for total timing."""
    try:
        from vllm.multimodal.processing.processor import BaseMultiModalProcessor
    except ImportError:
        logger.debug("[PERF] BaseMultiModalProcessor not found; skipping")
        return

    if not hasattr(BaseMultiModalProcessor, "apply"):
        logger.debug("[PERF] BaseMultiModalProcessor.apply not found; skipping")
        return

    original = BaseMultiModalProcessor.apply

    @functools.wraps(original)
    def timed_apply(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = original(self, *args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _perf_log(f"[PERF] mm_processor_apply time_ms={elapsed_ms:.2f}")
        return result

    BaseMultiModalProcessor.apply = timed_apply  # type: ignore[method-assign]
    logger.info("[PERF] Patched BaseMultiModalProcessor.apply -> mm_processor_apply")


# ---------------------------------------------------------------------------
# EngineCore / Worker patches (inherited via fork)
# ---------------------------------------------------------------------------


def _patch_vision_encoder() -> None:
    """Wrap GPUModelRunner._execute_mm_encoder to emit [PERF] vision_encoder."""
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
    """Wrap GPUModelRunner._model_forward to emit [PERF] llm_forward."""
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

    # APIServer patches — coarse-grained
    _patch_input_processor()
    _patch_hf_processor()
    _patch_openai_preprocess()

    # APIServer patches — deep sub-steps
    _patch_image_media_io()
    _patch_render_messages()
    _patch_tokenize()
    _patch_hf_processor_call()
    _patch_process_tokens_async()
    _patch_mm_processor_apply()

    # EngineCore/Worker patches (will activate after fork)
    _patch_vision_encoder()
    _patch_llm_forward()


# Apply patches on import
install()
