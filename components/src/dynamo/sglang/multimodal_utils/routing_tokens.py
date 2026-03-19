# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.schedule_batch import (
    MultimodalInputs,
    sanity_check_mm_pad_shift_value,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_tokenizer
from sglang.srt.utils.hf_transformers_utils import get_processor

logger = logging.getLogger(__name__)

_UNSUPPORTED_MM_TYPES = {"audio_url", "input_audio", "video_url"}


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_image_url(part: Any) -> str | None:
    image_url = _get_field(part, "image_url")
    if isinstance(image_url, str):
        return image_url
    if isinstance(image_url, dict):
        return image_url.get("url")
    return getattr(image_url, "url", None)


def extract_image_urls(messages: Sequence[Any]) -> list[str]:
    """Extract image URLs from OpenAI-format chat messages."""
    urls: list[str] = []
    for msg in messages:
        content = _get_field(msg, "content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if _get_field(part, "type") != "image_url":
                continue
            url = _extract_image_url(part)
            if url:
                urls.append(url)
    return urls


def has_unsupported_multimodal_content(messages: Sequence[Any]) -> bool:
    for msg in messages:
        content = _get_field(msg, "content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if _get_field(part, "type") in _UNSUPPORTED_MM_TYPES:
                return True
    return False


def build_multi_modal_data(
    image_urls: Sequence[str],
) -> dict[str, list[dict[str, str]]] | None:
    if not image_urls:
        return None
    return {"image_url": [{"Url": url} for url in image_urls]}


class SglangMultimodalTokenExpander:
    """Build routing token ids that match SGLang's pad-value rewritten MM input."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._processor: Any | None = None
        self._mm_processor: Any | None = None
        self._tokenizer: Any | None = None
        self._processor_init_failed = False

    def expand_prompt_tokens(
        self,
        prompt: str,
        image_urls: Sequence[str],
    ) -> list[int] | None:
        if not image_urls:
            return None

        mm_processor = self._ensure_mm_processor()
        if mm_processor is None:
            return None

        try:
            tokenizer = self._tokenizer
            if tokenizer is not None:
                vocab_size = getattr(tokenizer, "vocab_size", None)
                if vocab_size is None:
                    vocab_size = len(tokenizer)
                sanity_check_mm_pad_shift_value(vocab_size)

            base_output = mm_processor.load_mm_data(
                prompt=prompt,
                multimodal_tokens=mm_processor.mm_tokens,
                image_data=list(image_urls),
            )
            mm_items, input_ids, _ = mm_processor.process_and_combine_mm_data(
                base_output,
                mm_processor.mm_tokens,
            )
        except Exception:
            logger.warning(
                "Failed to build multimodal routing tokens for %s",
                self.model_path,
                exc_info=True,
            )
            return None

        try:
            mm_inputs = MultimodalInputs.from_dict(
                {
                    "mm_items": mm_items,
                    "im_start_id": getattr(
                        mm_processor,
                        "vision_start_token_id",
                        getattr(mm_processor, "IM_START_TOKEN_ID", None),
                    ),
                    "im_end_id": getattr(
                        mm_processor,
                        "vision_end_token_id",
                        getattr(mm_processor, "IM_END_TOKEN_ID", None),
                    ),
                    "im_token_id": getattr(
                        mm_processor,
                        "IM_TOKEN_ID",
                        getattr(mm_processor.mm_tokens, "image_token_id", None),
                    ),
                    "video_token_id": getattr(
                        mm_processor,
                        "VIDEO_TOKEN_ID",
                        getattr(mm_processor.mm_tokens, "video_token_id", None),
                    ),
                    "audio_token_id": getattr(
                        mm_processor,
                        "audio_token_id",
                        getattr(mm_processor.mm_tokens, "audio_token_id", None),
                    ),
                }
            )
        except Exception:
            logger.warning(
                "Failed to build multimodal pad values for %s",
                self.model_path,
                exc_info=True,
            )
            return None

        if hasattr(input_ids, "tolist"):
            return self._pad_with_mm_hashes(input_ids.tolist(), mm_inputs)
        return self._pad_with_mm_hashes(list(input_ids), mm_inputs)

    def _ensure_mm_processor(self) -> Any | None:
        if self._mm_processor is not None:
            return self._mm_processor
        if self._processor_init_failed:
            return None

        try:
            server_args = ServerArgs(
                model_path=self.model_path,
                tokenizer_path=self.model_path,
                trust_remote_code=True,
                device="cpu",
            )
            if server_args.mm_process_config is None:
                server_args.mm_process_config = {}
            set_global_server_args_for_tokenizer(server_args)

            import_processors("sglang.srt.multimodal.processors")
            self._processor = get_processor(
                self.model_path,
                trust_remote_code=True,
                use_fast=True,
            )
            self._tokenizer = getattr(self._processor, "tokenizer", self._processor)
            model_config = server_args.get_model_config()
            self._mm_processor = get_mm_processor(
                model_config.hf_config,
                server_args,
                self._processor,
                "default",
            )
        except Exception:
            self._processor_init_failed = True
            logger.warning(
                "Failed to initialize multimodal routing processor for %s",
                self.model_path,
                exc_info=True,
            )
            return None

        return self._mm_processor

    @staticmethod
    def _pad_with_mm_hashes(
        input_ids: list[int],
        mm_inputs: MultimodalInputs,
    ) -> list[int]:
        padded_ids = list(input_ids)
        for item in mm_inputs.mm_items:
            offsets = item.offsets or []
            if isinstance(offsets, tuple) and len(offsets) == 2:
                offsets = [offsets]
            for start, end in offsets:
                padded_ids[start : end + 1] = [item.pad_value] * (end - start + 1)
        return padded_ids
