# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interface for pluggable vision encoders in Dynamo.

The encoder runs **in the same process** as the vLLM aggregated worker — no
separate encode worker, no NIXL transfer.  It handles image encoding and
projection to the LM hidden dim; vLLM embeds the text tokens itself while
Dynamo splices the encoder's image embeds at the placeholder positions,
submitting the result as a mixed ``EmbedsPrompt`` (``prompt_token_ids`` +
``prompt_is_token_ids`` + ``prompt_embeds``).

Usage::

    # customer_encoder.py (in their private Docker image)
    class MyEncoder(CustomEncoder):
        def load(self, model_id, device):
            # load ViT + projector only — no LM weights needed
            ...
        def encode(self, image_urls):
            # encode each image, return one (n_tokens, lm_hidden_dim) tensor per URL
            ...

    # launch
    python -m dynamo.vllm \\
        --model /weights/my_lm \\
        --custom-encoder-class customer_encoder.MyEncoder \\
        --enable-multimodal
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch

DEFAULT_IMAGE_PLACEHOLDER_TOKEN = "<|image_pad|>"

# Static fallback only. The authoritative source for the placeholder token ID is
# the model's own tokenizer (see placeholder_token_id_from_tokenizer), which the
# encoder already loads. This name->ID table exists for callers that have no
# tokenizer at hand; maps model-family name fragments (lowercased) to their ID.
PLACEHOLDER_TOKEN_IDS: Dict[str, int] = {
    # Qwen2-VL / Qwen3-VL / Qwen3-VL-MoE — <|image_pad|> = 151655
    "qwen2-vl": 151655,
    "qwen2_vl": 151655,
    "qwen3-vl": 151655,
    "qwen3_vl": 151655,
    # Qwen3.5 family (text-only LM with VL tokenizer) — <|image_pad|> = 248056
    "qwen3.5": 248056,
}


def lookup_placeholder_token_id(model_name_or_path: str) -> Optional[int]:
    """Return the image placeholder token ID for a known model family.

    Matches against the last path component (the HF repo name or local dir
    name) case-insensitively.  Returns ``None`` for unknown families so callers
    can decide whether to raise or fall back.

    Example::

        tid = lookup_placeholder_token_id("Qwen/Qwen3-VL-2B-Instruct")
        # → 151655

    To register a new family add an entry to ``PLACEHOLDER_TOKEN_IDS``.
    """
    # Use the rightmost path segment so both local paths and HF names work.
    name = model_name_or_path.rstrip("/").rsplit("/", 1)[-1].lower()
    for fragment, token_id in PLACEHOLDER_TOKEN_IDS.items():
        if fragment in name:
            return token_id
    return None


def placeholder_token_id_from_tokenizer(
    tokenizer: object,
    token: str = DEFAULT_IMAGE_PLACEHOLDER_TOKEN,
) -> Optional[int]:
    """Resolve the image placeholder token ID from a loaded tokenizer.

    This is the authoritative source: the encoder already loads the model's
    tokenizer, which defines the special token. Prefer it over the static
    ``PLACEHOLDER_TOKEN_IDS`` name table. Returns ``None`` if the tokenizer does
    not define ``token`` (it maps to the unknown-token ID), so callers can fall
    back to the table or raise.
    """
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    tid = convert(token)
    unk = getattr(tokenizer, "unk_token_id", None)
    if tid is None or (unk is not None and tid == unk):
        return None
    return int(tid)


class CustomEncoder(ABC):
    """Pluggable image encoder for aggregated serving.

    Runs **in-process** inside the vLLM aggregated worker — no separate encode
    worker needed.  Handles image encoding + projection to the LM hidden dim.
    vLLM embeds the text tokens itself; Dynamo splices the returned image
    tensors at the placeholder positions and submits a mixed ``EmbedsPrompt``.
    The encoder never needs the LM's text embedding table.

    Returned tensor shape per image: ``(n_visual_tokens, lm_hidden_dim)``
    where ``n_visual_tokens`` is determined by the encoder (e.g. number of
    ViT patches after merging).

    Usage::

        class MyEncoder(CustomEncoder):
            def load(self, model_id, device):
                # load ViT + projector only — no text embedding table needed
                ...
            def encode(self, image_urls):
                # encode each image, return one tensor per URL
                ...

        python -m dynamo.vllm \\
            --model /weights/my_lm \\
            --custom-encoder-class my_module.MyEncoder \\
            --enable-multimodal
    """

    @abstractmethod
    def load(self, model_id: str, device: str) -> None:
        """Load vision encoder weights.

        Args:
            model_id: The LM checkpoint — a local dir or HF model id, passed
                verbatim from ``--model``.
            device: Target device string, e.g. ``"cuda"``.
        """
        ...

    @abstractmethod
    def encode(self, image_urls: List[str]) -> List[torch.Tensor]:
        """Encode images, returning per-image visual token embeddings.

        Called off the event loop via ``asyncio.to_thread``. Dynamo serializes
        calls on a single encoder instance with a lock, so implementations need
        not be re-entrant; but any state shared *across* encoder instances must
        still be guarded by the implementation.

        Args:
            image_urls: URLs of images to encode.

        Returns:
            List of tensors, one per URL, each of shape
            ``(n_visual_tokens, lm_hidden_dim)``.
        """
        ...

    def get_image_placeholder_token_id_override(self) -> Optional[int]:
        """Return a placeholder token ID to force, or ``None`` (default).

        This is the hook subclasses override when auto-detection from the
        tokenizer is not enough — e.g. read an env var / config value, or call
        ``lookup_placeholder_token_id(model_id)`` for a known family. Returning
        ``None`` falls through to tokenizer-based resolution in
        ``get_image_placeholder_token_id``. Most encoders never need this.
        """
        return None

    def get_image_placeholder_token_id(self) -> int:
        """Return the token ID that marks image positions in the prompt.

        Dynamo uses it to locate the image span and splice the encoder tensors
        into the mixed ``EmbedsPrompt``. Concrete — subclasses normally do
        **not** override this. Resolution order:

        1. ``get_image_placeholder_token_id_override()``, if it returns non-None;
        2. the model tokenizer at ``self.tokenizer`` (set by the encoder in
           ``load``), via ``placeholder_token_id_from_tokenizer`` — the
           authoritative source (the model itself defines ``<|image_pad|>``).

        Override ``get_image_placeholder_token_id_override`` for custom
        resolution; override this method only if neither source applies.
        """
        override = self.get_image_placeholder_token_id_override()
        if override is not None:
            return override
        tokenizer = getattr(self, "tokenizer", None)
        tid = (
            placeholder_token_id_from_tokenizer(tokenizer)
            if tokenizer is not None
            else None
        )
        if tid is None:
            raise ValueError(
                "Could not resolve the image placeholder token id. Set "
                "`self.tokenizer` in load(), or override "
                "get_image_placeholder_token_id_override() to return the id "
                "(e.g. from lookup_placeholder_token_id(model_id))."
            )
        return tid
