# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example custom encoder that fakes an image as a known text phrase.

Instead of a real vision encoder, ``forward_batch()`` returns the LM's
``embed_tokens`` embeddings of a fixed phrase (default: *"the Ultimate Question
of Life, the Universe, and Everything"*).  Splicing those embeddings in at the
image placeholder makes the assembled prompt read as one coherent sentence, so
the mixed-embeds path can be checked for **semantic** correctness, not just
shape:

    "Based on The Hitchhiker's Guide to the Galaxy, The Answer to"
        + <image>            # → embeds of " the Ultimate Question of Life, ..."
        + " is?"
    → the model answers "42".

The image URL is ignored — any URL yields the same phrase embeddings.

Subclasses ``QwenCustomEncoder`` (in ``qwen_custom_encoder``): the default model
is a Qwen-family LM (Qwen2.5), so the base loads the tokenizer and resolves the
``<|image_pad|>`` placeholder id; this class only loads the ``embed_tokens``
weight and implements the synchronous ``forward_batch``.

Usage (via agg_custom.sh):
    DYN_ENCODER_CLASS=examples.custom_encoder.hitchhikers_custom_encoder.HitchhikersCustomEncoder
    DYN_MODEL=Qwen/Qwen2.5-1.5B-Instruct
    DYN_CUSTOM_PHRASE=" the Ultimate Question of Life, the Universe, and Everything"
    ./agg_custom.sh
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List

import torch
from safetensors import safe_open
from transformers.utils import cached_file

from examples.custom_encoder.qwen_custom_encoder import QwenCustomEncoder

logger = logging.getLogger(__name__)

# The Answer to the Ultimate Question of Life, the Universe, and Everything is 42.
_PHRASE = os.environ.get(
    "DYN_CUSTOM_PHRASE",
    " the Ultimate Question of Life, the Universe, and Everything",
)


def _load_embed_tokens_weight(model_id: str) -> torch.Tensor:
    """Load only ``embed_tokens.weight`` from a HF checkpoint (lazy safetensors read).

    Works for both local directories and HF hub model IDs (resolved through the
    HF cache), and for sharded and single-file checkpoints.
    """
    try:
        index_path = cached_file(model_id, "model.safetensors.index.json")
        model_dir = Path(index_path).parent
        weight_map = json.loads(Path(index_path).read_text())["weight_map"]
        embed_key = next(
            (k for k in weight_map if k.endswith("embed_tokens.weight")), None
        )
        if embed_key is None:
            raise FileNotFoundError(
                f"No embed_tokens.weight key in safetensors index for {model_id}"
            )
        shard_path = model_dir / weight_map[embed_key]
    except (OSError, StopIteration):
        # Fallback: single-file safetensors model.
        shard_path = Path(cached_file(model_id, "model.safetensors"))
        embed_key = None  # scanned below

    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        if embed_key is None:
            embed_key = next(
                (k for k in f.keys() if k.endswith("embed_tokens.weight")), None
            )
        if embed_key is None:
            raise FileNotFoundError(f"embed_tokens.weight not found in {shard_path}")
        return f.get_tensor(embed_key)


class HitchhikersCustomEncoder(QwenCustomEncoder):
    """Encoder that returns the LM embeddings of a fixed phrase for any image URL.

    A test/example encoder, not a production vision encoder: it loads the LM's
    ``embed_tokens`` weight and returns the embeddings of ``DYN_CUSTOM_PHRASE``
    so the spliced prompt reads as a coherent sentence.
    """

    def build(self, model_id: str, device: str) -> None:
        """Load the tokenizer (via the Qwen base) and the LM ``embed_tokens`` weight."""
        super().build(model_id, device)  # loads self.tokenizer
        self._embed_weight = _load_embed_tokens_weight(model_id)
        logger.info(
            "[HitchhikersCustomEncoder] ready: embed_weight=%s dtype=%s phrase=%r",
            tuple(self._embed_weight.shape),
            self._embed_weight.dtype,
            _PHRASE,
        )

    def forward_batch(self, image_urls: List[str]) -> List[torch.Tensor]:
        """Return the ``embed_tokens`` embeddings of the phrase for each URL.

        Synchronous batched forward — the BatchedCustomEncoder base runs it on
        the dedicated thread, one batch at a time.
        """
        # Explicit check (not assert): asserts are stripped under `python -O`,
        # which would turn a missing build() into an opaque None-index crash.
        if (
            getattr(self, "_embed_weight", None) is None
            or getattr(self, "tokenizer", None) is None
        ):
            raise RuntimeError(
                "HitchhikersCustomEncoder.forward_batch() called before "
                "build(); call load(model_id, device) first."
            )
        ids = self.tokenizer.encode(_PHRASE, add_special_tokens=False)
        phrase_embeds = self._embed_weight[torch.tensor(ids, dtype=torch.long)]
        logger.debug(
            "[HitchhikersCustomEncoder] phrase tokens=%d → shape=%s",
            len(ids),
            tuple(phrase_embeds.shape),
        )
        return [phrase_embeds.clone() for _ in image_urls]
