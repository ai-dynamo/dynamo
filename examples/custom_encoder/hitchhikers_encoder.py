# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text-only FullPromptEncoder for end-to-end pipeline testing.

Encodes the text token IDs with the model's own embed_tokens and ignores any
image URLs.  Semantically identical to what the PD worker would compute in
aggregated mode — the purpose is to validate the encoder → LOCAL-transfer → PD
EmbedsPrompt path, not to add a new vision capability.

Usage (via enc_full_prompt_pd.sh):
    DYN_ENCODER_CLASS=examples.custom_encoder.hitchhikers_encoder.HitchhikersGuideEncoder \\
    DYN_ENCODER_MODEL=Qwen/Qwen3-0.6B \\
    DYN_PD_MODEL=Qwen/Qwen3-0.6B \\
    ./enc_full_prompt_pd.sh --single-gpu --transfer-mode local
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from dynamo.vllm.multimodal_utils.custom_encoder import FullPromptEncoder

logger = logging.getLogger(__name__)


class HitchhikersGuideEncoder(FullPromptEncoder):
    """Text-only encoder: returns embed_tokens(lm_token_ids) as the full prompt tensor.

    image_urls is silently ignored.  This encoder is a plumbing test, not a
    production vision encoder.
    """

    def __init__(self) -> None:
        self._embed: Optional[torch.nn.Embedding] = None
        self._device: str = "cpu"

    def load(self, checkpoint_path: str, device: str) -> None:
        """Load embed_tokens from the LM checkpoint; free the rest of the model."""
        from transformers import AutoModelForCausalLM

        self._device = device
        logger.info(
            "[HitchhikersGuideEncoder] loading embed_tokens from %s on %s",
            checkpoint_path,
            device,
        )
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float16,
            device_map=device,
        )
        vocab_size, hidden_dim = model.model.embed_tokens.weight.shape
        weight = model.model.embed_tokens.weight.data.clone()
        del model
        torch.cuda.empty_cache()

        self._embed = (
            torch.nn.Embedding(vocab_size, hidden_dim, _weight=weight).eval().to(device)
        )
        logger.info(
            "[HitchhikersGuideEncoder] ready: vocab=%d hidden=%d",
            vocab_size,
            hidden_dim,
        )

    def encode(
        self,
        image_urls: List[str],
        lm_token_ids: List[int],
    ) -> torch.Tensor:
        """Return embed_tokens(lm_token_ids); image_urls is ignored."""
        assert self._embed is not None, "call load() before encode()"
        ids = torch.tensor(lm_token_ids, dtype=torch.long, device=self._device)
        with torch.no_grad():
            return self._embed(ids)  # (seq_len, hidden_dim)
