# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interface for pluggable vision encoders in the Dynamo E→PD disaggregated setup.

Customers implement ``FullPromptEncoder`` to plug a proprietary or non-standard
vision encoder (ViT + projector) into Dynamo without modifying vLLM.

The encoder worker calls ``encode()`` and transfers the returned tensor to the
PD worker as an ``EmbedsPrompt``, bypassing vLLM's multimodal renderer entirely.
The PD just runs transformer layers on the received embeddings.

Usage::

    # customer_encoder.py (in their private Docker image)
    class MyEncoder(FullPromptEncoder):
        def load(self, checkpoint_path, device):
            # load ViT, projector, and any text embedding table needed
            ...
        def encode(self, image_urls, lm_token_ids):
            # encode images, embed text tokens, splice, return full tensor
            ...

    # launch
    python -m dynamo.vllm \\
        --multimodal-encode-worker \\
        --model /weights/my_encoder \\
        --full-prompt-encoder-class customer_encoder.MyEncoder \\
        --served-model-name my-lm
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch


class FullPromptEncoder(ABC):
    """Pluggable vision encoder for the E→PD disaggregated setup (Mode 2).

    The implementor is responsible for all encoding work: loading the ViT,
    projector, and any text embedding table needed.  Dynamo passes the raw
    token IDs from the frontend and expects a single ready-to-use embedding
    tensor back.

    Returned tensor shape: ``(seq_len, lm_hidden_dim)`` — the full prompt,
    ready to feed directly into the PD's transformer layers as ``EmbedsPrompt``.
    """

    @abstractmethod
    def load(self, checkpoint_path: str, device: str) -> None:
        """Load all weights needed for encoding (ViT, projector, embed_tokens, …).

        Args:
            checkpoint_path: Path to the encoder checkpoint (local dir or HF id).
                Passed verbatim from ``--model`` on the encoder worker.
            device: Target device string, e.g. ``"cuda"``.
        """
        ...

    @abstractmethod
    def encode(
        self,
        image_urls: List[str],
        lm_token_ids: List[int],
    ) -> torch.Tensor:
        """Produce a full prompt embedding: images spliced with text.

        Args:
            image_urls: Image URLs to encode (one per image in the prompt).
            lm_token_ids: Token IDs as tokenized by the PD model's tokenizer,
                forwarded verbatim from the Dynamo frontend.

        Returns:
            Tensor of shape ``(seq_len, lm_hidden_dim)`` representing the full
            prompt with image embeddings spliced at the correct positions.
        """
        ...
