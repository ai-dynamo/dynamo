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
        def load(self, checkpoint_path, device): ...
        def encode(self, image_urls, lm_token_ids, lm_embed_tokens): ...

    # launch
    python -m dynamo.vllm \\
        --multimodal-encode-worker \\
        --model Qwen/Qwen2.5-VL-3B-Instruct \\
        --full-prompt-encoder-class customer_encoder.MyEncoder \\
        --full-prompt-encoder-checkpoint /weights/my_encoder.pt \\
        --served-model-name my-lm
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import List

import torch


class FullPromptEncoder(ABC):
    """Pluggable vision encoder for the E→PD disaggregated setup (Mode 2).

    The implementor provides the ViT + projector encoding.  Dynamo loads the
    LM's ``embed_tokens`` weight and passes it in; the implementor is
    responsible for embedding text tokens and splicing image embeddings at the
    correct positions.

    Returned tensor shape: ``(seq_len, lm_hidden_dim)`` — the full prompt,
    ready to feed directly into the PD's transformer layers as ``EmbedsPrompt``.
    """

    @abstractmethod
    def load(self, checkpoint_path: str, device: str) -> None:
        """Load vision encoder and projector weights.

        Args:
            checkpoint_path: Path to the encoder checkpoint (local dir or HF id).
            device: Target device string, e.g. ``"cuda"``.
        """
        ...

    @abstractmethod
    def encode(
        self,
        image_urls: List[str],
        lm_token_ids: List[int],
        lm_embed_tokens: torch.nn.Embedding,
    ) -> torch.Tensor:
        """Produce a full prompt embedding by encoding images and splicing with text.

        Args:
            image_urls: Image URLs to encode (one per image placeholder in prompt).
            lm_token_ids: Token IDs as tokenized by the PD model's tokenizer.
                These come from the Dynamo frontend and are in the LM's token space.
            lm_embed_tokens: The PD model's embedding layer, loaded by Dynamo.
                Use this to look up text token embeddings instead of loading
                the LM a second time.

        Returns:
            Tensor of shape ``(seq_len, lm_hidden_dim)`` representing the full
            prompt with image embeddings spliced at the correct positions.
        """
        ...


def load_encoder_class(dotted_class_path: str) -> type:
    """Dynamically import and return a ``FullPromptEncoder`` subclass.

    Args:
        dotted_class_path: Dotted module.ClassName path, e.g.
            ``"customer_encoder.MyEncoder"`` or
            ``"mypackage.encoders.SafeguardEncoder"``.

    Returns:
        The class object (not an instance).

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class does not exist in the module.
        TypeError: If the class is not a subclass of ``FullPromptEncoder``.
    """
    module_path, _, class_name = dotted_class_path.rpartition(".")
    if not module_path:
        raise ImportError(
            f"Expected 'module.ClassName', got '{dotted_class_path}'. "
            "Provide the full dotted path including the module."
        )
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not (isinstance(cls, type) and issubclass(cls, FullPromptEncoder)):
        raise TypeError(f"{dotted_class_path} is not a subclass of FullPromptEncoder")
    return cls
