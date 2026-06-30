# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-only stub backend for the client-supplied-embeddings path.

A client that already has per-image embeddings (e.g. CLIP/ViT outputs) can drive
the existing in-process custom-encoder path **with no new Dynamo code**: it sends
each embedding inline as a safetensors ``data:`` URI on a normal ``image_url``
content part, and provides a ``VisionEncoderBackend`` whose ``preprocess`` decodes
the embedding and whose ``forward_batch`` **projects** it to the LM hidden dim.
Dynamo splices the result in via ``build_mixed_embeds`` (unchanged).

This module is **not a shipped example** — the projector is the client's code. It
exists only to prove + test that path end to end: ``EmbedsPassthroughEncoder``
decodes the client's embedding and passes it through unchanged (identity — it
assumes the client already sent hidden-dim embeds). See
``examples/custom_encoder/README.md`` for the client-facing contract.
"""

from __future__ import annotations

import base64
from typing import List, Optional

import torch
from safetensors.torch import load as _safetensors_load
from safetensors.torch import save as _safetensors_save

from dynamo.vllm.multimodal_utils.vision_encoder_backend import Preprocessed
from examples.custom_encoder.qwen_vision_encoder import QwenVisionEncoderBackend

# Media type marking a Dynamo per-image embeddings payload (safetensors bytes,
# base64-encoded). This is the client↔server wire convention for PR6; PR7 will add
# a first-class ``image_embeds`` request field that carries the same bytes.
EMBEDS_DATA_URI_PREFIX = "data:application/x-dynamo-embeds;base64,"
_EMBEDS_KEY = "embeds"


def encode_embeds_data_uri(embeds: torch.Tensor) -> str:
    """Serialize a 2D ``(n_tokens, hidden)`` tensor as a safetensors base64 ``data:`` URI.

    The client-side contract: compute your per-image embeddings, encode them with
    this, and send the returned string as the ``image_url`` of an image content
    part. safetensors carries dtype + shape and never executes code on load.
    """
    if embeds.dim() != 2:
        raise ValueError(f"embeds must be 2D (n_tokens, hidden), got {embeds.dim()}D")
    blob = _safetensors_save({_EMBEDS_KEY: embeds.contiguous().cpu()})
    return EMBEDS_DATA_URI_PREFIX + base64.b64encode(blob).decode("ascii")


def decode_embeds_data_uri(url: str) -> torch.Tensor:
    """Inverse of :func:`encode_embeds_data_uri`: ``data:`` URI → CPU ``(n_tokens, hidden)``.

    Raises ``ValueError`` on a non-``data:`` URL (e.g. an http URL) or a malformed
    payload, so a bad input fails only that image (the A5 barrier turns it into a
    clean request-level error).
    """
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ValueError(
            f"expected an embeds data: URI starting with 'data:', got {url[:48]!r}"
        )
    _, _, b64 = url.partition(",")
    if not b64:
        raise ValueError("malformed embeds data: URI (no base64 payload after ',')")
    try:
        blob = base64.b64decode(b64, validate=True)
        tensors = _safetensors_load(blob)
    except Exception as exc:  # noqa: BLE001 — bad input fails just this image
        raise ValueError(f"failed to decode embeds data: URI: {exc}") from exc
    if _EMBEDS_KEY not in tensors:
        raise ValueError(f"embeds payload missing '{_EMBEDS_KEY}' tensor key")
    return tensors[_EMBEDS_KEY]


class EmbedsPassthroughEncoder(QwenVisionEncoderBackend):
    """Decode client embeddings and pass them through unchanged (identity).

    Test-only stub. A real client backend would **project** (``embed_dim`` →
    ``lm_hidden_dim``) inside ``forward_batch``; this stub assumes the client
    already sent hidden-dim embeds, so the LM reads them directly through the
    mixed-embeds splice. Eager + pass-through (no batch cap, no graph ladder).
    """

    buckets = None
    max_batch_cost = None  # pass-through: the whole drained batch in one forward

    def preprocess(self, image_url: str) -> Preprocessed:
        """Off-thread: decode the client's safetensors ``data:`` URI to a CPU tensor."""
        embeds = decode_embeds_data_uri(image_url)
        if embeds.dim() != 2:
            raise ValueError(
                f"decoded embeds must be 2D (n_tokens, hidden), got {embeds.dim()}D"
            )
        return Preprocessed(item=embeds)

    def forward_batch(
        self, items: List[torch.Tensor], target_bucket: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Identity: return one tensor per item, unchanged (a real client projects here)."""
        return list(items)
