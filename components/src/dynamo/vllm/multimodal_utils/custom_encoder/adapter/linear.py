# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Linear embedding adapter for the aggregated CustomEncoder path.

The encoder returns only the visual token embeddings; this module builds the
inputs for vLLM's mixed ``EmbedsPrompt`` mode (``prompt_token_ids`` +
``prompt_is_token_ids`` + ``prompt_embeds``):

    prompt_token_ids    = [ text  ... <img> <img> <img> ...  text  ]
    prompt_is_token_ids = [ True  ... False False False ...  True  ]
    prompt_embeds       = [ zeros ...  e0    e1    e2   ... zeros  ]   (seq_len, hidden)

One image occupies a **contiguous run** of ``False`` positions — the single
placeholder token is expanded to the encoder tensor's row count (3 here), and
that image's embeds (``e0,e1,e2``) fill exactly those rows. vLLM embeds the
``True`` (text) positions itself with the model's real embedding table and
substitutes each ``False`` (image) row from ``prompt_embeds`` in the forward
pass. Dynamo therefore only fills the image rows — text rows stay zero (they are
overwritten) and no LM embedding weight is needed on the Dynamo side.

The contract is **one placeholder token per image**: each occurrence of the
placeholder token in ``prompt_token_ids`` is one image slot, matched
positionally to the encoder tensors, and the single placeholder is expanded to
the tensor's row count so the encoder dictates the span length (mirroring
vLLM's own placeholder expansion); this keeps a mismatch between the tokenizer's
placeholder count and the encoder's visual-token count from raising.  The chat
template therefore emits exactly one placeholder token per image and needs no
separator between consecutive images.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from vllm.inputs import EmbedsPrompt, TokensPrompt

from dynamo.vllm.multimodal_utils.custom_encoder.adapter.base import (
    CustomEncoderAdapter,
)
from dynamo.vllm.multimodal_utils.custom_encoder.backend.base import (
    VisionEncoderBackend,
)
from dynamo.vllm.multimodal_utils.model_config import _hidden_size, _is_multimodal_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LinearVisualPrompt:
    """Compact mixed-prompt layout plus only its visual embedding rows."""

    visual_embeds: torch.Tensor
    prompt_token_ids: list[int]
    prompt_is_token_ids: list[bool]
    image_token_id: int


def build_mixed_layout(
    token_ids: list[int],
    img_tensors: list[torch.Tensor],
    placeholder_id: int,
) -> LinearVisualPrompt:
    """Build a mixed-prompt layout without allocating its dense text rows.

    Args:
        token_ids: The full prompt token IDs (text + one placeholder token per
            image).
        img_tensors: Per-image visual token tensors, each ``(n_tokens, hidden)``,
            in prompt order.
        placeholder_id: The token ID marking image positions.

    Returns:
        A compact prompt containing the concatenated visual rows, expanded
        token IDs, and the mixed-prompt token mask. Visual rows are ordered
        exactly like the ``False`` positions in the mask.

    Raises:
        ValueError: if ``img_tensors`` is empty, the number of placeholder
            tokens does not equal the number of image tensors, or the tensors
            are not 2D with a consistent hidden dim.
    """
    if not img_tensors:
        raise ValueError("img_tensors must not be empty")

    positions = [i for i, tid in enumerate(token_ids) if tid == placeholder_id]
    if len(positions) != len(img_tensors):
        raise ValueError(
            f"placeholder tokens ({len(positions)}) != image tensors "
            f"({len(img_tensors)}) for placeholder token {placeholder_id} "
            f"in sequence of length {len(token_ids)}"
        )

    # Check tensor 0 is 2D before reading its hidden dim, so a 1D encoder output
    # raises a clear ValueError here instead of an opaque IndexError on shape[1].
    if img_tensors[0].dim() != 2:
        raise ValueError(
            f"image tensor 0 has shape {tuple(img_tensors[0].shape)}; expected "
            "2D (n_tokens, hidden)"
        )
    hidden = img_tensors[0].shape[1]
    dtype = img_tensors[0].dtype
    # Validate shapes before scattering so a bad encoder output raises a clear
    # ValueError here (caught by the caller) instead of an opaque RuntimeError
    # from the row-copy below on a width mismatch.
    for i, tensor in enumerate(img_tensors):
        if tensor.dim() != 2 or tensor.shape[1] != hidden:
            raise ValueError(
                f"image tensor {i} has shape {tuple(tensor.shape)}; expected "
                f"2D with hidden dim {hidden} (from image tensor 0)"
            )
        # A (0, hidden) tensor passes the 2D/hidden checks but would erase the
        # image's placeholder token entirely, silently dropping the image from
        # the prompt. An encoder returning no visual tokens for an image is a
        # bug — fail loudly instead.
        if tensor.shape[0] == 0:
            raise ValueError(
                f"image tensor {i} has 0 rows (shape {tuple(tensor.shape)}); the "
                "encoder returned no visual tokens for an image"
            )
        if tensor.dtype != dtype:
            raise ValueError(
                f"image tensor {i} has dtype {tensor.dtype}; expected {dtype} "
                "from image tensor 0"
            )
        # forward_batch must fence + copy to CPU before returning, so the scatter
        # below is a plain assignment into the CPU prompt_embeds buffer. Fail loud
        # here instead of an opaque cross-device error on the row-copy.
        if tensor.device.type != "cpu":
            raise ValueError(
                f"image tensor {i} is on {tensor.device}; forward_batch must "
                "return CPU tensors"
            )

    # Build the token-id / mask layout. The visual rows stay compact and in the
    # same order as the False positions; a remote consumer can transfer these
    # rows without also moving zero-filled text rows.
    out_token_ids: list[int] = []
    is_token_ids: list[bool] = []

    def _emit_text(text_ids: list[int]) -> None:
        if not text_ids:
            return
        out_token_ids.extend(text_ids)
        is_token_ids.extend([True] * len(text_ids))

    cursor = 0
    for pos, tensor in zip(positions, img_tensors):
        _emit_text(token_ids[cursor:pos])
        n = tensor.shape[0]
        out_token_ids.extend([placeholder_id] * n)
        is_token_ids.extend([False] * n)
        cursor = pos + 1
    _emit_text(token_ids[cursor:])

    visual_embeds = (
        img_tensors[0].contiguous()
        if len(img_tensors) == 1
        else torch.cat(img_tensors, dim=0).contiguous()
    )
    return LinearVisualPrompt(
        visual_embeds=visual_embeds,
        prompt_token_ids=out_token_ids,
        prompt_is_token_ids=is_token_ids,
        image_token_id=placeholder_id,
    )


def build_mixed_embeds(
    token_ids: list[int],
    img_tensors: list[torch.Tensor],
    placeholder_id: int,
) -> tuple[torch.Tensor, list[int], list[bool]]:
    """Build the dense mixed prompt consumed by in-process vLLM.

    The compact layout helper is shared with the remote route. Only this inline
    path allocates the zero-filled text rows before handing the prompt to vLLM.
    """
    layout = build_mixed_layout(token_ids, img_tensors, placeholder_id)

    seq_len = len(layout.prompt_token_ids)
    hidden = layout.visual_embeds.shape[1]
    # CPU tensor: vLLM's renderer forces prompt_embeds to CPU anyway.
    prompt_embeds = torch.zeros(seq_len, hidden, dtype=layout.visual_embeds.dtype)
    visual_mask = torch.tensor(
        [not is_token_id for is_token_id in layout.prompt_is_token_ids],
        dtype=torch.bool,
    )
    prompt_embeds[visual_mask] = layout.visual_embeds

    logger.debug(
        "[custom_embeds] images=%d seq_len=%d hidden=%d dtype=%s",
        len(img_tensors),
        seq_len,
        hidden,
        layout.visual_embeds.dtype,
    )
    return prompt_embeds, layout.prompt_token_ids, layout.prompt_is_token_ids


class LinearEmbedsAdapter(CustomEncoderAdapter[torch.Tensor]):
    """Build mixed ``EmbedsPrompt`` inputs for a text-only decoder."""

    def __init__(
        self,
        backend: VisionEncoderBackend[Any, Any, torch.Tensor],
        model_config: Any,
        engine_args: Any,
    ) -> None:
        if model_config is None:
            raise ValueError("CustomEncoder requires the resolved vLLM ModelConfig")
        if _is_multimodal_model(model_config):
            raise ValueError(
                "CustomEncoder does not yet support this multimodal decoder; "
                "the linear EmbedsPrompt adapter is only valid for text-only models"
            )
        if not getattr(engine_args, "enable_prompt_embeds", False):
            raise ValueError(
                "text-only CustomEncoder output requires --enable-prompt-embeds"
            )
        image_token_id = getattr(backend, "image_token_id", None)
        if not isinstance(image_token_id, int) or isinstance(image_token_id, bool):
            raise ValueError(
                "text-only CustomEncoder output requires an integer image_token_id"
            )

        self._image_token_id = image_token_id
        self._hidden_size = _hidden_size(model_config)
        model_dtype = getattr(model_config, "dtype", None)
        self._dtype = model_dtype if isinstance(model_dtype, torch.dtype) else None

    def prepare_compact_prompt(
        self,
        token_ids: list[int],
        artifacts: Sequence[torch.Tensor],
    ) -> LinearVisualPrompt:
        """Validate artifacts and retain only rows that cross a remote hop."""
        rows = self._validated_rows(artifacts)
        return build_mixed_layout(token_ids, rows, self._image_token_id)

    def _validated_rows(self, artifacts: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Validate the decoder-facing tensor contract once for both routes."""
        rows = list(artifacts)
        for index, tensor in enumerate(rows):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    "text-only CustomEncoder must return tensors; "
                    f"result {index} is {type(tensor).__name__}"
                )
            if tensor.dim() != 2 or tensor.shape[1] != self._hidden_size:
                raise ValueError(
                    f"image tensor {index} has shape {tuple(tensor.shape)}; "
                    f"expected 2D with decoder hidden size {self._hidden_size}"
                )
            if self._dtype is not None and tensor.dtype != self._dtype:
                raise ValueError(
                    f"image tensor {index} has dtype {tensor.dtype}; "
                    f"expected decoder dtype {self._dtype}"
                )
        return rows

    def prepare_prompt(
        self,
        token_ids: list[int],
        artifacts: Sequence[torch.Tensor],
    ) -> EmbedsPrompt | TokensPrompt:
        """Build a mixed prompt from per-image visual embedding tensors.

        Each artifact must be a CPU tensor shaped
        ``(n_visual_tokens, decoder_hidden_size)`` with the decoder's dtype.
        Artifacts must appear in the same order as the image placeholders in
        ``token_ids``.
        """
        rows = self._validated_rows(artifacts)

        prompt_embeds, prompt_token_ids, prompt_is_token_ids = build_mixed_embeds(
            token_ids, rows, self._image_token_id
        )
        return EmbedsPrompt(
            prompt_embeds=prompt_embeds,
            prompt_token_ids=prompt_token_ids,
            prompt_is_token_ids=prompt_is_token_ids,
        )
