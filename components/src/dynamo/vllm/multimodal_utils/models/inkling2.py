# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image preprocessing and placeholder adaptation for Inkling 2 checkpoints."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from PIL import Image

from dynamo.llm.exceptions import InvalidArgument

INKLING2_MODEL_TYPES: Final = frozenset({"inkling2", "inkling2_loop"})
_IMAGE_MARKER: Final = "<|content_image|>"
_IMAGE_PAD: Final = "<|unused_200054|>"


def inkling2_rgb(image: Image.Image) -> Image.Image:
    """Composite transparency onto an 8x8 gray checkerboard without EXIF rotation."""
    if not image.has_transparency_data:
        return image.convert("RGB")
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint16)
    height, width, _ = rgba.shape
    rows, cols = np.ogrid[:height, :width]
    background = np.where(((cols // 8) + (rows // 8)) % 2 == 0, 204, 153).astype(
        np.uint16
    )
    alpha = rgba[:, :, 3:4]
    blended = (
        rgba[:, :, :3] * alpha + background[:, :, None] * (255 - alpha) + 127
    ) // 255
    return Image.fromarray(blended.astype(np.uint8), mode="RGB")


@dataclass(frozen=True)
class Inkling2ImagePlaceholders:
    """Strip template-emitted ``<|unused_200054|>`` seed pads before vLLM runs.

    Hugging Face's ``chat_template.jinja`` emits ``<|content_image|><|unused_200054|>``
    per image for ``Inkling2Processor``, whereas vLLM's
    ``Inkling2MultiModalProcessor`` replaces ``<|content_image|>`` with
    ``<|content_image|>`` followed by ``N`` copies of ``<|unused_200054|>``.
    Without stripping the template seed pad, each image leaves an extra
    unembedded ``<|unused_200054|>`` token in the prompt.
    """

    marker_id: int
    pad_id: int

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> Inkling2ImagePlaceholders:
        marker_ids = list(tokenizer.encode(_IMAGE_MARKER, add_special_tokens=False))
        pad_ids = list(tokenizer.encode(_IMAGE_PAD, add_special_tokens=False))
        if len(marker_ids) != 1 or type(marker_ids[0]) is not int:
            raise ValueError(
                "Inkling 2 <|content_image|> must encode to a single integer token ID"
            )
        if len(pad_ids) != 1 or type(pad_ids[0]) is not int:
            raise ValueError(
                "Inkling 2 <|unused_200054|> must encode to a single integer token ID"
            )
        return cls(marker_id=marker_ids[0], pad_id=pad_ids[0])

    def strip_seed_pads(self, token_ids: list[int], images: Sequence[Any]) -> list[int]:
        if not images:
            return token_ids

        try:
            first = token_ids.index(self.marker_id)
        except ValueError:
            return token_ids

        positions = [first]
        while True:
            try:
                positions.append(token_ids.index(self.marker_id, positions[-1] + 1))
            except ValueError:
                break

        if len(positions) != len(images):
            raise InvalidArgument(
                f"Inkling 2 prompt carries {len(positions)} <|content_image|> "
                f"token(s) but {len(images)} image(s) were supplied; refusing to adapt."
            )

        has_seed_pad = [
            pos + 1 < len(token_ids) and token_ids[pos + 1] == self.pad_id
            for pos in positions
        ]
        if not any(has_seed_pad):
            return token_ids
        if not all(has_seed_pad):
            raise InvalidArgument(
                "Inkling 2 prompt mixes bare <|content_image|> markers and "
                "<|unused_200054|> seed pads; refusing to adapt."
            )
        if any(
            pos + 2 < len(token_ids) and token_ids[pos + 2] == self.pad_id
            for pos in positions
        ):
            return token_ids

        adapted: list[int] = []
        previous = 0
        for position in positions:
            adapted.extend(token_ids[previous : position + 1])
            previous = position + 2
        adapted.extend(token_ids[previous:])
        return adapted
