# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image placeholders of Mistral-format checkpoints served by vLLM."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class MistralImagePlaceholders:
    """Expand one ``[IMG]`` per image into that image's token grid.

    vLLM serves Mistral-format checkpoints (``--tokenizer-mode mistral``) with a
    Pixtral processor that never expands image placeholders: it only checks
    that the prompt already holds each image's full
    ``[IMG]``/``[IMG_BREAK]``/``[IMG_END]`` grid. The Rust frontend's Mistral
    renderer emits one ``[IMG]`` per image because the grid depends on the
    image size, so the worker, which loads the image, sizes each grid with the
    engine's own mistral-common image encoder.
    """

    encoder: Any
    img: int
    img_break: int
    img_end: int

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> Optional[MistralImagePlaceholders]:
        """Return the placeholders of a Mistral tokenizer with an image encoder."""
        encoder = getattr(getattr(tokenizer, "instruct", None), "mm_encoder", None)
        special_ids = getattr(encoder, "special_ids", None)
        img, img_break, img_end = (
            getattr(special_ids, name, None) for name in ("img", "img_break", "img_end")
        )
        if not all(type(token_id) is int for token_id in (img, img_break, img_end)):
            return None
        return cls(encoder=encoder, img=img, img_break=img_break, img_end=img_end)

    def expand(self, token_ids: list[int], images: Sequence[Any]) -> list[int]:
        """Replace each ``[IMG]`` with its image's grid.

        A prompt that already carries ``[IMG_END]`` was expanded upstream (the
        Python frontend's chat path) and is returned as is.
        """
        if not images or self.img_end in token_ids:
            return token_ids

        try:
            first = token_ids.index(self.img)
        except ValueError:
            return token_ids
        positions = [first]
        while True:
            try:
                positions.append(token_ids.index(self.img, positions[-1] + 1))
            except ValueError:
                break

        if len(positions) != len(images):
            raise ValueError(
                f"Mistral prompt carries {len(positions)} [IMG] placeholder(s) "
                f"but {len(images)} image(s) were supplied; refusing to expand."
            )

        expanded: list[int] = []
        previous = 0
        for position, image in zip(positions, images):
            if image is None:
                raise ValueError(
                    "Mistral image placeholders are sized from the image; "
                    "UUID-only image slots cannot be expanded."
                )
            ncols, nrows = self.encoder._image_to_num_tokens(image)
            grid = ([self.img] * ncols + [self.img_break]) * nrows
            grid[-1] = self.img_end
            expanded.extend(token_ids[previous:position])
            expanded.extend(grid)
            previous = position + 1
        expanded.extend(token_ids[previous:])
        return expanded
