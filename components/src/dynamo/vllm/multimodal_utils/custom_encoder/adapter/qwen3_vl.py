# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native external-multimodal adapter for Qwen3-VL decoders."""

from dataclasses import dataclass
from typing import Sequence

import torch
from vllm.inputs import TokensPrompt

from dynamo.vllm.multimodal_utils.custom_encoder.adapter.base import (
    CustomEncoderAdapter,
)

_QWEN3_VL_ARCHITECTURES = frozenset({"Qwen3VLForConditionalGeneration"})


@dataclass(frozen=True)
class Qwen3VLImageEncoding:
    """Packed Qwen3-VL image features and their pre-merge grid."""

    embeddings: torch.Tensor
    grid_thw: tuple[int, int, int]


class Qwen3VLNativeAdapter(CustomEncoderAdapter[Qwen3VLImageEncoding]):
    """Build a native external-MM ``TokensPrompt`` for Qwen3-VL."""

    def prepare_prompt(
        self,
        token_ids: list[int],
        artifacts: Sequence[Qwen3VLImageEncoding],
    ) -> TokensPrompt:
        return TokensPrompt(
            prompt_token_ids=token_ids,
            multi_modal_data={
                "image": {
                    "image_embeds": torch.cat(
                        [artifact.embeddings for artifact in artifacts], dim=0
                    ),
                    "image_grid_thw": torch.tensor(
                        [artifact.grid_thw for artifact in artifacts],
                        dtype=torch.int64,
                        device="cpu",
                    ),
                }
            },
        )
