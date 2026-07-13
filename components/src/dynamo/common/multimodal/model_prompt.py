# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-specific prompt expansion over backend-neutral processed media."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dynamo.common.multimodal.processed_media import ProcessedMedia


@dataclass(frozen=True)
class EmbeddingRange:
    offset: int
    length: int
    mask: list[bool]


@dataclass(frozen=True)
class ExpandedPrompt:
    token_ids: list[int]
    ranges: list[EmbeddingRange]


def _field(media: ProcessedMedia, name: str) -> Any:
    try:
        return media.fields[name].value
    except KeyError as exc:
        raise ValueError(f"Processed media is missing required field {name!r}") from exc


def expand_processed_media_prompt(
    prompt_ids: list[int],
    media_items: list[ProcessedMedia],
    tokenizer: Any,
    hf_config: Any,
) -> ExpandedPrompt:
    """Dispatch to the model family without importing an inference backend."""
    model_type = hf_config.model_type
    expander = _PROMPT_EXPANDERS.get(model_type)
    if expander is None:
        raise ValueError(
            f"No processed-media prompt specification for model type {model_type!r}"
        )
    return expander(prompt_ids, media_items, tokenizer, hf_config)


def _expand_qwen3_vl(
    prompt_ids: list[int],
    media_items: list[ProcessedMedia],
    tokenizer: Any,
    config: Any,
) -> ExpandedPrompt:
    merge_size = int(config.vision_config.spatial_merge_size)
    replacements: list[list[int]] = []
    for media in media_items:
        if media.modality != "video":
            raise ValueError("Qwen3-VL video prompt spec received non-video media")
        grid = _field(media, "video_grid_thw").reshape(-1).tolist()
        timestamps = _field(media, "timestamps").reshape(-1).tolist()
        if len(grid) != 3 or len(timestamps) != int(grid[0]):
            raise ValueError(
                "Qwen3-VL processed video has inconsistent grid/timestamps"
            )
        t, h, w = map(int, grid)
        expected = t * h * w // (merge_size**2)
        if media.feature_token_counts != [expected]:
            raise ValueError(
                "Qwen3-VL processed video feature count does not match its grid"
            )
        replacement: list[int] = []
        tokens_per_frame = h * w // (merge_size**2)
        for timestamp in timestamps:
            replacement.extend(
                tokenizer.encode(
                    f"<{float(timestamp):.1f} seconds>", add_special_tokens=False
                )
            )
            replacement.append(int(config.vision_start_token_id))
            replacement.extend([int(config.video_token_id)] * tokens_per_frame)
            replacement.append(int(config.vision_end_token_id))
        replacements.append(replacement)

    target = (
        int(config.vision_start_token_id),
        int(config.video_token_id),
        int(config.vision_end_token_id),
    )
    output: list[int] = []
    ranges: list[EmbeddingRange] = []
    replacement_index = 0
    index = 0
    while index < len(prompt_ids):
        if tuple(prompt_ids[index : index + 3]) == target:
            if replacement_index >= len(replacements):
                raise ValueError("Prompt contains more video placeholders than videos")
            replacement = replacements[replacement_index]
            ranges.append(
                EmbeddingRange(
                    offset=len(output),
                    length=len(replacement),
                    mask=[token == config.video_token_id for token in replacement],
                )
            )
            output.extend(replacement)
            replacement_index += 1
            index += 3
        else:
            output.append(prompt_ids[index])
            index += 1
    if replacement_index != len(replacements):
        raise ValueError(
            f"Prompt/video count mismatch: found {replacement_index} placeholders "
            f"for {len(replacements)} videos"
        )
    return ExpandedPrompt(output, ranges)


_PROMPT_EXPANDERS = {"qwen3_vl": _expand_qwen3_vl}
