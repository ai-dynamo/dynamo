# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert Dynamo's backend-neutral processed-media IR to vLLM inputs."""

from __future__ import annotations

from typing import Any

import torch
from transformers import BatchFeature
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    PlaceholderRange,
)

from dynamo.common.multimodal.model_prompt import expand_processed_media_prompt
from dynamo.common.multimodal.processed_media import ProcessedMedia


def _combine_fields(
    media_items: list[ProcessedMedia],
) -> tuple[dict[str, Any], dict[str, Any]]:
    names = set(media_items[0].fields)
    if any(set(media.fields) != names for media in media_items[1:]):
        raise ValueError("Processed media items must have identical field sets")
    modality = media_items[0].modality
    hf_inputs: dict[str, Any] = {}
    configs: dict[str, Any] = {}
    for name in sorted(names):
        fields = [media.fields[name] for media in media_items]
        first = fields[0]
        if any(
            (field.layout, field.keep_on_host, field.forward)
            != (first.layout, first.keep_on_host, first.forward)
            for field in fields[1:]
        ):
            raise ValueError(f"Processed field {name!r} has inconsistent metadata")
        tensors = [torch.from_numpy(field.value) for field in fields]
        kind = first.layout["kind"]
        if kind == "flat":
            sizes_key = first.layout["sizes_key"]
            if any(sizes_key not in media.fields for media in media_items):
                raise ValueError(
                    f"Processed field {name!r} references missing sizes field "
                    f"{sizes_key!r}"
                )
            size_tensors = [
                torch.from_numpy(media.fields[sizes_key].value).reshape(-1)
                for media in media_items
            ]
            if len(media_items) == 1:
                sizes = size_tensors[0]
                hf_inputs[name] = tensors[0]
            else:
                sizes = torch.cat(size_tensors)
                hf_inputs[name] = torch.cat(tensors, dim=0)
            configs[name] = MultiModalFieldConfig.flat_from_sizes(modality, sizes)
        elif kind == "batched":
            if any(tensor.shape[0] != 1 for tensor in tensors):
                raise ValueError(
                    f"Processed field {name!r} must contain one batched item"
                )
            item_tensors = [tensor[0] for tensor in tensors]
            if len(media_items) == 1:
                hf_inputs[name] = tensors[0]
            elif all(tensor.shape == item_tensors[0].shape for tensor in item_tensors):
                hf_inputs[name] = torch.stack(item_tensors)
            else:
                hf_inputs[name] = item_tensors
            configs[name] = MultiModalFieldConfig.batched(
                modality, keep_on_cpu=first.keep_on_host
            )
        elif kind == "shared":
            if any(not torch.equal(tensors[0], tensor) for tensor in tensors[1:]):
                raise ValueError(
                    f"Shared processed field {name!r} differs between items"
                )
            hf_inputs[name] = tensors[0]
            configs[name] = MultiModalFieldConfig.shared(
                modality,
                len(media_items),
                keep_on_cpu=first.keep_on_host,
            )
        else:
            raise ValueError(f"Unknown processed field layout: {kind!r}")
    forwarded = {
        name: value
        for name, value in hf_inputs.items()
        if media_items[0].fields[name].forward
    }
    forwarded_configs = {
        name: value
        for name, value in configs.items()
        if media_items[0].fields[name].forward
    }
    return forwarded, forwarded_configs


def build_processed_media_input(
    *,
    prompt_token_ids: list[int],
    media_items: list[ProcessedMedia],
    engine_client: Any,
    mm_processor_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    if not media_items:
        raise ValueError("At least one processed media item is required")
    modalities = {media.modality for media in media_items}
    if len(modalities) != 1:
        raise ValueError("A processed-media batch must contain one modality")
    kwargs = mm_processor_kwargs or {}
    if set(kwargs) - {"do_sample_frames"} or kwargs.get("do_sample_frames") not in (
        None,
        False,
    ):
        raise ValueError(
            "Frontend-processed media does not support these mm_processor_kwargs"
        )

    input_processor = engine_client.input_processor
    tokenizer = input_processor.get_tokenizer()
    hf_config = engine_client.vllm_config.model_config.hf_config
    expanded = expand_processed_media_prompt(
        prompt_token_ids, media_items, tokenizer, hf_config
    )
    hf_inputs, field_configs = _combine_fields(media_items)
    mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
        BatchFeature(data=hf_inputs, tensor_type=None), field_configs
    )
    modality = media_items[0].modality
    kwargs_dict = {modality: list(mm_kwargs[modality])}
    hashes = {
        modality: [
            value.ljust(64, "0")
            for media in media_items
            for value in media.content_hashes
        ]
    }
    placeholders = [
        PlaceholderRange(
            offset=value.offset,
            length=value.length,
            is_embed=None
            if all(value.mask)
            else torch.tensor(value.mask, dtype=torch.bool),
        )
        for value in expanded.ranges
    ]
    input_processor.inject_into_mm_cache(hashes, kwargs_dict)
    return {
        "type": "multimodal",
        "prompt_token_ids": expanded.token_ids,
        "mm_kwargs": kwargs_dict,
        "mm_hashes": hashes,
        "mm_placeholders": {modality: placeholders},
    }
