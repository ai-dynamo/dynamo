# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Any, Dict

import safetensors
import torch

import dynamo.nixl_connect as connect

from .model import construct_mm_data
from .protocol import MultiModalGroup

logger = logging.getLogger(__name__)

TRANSFER_LOCAL = int(os.getenv("TRANSFER_LOCAL", 1))


async def load_embeddings(
    mi: MultiModalGroup,
    embeddings_dtype: torch.dtype,
    embeddings_device: str,
    connector: connect.Connector | None,
) -> torch.Tensor:
    """Load pre-computed embedding tensor via local safetensors or NIXL RDMA.

    Args:
        mi: A single MultiModalGroup whose ``serialized_request`` field
            contains either a local file path or NIXL RDMA metadata.
        embeddings_dtype: Torch dtype for the tensor (used for RDMA path).
        embeddings_device: Device string for the tensor (used for RDMA path).
        connector: NIXL Connector for RDMA reads (required when TRANSFER_LOCAL=0).

    Returns:
        The embedding tensor loaded into CPU memory.
    """
    if TRANSFER_LOCAL:
        logger.info("PD: Loading local safetensors file")
        return safetensors.torch.load_file(mi.serialized_request)["ec_cache"]

    embeddings = torch.empty(
        mi.embeddings_shape,
        dtype=embeddings_dtype,
        device=embeddings_device,
    )
    descriptor = connect.Descriptor(embeddings)

    if descriptor is None:
        raise RuntimeError(
            "Descriptor is None in PD worker - cannot process embeddings"
        )

    read_op = await connector.begin_read(mi.serialized_request, descriptor)
    await read_op.wait_for_completion()
    return embeddings


def accumulate_embeddings(
    multi_modal_data: Dict[str, Any],
    model: str,
    embeddings_dtype: torch.dtype,
    embeddings: torch.Tensor,
    image_grid_thw,
) -> None:
    """Construct model-specific mm_data from embeddings and merge into the
    accumulated ``multi_modal_data`` dict (mutated in-place).

    Handles both video (numpy conversion) and image modalities, including
    the Qwen-VL dict-style embeddings with ``image_embeds`` + ``image_grid_thw``.
    """
    if "video" in model.lower():
        video_numpy = embeddings.numpy()
        mm_data = construct_mm_data(
            model,
            embeddings_dtype,
            video_numpy=video_numpy,
        )
        multi_modal_data["video"].append(mm_data["video"])
        return

    mm_data = construct_mm_data(
        model,
        embeddings_dtype,
        image_embeds=embeddings,
        image_grid_thw=image_grid_thw,
    )

    if isinstance(mm_data["image"], dict):
        # Qwen-VL style: dict with image_embeds + image_grid_thw tensors
        if multi_modal_data["image"] == []:
            multi_modal_data["image"] = mm_data["image"]
        else:
            # [gluo FIXME] need to understand how Qwen consumes multi-image embeddings
            multi_modal_data["image"]["image_embeds"] = torch.cat(
                (
                    multi_modal_data["image"]["image_embeds"],
                    mm_data["image"]["image_embeds"],
                )
            )
            multi_modal_data["image"]["image_grid_thw"] = torch.cat(
                (
                    multi_modal_data["image"]["image_grid_thw"],
                    mm_data["image"]["image_grid_thw"],
                )
            )
    else:
        # Plain tensor embeddings
        logger.info(f"Get embedding of shape {mm_data['image'].shape}")
        # [gluo FIXME] embedding with multiple images?
        if multi_modal_data["image"] == []:
            multi_modal_data["image"] = mm_data["image"]
        else:
            multi_modal_data["image"] = torch.cat(
                (multi_modal_data["image"], mm_data["image"])
            )
