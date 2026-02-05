# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared utility for batch loading images from multimodal data items."""

import asyncio
import logging
from typing import Any, Dict, Final, List, Optional

import dynamo.nixl_connect as nixl_connect
from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.common.utils.media_nixl import read_decoded_media_via_nixl

logger = logging.getLogger(__name__)

# Constants for multimodal data variants
URL_VARIANT_KEY: Final = "Url"
DECODED_VARIANT_KEY: Final = "Decoded"


async def load_image_batch(
    image_mm_items: List[Dict[str, Any]],
    image_loader: ImageLoader,
    enable_frontend_decoding: bool = False,
    nixl_connector: Optional[nixl_connect.Connector] = None,
) -> List[Any]:
    """
    Load a batch of images from multimodal data items.

    Supports two paths:
    1. Url variant: Download and decode image from URL (default)
    2. Decoded variant: Read pre-decoded image via NIXL RDMA (requires enable_frontend_decoding=True)

    Args:
        image_mm_items: List of multimodal data items for images
        image_loader: ImageLoader instance to use for loading
        enable_frontend_decoding: If True, enables NIXL RDMA for decoded images
        nixl_connector: NIXL connector for frontend decoding (required if enable_frontend_decoding=True)

    Returns:
        List of loaded image data

    Raises:
        Exception: If any image fails to load
        ValueError: If enable_frontend_decoding=True but nixl_connector is None
    """
    image_futures = []

    for item in image_mm_items:
        if isinstance(item, dict) and URL_VARIANT_KEY in item:
            # URL path: download and decode in Python backend
            url = item[URL_VARIANT_KEY]
            image_futures.append(image_loader.load_image(url))
            logger.debug(f"Preparing to load image from URL: {url[:80]}...")
        elif isinstance(item, dict) and DECODED_VARIANT_KEY in item:
            if enable_frontend_decoding:
                if nixl_connector is None:
                    logger.error(
                        "Frontend decoding enabled but nixl_connector not provided. "
                        "Caller must pass an initialized NIXL connector."
                    )
                    raise ValueError(
                        "nixl_connector required when enable_frontend_decoding=True"
                    )

                metadata = item[DECODED_VARIANT_KEY]
                image_futures.append(
                    read_decoded_media_via_nixl(nixl_connector, metadata)
                )
            else:
                logger.error(
                    "Received Decoded multimodal data but enable_frontend_decoding=False. "
                    "Set enable_frontend_decoding=True to enable NIXL RDMA image transfer."
                )
                raise ValueError("Could not load decoded media from frontend")

    # Process images in parallel
    results = await asyncio.gather(*image_futures, return_exceptions=True)
    loaded_images = []
    collective_exceptions = ""
    for media_item, result in zip(image_mm_items, results):
        if isinstance(result, Exception):
            source = media_item.get(URL_VARIANT_KEY, "decoded")
            logger.error(f"Failed to load image from {source[:80]}...: {result}")
            collective_exceptions += (
                f"Failed to load image from {source[:80]}...: {result}\n"
            )
            continue
        loaded_images.append(result)

    if collective_exceptions:
        raise Exception(collective_exceptions)

    return loaded_images
